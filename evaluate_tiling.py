"""Check what the whole-slide path costs, using a stitched mosaic.

Stitches Leukemia-NFXZN test images into a pyramidal TIFF and scores the
sliding-window pipeline (tiling, background skip, coordinate mapping,
global NMS) against direct per-image prediction on the same pixels.
It measures the tiling machinery only: a mosaic of camera fields is not a
scanned slide, so scanner and staining shift are not covered.

Usage: python evaluate_tiling.py [--model PATH] [--grid 8] [--out DIR]
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

import inference_pipeline as ip
from config import settings
from training_pipeline import WarmStartTrainer, _to_box

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--model", default=str(settings.active_model_path))
parser.add_argument("--grid", type=int, default=8, help="images per mosaic side")
parser.add_argument("--out", default=None, help="working directory (default: temp)")
args = parser.parse_args()
S = Path(args.out or tempfile.mkdtemp(prefix="tiling_eval_"))
S.mkdir(parents=True, exist_ok=True)
MODEL = args.model
GRID = args.grid
CELL = 600
IOUS = torch.linspace(0.5, 0.95, 10)

rules = WarmStartTrainer().remapping_rules["leukemia-1"]
images = sorted(Path("leukemia-1/test/images").iterdir())[: GRID * GRID]

# Build the mosaic and its ground truth in slide pixels.
slide = np.full((GRID * CELL, GRID * CELL, 3), 255, np.uint8)
gt, per_image = [], []
for k, path in enumerate(images):
    r, c = divmod(k, GRID)
    im = Image.open(path).convert("RGB").resize((CELL, CELL))
    slide[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = np.asarray(im)
    rows = []
    for line in (path.parent.parent / "labels" / f"{path.stem}.txt").read_text().splitlines():
        box = _to_box(line.split()) if line.strip() else None
        if not box or int(box[0]) not in rules:
            continue
        x, y, w, h = (float(v) * CELL for v in box[1:])
        rows.append([rules[int(box[0])], c * CELL + x - w / 2, r * CELL + y - h / 2,
                     c * CELL + x + w / 2, r * CELL + y + h / 2])
    gt += rows
    per_image.append((path, r, c))

tif = S / "mosaic.tif"
with tifffile.TiffWriter(tif, bigtiff=False) as w:
    w.write(slide, tile=(256, 256), photometric="rgb", compression="jpeg",
            subifds=1, resolution=(1e4 / 0.25, 1e4 / 0.25), resolutionunit="CENTIMETER")
    w.write(slide[::2, ::2], tile=(256, 256), photometric="rgb", compression="jpeg",
            subfiletype=1)


def score(preds):
    """COCO-style mAP50 / mAP50-95 with Ultralytics' matching."""
    g = torch.tensor(gt, dtype=torch.float32)
    if not preds:
        return 0.0, 0.0
    p = torch.tensor(preds, dtype=torch.float32)  # x1 y1 x2 y2 conf cls
    iou = box_iou(g[:, 1:], p[:, :4])
    correct = np.zeros((len(p), len(IOUS)), bool)
    same = g[:, 0:1] == p[:, 5]
    for i, t in enumerate(IOUS):
        x = torch.nonzero((iou >= t) & same)
        if len(x):
            m = torch.cat((x, iou[x[:, 0], x[:, 1]][:, None]), 1).numpy()
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    res = ap_per_class(correct, p[:, 4].numpy(), p[:, 5].numpy(), g[:, 0].numpy())
    ap = res[5]
    return float(ap[:, 0].mean()), float(ap.mean())


model = YOLO(MODEL)

# Direct per-image prediction on identical pixels.
direct = []
for path, r, c in per_image:
    crop = Image.fromarray(slide[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL])
    for b in model.predict(crop, conf=0.001, verbose=False)[0].boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        direct.append([c * CELL + x1, r * CELL + y1, c * CELL + x2, r * CELL + y2,
                       float(b.conf), float(b.cls)])

# Sliding-window pipeline, capturing detections instead of writing SQLite.
settings.inference_conf_threshold = 0.001
settings.active_model_path = Path(MODEL)
ip.RAW_TILES_DIR = S / "wsi_tiles"
ip.RAW_TILES_DIR.mkdir(exist_ok=True)
captured = {}
pipe = ip.InferencePipeline.__new__(ip.InferencePipeline)
pipe.model, pipe.class_names, pipe.run_id = model, model.names, "eval"
pipe._commit_to_db = lambda name, tiles, dets: captured.update(tiles=tiles, dets=dets)
pipe.run(tif)
wsi = [d["box"] + [d["score"], d["class_idx"]] for d in captured["dets"]]

out = {
    "slide_px": slide.shape[:2], "images": len(per_image), "gt_boxes": len(gt),
    "tiles_scanned": len(captured["tiles"]),
    "direct": dict(zip(("mAP50", "mAP50-95"), score(direct))),
    "wsi_pipeline": dict(zip(("mAP50", "mAP50-95"), score(wsi))),
}
print(json.dumps(out, indent=1))
