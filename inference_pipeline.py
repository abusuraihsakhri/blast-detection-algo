"""Sliding-window WSI inference with overlap-aware global NMS."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from loguru import logger
from ultralytics import YOLO

from config import RAW_TILES_DIR, ensure_directories, settings
from database import SessionLocal, init_db
from models import Annotation, Tile

try:
    import tiffslide
except ImportError:
    tiffslide = None


def _axis_origins(length: int, tile_size: int, stride: int) -> List[int]:
    """Return origins that include both image boundaries without duplicates."""
    if length <= tile_size:
        return [0]
    last = length - tile_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


class WsiScanner:
    """Extract overlapping tissue tiles while preserving level-0 coordinates."""

    def __init__(self, wsi_path: str | Path):
        if tiffslide is None:
            raise RuntimeError(
                "tiffslide is required for WSI scanning. "
                "Install project dependencies first."
            )

        self.wsi_path = Path(wsi_path).expanduser().resolve()
        if not self.wsi_path.exists() or not self.wsi_path.is_file():
            raise FileNotFoundError(f"WSI not found: {self.wsi_path}")

        self.slide = tiffslide.TiffSlide(self.wsi_path)
        self.level = settings.magnification_level
        if self.level >= len(self.slide.level_dimensions):
            self.slide.close()
            raise ValueError(
                f"Requested pyramid level {self.level}, but slide has "
                f"{len(self.slide.level_dimensions)} level(s)."
            )

        self.width, self.height = self.slide.level_dimensions[self.level]
        self.downsample = float(self.slide.level_downsamples[self.level])
        self.tile_size = settings.tile_size
        self.stride = max(
            1,
            int(round(self.tile_size * (1.0 - settings.inference_overlap_pct))),
        )
        self.x_origins = _axis_origins(self.width, self.tile_size, self.stride)
        self.y_origins = _axis_origins(self.height, self.tile_size, self.stride)
        logger.info(
            "Loaded WSI '{}' — {}x{} at level {} (downsample {:.3f})",
            self.wsi_path.name,
            self.width,
            self.height,
            self.level,
            self.downsample,
        )

    def close(self) -> None:
        self.slide.close()

    def _is_empty_tissue(self, img: Image.Image) -> bool:
        gray = np.asarray(img.convert("L"))
        background_pct = float(np.mean(gray > 220))
        return background_pct > settings.background_threshold

    def extract_patches(
        self,
    ) -> Generator[Tuple[int, int, Image.Image, float], None, None]:
        total_tiles = len(self.x_origins) * len(self.y_origins)
        logger.info("Scanning up to {} sliding windows...", total_tiles)
        extracted_count = 0

        for y_level in self.y_origins:
            for x_level in self.x_origins:
                x0 = int(round(x_level * self.downsample))
                y0 = int(round(y_level * self.downsample))
                img = self.slide.read_region(
                    (x0, y0),
                    self.level,
                    (self.tile_size, self.tile_size),
                ).convert("RGB")
                if self._is_empty_tissue(img):
                    continue

                gray = np.asarray(img.convert("L"))
                tissue_pct = round(1.0 - float(np.mean(gray > 220)), 4)
                yield x0, y0, img, tissue_pct
                extracted_count += 1
                if extracted_count >= settings.max_tiles_to_save_per_wsi:
                    logger.warning(
                        "Reached tile limit ({}). Halting scan.",
                        settings.max_tiles_to_save_per_wsi,
                    )
                    return


class InferencePipeline:
    """Run tiled YOLO inference and write the review queue to SQLite."""

    def __init__(self):
        ensure_directories()
        if not settings.active_model_path.exists():
            raise FileNotFoundError(
                f"Active model missing at {settings.active_model_path}"
            )
        self.model = YOLO(str(settings.active_model_path))
        self.db = SessionLocal()
        self.class_names = self.model.names
        self.run_id = uuid.uuid4().hex[:10]

    def run(self, wsi_path: str | Path) -> None:
        wsi = WsiScanner(wsi_path)
        wsi_name = Path(wsi_path).name
        all_global_detections: List[Dict] = []
        valid_tiles_info: List[Dict] = []
        start_time = time.time()

        try:
            batch: List[Tuple[int, int, Image.Image, float]] = []
            for patch in wsi.extract_patches():
                batch.append(patch)
                if len(batch) >= settings.inference_batch_size:
                    self._process_batch(
                        batch, wsi, wsi_name,
                        all_global_detections, valid_tiles_info,
                    )
                    batch = []
            if batch:
                self._process_batch(
                    batch, wsi, wsi_name,
                    all_global_detections, valid_tiles_info,
                )
        finally:
            wsi.close()

        final_detections = (
            self._apply_global_nms(all_global_detections)
            if all_global_detections
            else []
        )
        self._commit_to_db(
            wsi_name,
            valid_tiles_info,
            final_detections,
        )
        logger.success(
            "Inference complete in {:.2f}s",
            time.time() - start_time,
        )

    def _process_batch(
        self,
        batch: List[Tuple[int, int, Image.Image, float]],
        wsi: "WsiScanner",
        wsi_name: str,
        detections: List[Dict],
        tiles_info: List[Dict],
    ) -> None:
        results = self.model.predict(
            [img for _, _, img, _ in batch],
            conf=settings.inference_conf_threshold,
            verbose=False,
        )
        for (x0, y0, img, tissue_pct), result in zip(batch, results):
            boxes = result.boxes
            if len(boxes) == 0 and not settings.save_empty_tiles:
                continue

            img_filename = (
                f"{Path(wsi_name).stem}_{self.run_id}_{x0}_{y0}.jpg"
            )
            img.save(RAW_TILES_DIR / img_filename, "JPEG", quality=90)
            tiles_info.append(
                {
                    "x": x0,
                    "y": y0,
                    "tissue": tissue_pct,
                    "filename": img_filename,
                    "downsample": wsi.downsample,
                }
            )

            scale = settings.tile_size * wsi.downsample
            for box in boxes:
                x_c_n, y_c_n, w_n, h_n = box.xywhn[0].cpu().numpy()
                gx1 = x0 + (float(x_c_n) - float(w_n) / 2.0) * scale
                gy1 = y0 + (float(y_c_n) - float(h_n) / 2.0) * scale
                gx2 = x0 + (float(x_c_n) + float(w_n) / 2.0) * scale
                gy2 = y0 + (float(y_c_n) + float(h_n) / 2.0) * scale
                detections.append(
                    {
                        "box": [gx1, gy1, gx2, gy2],
                        "score": float(box.conf[0].cpu()),
                        "class_idx": int(box.cls[0].cpu()),
                        "tile_filename": img_filename,
                    }
                )

    def _apply_global_nms(self, detections: List[Dict]) -> List[Dict]:
        boxes_tensor = torch.tensor(
            [d["box"] for d in detections],
            dtype=torch.float32,
        )
        scores_tensor = torch.tensor(
            [d["score"] for d in detections],
            dtype=torch.float32,
        )
        keep_indices = torchvision.ops.nms(
            boxes_tensor,
            scores_tensor,
            settings.nms_iou_threshold,
        )
        survivors = [detections[int(i)] for i in keep_indices]
        logger.info(
            "Global NMS reduced detections from {} to {}",
            len(detections),
            len(survivors),
        )
        return survivors

    def _class_name(self, class_idx: int) -> str:
        if isinstance(self.class_names, dict):
            return str(self.class_names.get(class_idx, "unknown"))
        if 0 <= class_idx < len(self.class_names):
            return str(self.class_names[class_idx])
        return "unknown"

    def _commit_to_db(
        self,
        wsi_name: str,
        tiles_info: List[Dict],
        survivor_detections: List[Dict],
    ) -> None:
        try:
            tile_records: Dict[str, Tile] = {}
            tile_info_by_name = {
                item["filename"]: item
                for item in tiles_info
            }
            for info in tiles_info:
                relative_path = Path("raw_tiles") / info["filename"]
                tile = Tile(
                    file_path=str(relative_path),
                    source_wsi=wsi_name,
                    x_coord=info["x"],
                    y_coord=info["y"],
                    level=settings.magnification_level,
                    tissue_pct=info["tissue"],
                    is_annotated=False,
                )
                self.db.add(tile)
                tile_records[info["filename"]] = tile
            self.db.flush()

            for det in survivor_detections:
                tile = tile_records.get(det["tile_filename"])
                info = tile_info_by_name.get(det["tile_filename"])
                if tile is None or info is None:
                    continue

                tile_span = (
                    settings.tile_size
                    * float(info["downsample"])
                )
                gx1, gy1, gx2, gy2 = det["box"]
                lx1 = max(
                    0.0,
                    min((gx1 - tile.x_coord) / tile_span, 1.0),
                )
                ly1 = max(
                    0.0,
                    min((gy1 - tile.y_coord) / tile_span, 1.0),
                )
                lx2 = max(
                    0.0,
                    min((gx2 - tile.x_coord) / tile_span, 1.0),
                )
                ly2 = max(
                    0.0,
                    min((gy2 - tile.y_coord) / tile_span, 1.0),
                )
                width = lx2 - lx1
                height = ly2 - ly1
                if width <= 0.001 or height <= 0.001:
                    continue
                class_name = self._class_name(det["class_idx"])
                if class_name == "unknown":
                    continue
                self.db.add(
                    Annotation(
                        tile_id=tile.id,
                        class_label=class_name,
                        x_center=lx1 + width / 2.0,
                        y_center=ly1 + height / 2.0,
                        width=width,
                        height=height,
                        confidence=det["score"],
                        is_manual=False,
                    )
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        finally:
            self.db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sliding-window WSI inference"
    )
    parser.add_argument(
        "wsi_path",
        type=str,
        help="Path to an SVS/TIFF whole-slide image",
    )
    args = parser.parse_args()
    init_db()
    InferencePipeline().run(args.wsi_path)
