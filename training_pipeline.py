"""Warm-start and incremental YOLO training workflows."""

from __future__ import annotations

import json
import re
import shutil
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from PIL import Image
from sqlalchemy.orm import Session
from ultralytics import YOLO

from config import MODELS_DIR, PROJECT_ROOT, TEMP_TRAIN_DIR, settings
from database import SessionLocal, init_db
from models import TrainingRun, TrainingStatus
from replay_buffer import ExperienceReplayBuffer


class TrainingAuditLogger:
    def __init__(self, db: Session):
        self.db = db
        self.run_id: Optional[int] = None

    def start_run(
        self,
        num_new: int,
        num_replay: int,
        hyperparams: dict,
    ) -> int:
        run = TrainingRun(
            status=TrainingStatus.RUNNING,
            num_new_images=num_new,
            num_replay_images=num_replay,
            config_json=json.dumps(hyperparams),
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(run)
        self.db.commit()
        self.run_id = run.id
        return run.id

    def complete_run(
        self,
        model_path: Path,
        map50: float,
    ) -> None:
        if self.run_id is None:
            return
        run = self.db.get(TrainingRun, self.run_id)
        if run:
            run.status = TrainingStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            run.model_path = str(model_path)
            run.final_map50 = float(map50)
            self.db.commit()

    def fail_run(self) -> None:
        if self.run_id is None:
            return
        run = self.db.get(TrainingRun, self.run_id)
        if run:
            run.status = TrainingStatus.FAILED
            run.completed_at = datetime.now(timezone.utc)
            self.db.commit()


def _metric_map50(results) -> float:
    return float(
        getattr(results, "results_dict", {}).get(
            "metrics/mAP50(B)",
            0.0,
        )
    )


MAX_CELL_BOX_AREA = 0.8


def _to_box(parts: list[str]) -> Optional[list[str]]:
    """Normalize one YOLO label row to [class, x_center, y_center, w, h].

    Some sources store polygons (class x1 y1 x2 y2 ...), and Leukemia-NFXZN
    mixes polygon and box rows in one file. Ultralytics reads a file with any
    polygon row as all-polygon, which garbles the box rows, so every row is
    converted to a plain box here. Returns None for malformed rows.
    """
    try:
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(values) == 4:
        x, y, w, h = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = [min(max(v, 0.0), 1.0) for v in values[0::2]]
        ys = [min(max(v, 0.0), 1.0) for v in values[1::2]]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        x, y = min(xs) + w / 2, min(ys) + h / 2
    else:
        return None
    if w <= 0 or h <= 0:
        return None
    return [parts[0]] + [f"{v:.6f}" for v in (x, y, w, h)]


def _is_cell_box(parts: list[str]) -> bool:
    """Return False for YOLO label rows that are not real cell boxes.

    ``parts`` is one label row split on whitespace:
    [class_id, x_center, y_center, width, height], coordinates normalized 0-1.

    Expects a row already normalized by ``_to_box``. Myeloblast-fbliw tags
    465 of its 472 "RBC" boxes as near-full-frame rectangles
    (width * height > 0.8). These are image-level tags, not cells.
    """
    try:
        width, height = float(parts[3]), float(parts[4])
    except (IndexError, ValueError):
        return False
    return width * height <= MAX_CELL_BOX_AREA


HOLDOUT_EVERY = 10
_AUGMENT_SUFFIX = re.compile(r"_(fH|fV|r\d{3})$")


def _source_image_id(filename: str) -> str:
    """Strip Roboflow hashes and baked-in flip/rotate suffixes.

    ``ALL1_100_fH_jpg.rf.<hash>.jpg`` and ``ALL1_100_jpg.rf.<hash>.jpg`` are
    copies of one source image and must land in the same split.
    """
    stem = filename.split(".rf.")[0].removesuffix("_jpg")
    return _AUGMENT_SUFFIX.sub("", stem)


def _in_holdout(filename: str) -> bool:
    key = _source_image_id(filename).encode("utf-8")
    return zlib.crc32(key) % HOLDOUT_EVERY == 0


def _split_source(
    ds_path: Path, split: str
) -> tuple[Path, Path, Optional[bool]]:
    """Return (images, labels, holdout) for one split of a source dataset.

    Datasets that ship without a validation split lend about one in
    ``HOLDOUT_EVERY`` source images to validation, so every class gets
    measured. ``holdout`` is None when the split exists as-is, True to keep
    only held-out images, and False to skip them.
    """
    train = ds_path / "train"
    for name in ("valid", "val"):
        if (ds_path / name / "images").exists():
            source = train if split == "train" else ds_path / name
            return source / "images", source / "labels", None
    return train / "images", train / "labels", split == "valid"


NEAR_DUPLICATE_BITS = 10
_HASH_BANDS = NEAR_DUPLICATE_BITS + 1
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((17, 16))
    px = gray.tobytes()
    bits = 0
    for row in range(16):
        for col in range(16):
            if px[row * 17 + col] > px[row * 17 + col + 1]:
                bits |= 1 << (row * 16 + col)
    return bits


def _orientation_free_hash(path: Path) -> int:
    """256-bit difference hash, minimized over flips and 90° rotations."""
    with Image.open(path) as image:
        variants = [
            image,
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            image.transpose(Image.Transpose.ROTATE_90),
            image.transpose(Image.Transpose.ROTATE_180),
            image.transpose(Image.Transpose.ROTATE_270),
        ]
        return min(_dhash(v) for v in variants)


def _bands(value: int) -> list[tuple[int, int]]:
    # Two hashes within NEAR_DUPLICATE_BITS share at least one exact band.
    width = -(-256 // _HASH_BANDS)
    mask = (1 << width) - 1
    return [(i, (value >> (i * width)) & mask) for i in range(_HASH_BANDS)]


def _near_duplicates(
    candidates: list[Path], references: list[Path]
) -> list[Path]:
    """Return candidates that nearly duplicate any reference image."""
    index: dict[tuple[int, int], list[int]] = {}
    for ref in references:
        value = _orientation_free_hash(ref)
        for band in _bands(value):
            index.setdefault(band, []).append(value)
    hits = []
    for path in candidates:
        value = _orientation_free_hash(path)
        if any(
            (value ^ ref).bit_count() <= NEAR_DUPLICATE_BITS
            for band in _bands(value)
            for ref in index.get(band, ())
        ):
            hits.append(path)
    return hits


def _latest_checkpoint(prefix: str) -> Path:
    detect_dir = PROJECT_ROOT / "runs" / "detect"
    if not detect_dir.exists():
        raise FileNotFoundError(
            f"No training run directory exists at {detect_dir}"
        )
    runs = [
        d
        for d in detect_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not runs:
        raise FileNotFoundError(
            f"No training runs found with prefix '{prefix}'."
        )
    checkpoint = (
        max(runs, key=lambda d: d.stat().st_mtime)
        / "weights"
        / "last.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )
    return checkpoint


class WarmStartTrainer:
    """Merge configured source datasets and train the initial detector."""

    def __init__(self):
        self.unified_train_dir = TEMP_TRAIN_DIR / "warm_start"
        self.datasets_dir = PROJECT_ROOT
        self.taxonomy = {
            idx: name
            for idx, name in enumerate(settings.default_classes)
        }
        self.remapping_rules = {
            "leukemia-1": {0: 0, 1: 1, 2: 2, 3: 3},
            "blast-cell-detection-1": {0: 2, 1: 4},
            "bccd-blood-cell-1": {0: 6, 1: 5, 2: 4},
            "blood-cell-znm2t-1": {
                0: 4,
                1: 4,
                2: 4,
                3: 4,
                4: 4,
                5: 4,
                6: 4,
                7: 4,
            },
            "acute-leukemia-1": {0: 8, 1: 7, 2: 4},
            "myeloblast-fbliw-1": {0: 7, 1: 5, 2: 7},
        }

    def prepare_unified_dataset(
        self,
        max_samples: Optional[int] = None,
    ) -> Path:
        if self.unified_train_dir.exists():
            shutil.rmtree(self.unified_train_dir)
        for split in ("train", "valid"):
            (
                self.unified_train_dir
                / split
                / "images"
            ).mkdir(parents=True, exist_ok=True)
            (
                self.unified_train_dir
                / split
                / "labels"
            ).mkdir(parents=True, exist_ok=True)

        for ds_name, class_map in self.remapping_rules.items():
            ds_path = self.datasets_dir / ds_name
            if not ds_path.exists():
                logger.warning(
                    "Dataset missing: {}. Skipping.",
                    ds_name,
                )
                continue

            for split in ("train", "valid"):
                src_images, src_labels, holdout = _split_source(
                    ds_path, split
                )
                if not src_images.exists():
                    continue

                copied = 0
                for img_file in sorted(src_images.iterdir()):
                    if (
                        max_samples is not None
                        and copied >= max_samples
                    ):
                        break
                    if img_file.suffix.lower() not in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                    }:
                        continue
                    label = src_labels / f"{img_file.stem}.txt"
                    if not label.exists():
                        continue
                    if (
                        holdout is not None
                        and _in_holdout(img_file.name) != holdout
                    ):
                        continue

                    dest_label = (
                        self.unified_train_dir
                        / split
                        / "labels"
                        / f"{ds_name}_{label.name}"
                    )
                    if not self._remap_and_copy_label(
                        label, dest_label, class_map
                    ):
                        # Every row was an artifact; the real cell is
                        # unlabeled, so the image would teach it as
                        # background.
                        dest_label.unlink()
                        continue
                    shutil.copy2(
                        img_file,
                        self.unified_train_dir
                        / split
                        / "images"
                        / f"{ds_name}_{img_file.name}",
                    )
                    copied += 1
        self._drop_leaked_train_images()
        return self._generate_yaml()

    def _drop_leaked_train_images(self) -> None:
        """Remove train images that reappear in any validation or test split.

        Several sources re-export the same micrographs (Acute-Leukemia shares
        images with Blast-Cell-Detection and Myeloblast-fbliw; BCCD with
        Blood-Cell-znm2t), so per-dataset splits leak across datasets.
        """
        references = [
            p
            for p in (self.unified_train_dir / "valid" / "images").iterdir()
        ]
        for ds_name in self.remapping_rules:
            test_dir = self.datasets_dir / ds_name / "test" / "images"
            if test_dir.exists():
                references.extend(test_dir.iterdir())
        references = [
            p for p in references if p.suffix.lower() in _IMAGE_SUFFIXES
        ]
        train_images = self.unified_train_dir / "train" / "images"
        leaked = _near_duplicates(
            sorted(train_images.iterdir()), references
        )
        for image in leaked:
            image.unlink()
            label = (
                self.unified_train_dir
                / "train"
                / "labels"
                / f"{image.stem}.txt"
            )
            label.unlink(missing_ok=True)
        logger.info(
            "Removed {} train images that duplicate validation/test images.",
            len(leaked),
        )

    def _remap_and_copy_label(
        self,
        src: Path,
        dest: Path,
        class_map: dict,
    ) -> bool:
        """Write remapped boxes; False if the source had rows but none survived."""
        lines = []
        rows = 0
        for line in src.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            rows += 1
            try:
                original = int(parts[0])
            except ValueError:
                logger.warning(
                    "Skipping malformed class id in {}",
                    src,
                )
                continue
            box = _to_box(parts)
            if original in class_map and box and _is_cell_box(box):
                box[0] = str(class_map[original])
                lines.append(" ".join(box))

        dest.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        return bool(lines) or rows == 0

    def _generate_yaml(self) -> Path:
        path = self.unified_train_dir / "dataset.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "path": str(
                        self.unified_train_dir.resolve()
                    ),
                    "train": "train/images",
                    "val": "valid/images",
                    "names": self.taxonomy,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def train(
        self,
        max_samples: Optional[int] = None,
        resume: bool = False,
    ) -> None:
        if resume:
            # The checkpoint points at the existing merged dataset;
            # rebuilding it would change the data mid-run.
            data_yaml = self.unified_train_dir / "dataset.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(
                    "Cannot resume: merged dataset missing at "
                    f"{data_yaml}. Start a new run instead."
                )
        else:
            data_yaml = self.prepare_unified_dataset(
                max_samples=max_samples
            )
        session = SessionLocal()
        audit = TrainingAuditLogger(session)
        actual_epochs = (
            2
            if max_samples is not None
            else settings.warm_start_epochs
        )
        audit.start_run(
            0,
            -1,
            {
                "epochs": actual_epochs,
                "imgsz": settings.warm_start_imgsz,
                "batch": settings.batch_size,
                "mode": "warm_start",
                "max_samples": max_samples,
                "resume": resume,
            },
        )
        try:
            if resume:
                model = YOLO(
                    str(
                        _latest_checkpoint(
                            "warm_start_"
                        )
                    )
                )
                results = model.train(resume=True)
            else:
                model = YOLO(settings.base_model_name)
                results = model.train(
                    data=str(data_yaml),
                    epochs=actual_epochs,
                    imgsz=settings.warm_start_imgsz,
                    batch=settings.batch_size,
                    project=str(
                        PROJECT_ROOT
                        / "runs"
                        / "detect"
                    ),
                    name=f"warm_start_{int(time.time())}",
                    exist_ok=True,
                    verbose=True,
                    workers=0,
                )

            best = (
                Path(results.save_dir)
                / "weights"
                / "best.pt"
            )
            if not best.exists():
                raise FileNotFoundError(
                    "Training completed without "
                    f"expected checkpoint: {best}"
                )
            MODELS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                best,
                settings.active_model_path,
            )
            audit.complete_run(
                settings.active_model_path,
                _metric_map50(results),
            )
        except Exception:
            audit.fail_run()
            raise
        finally:
            session.close()


class HitlTrainer:
    """Fine-tune the active model on new reviews plus prior reviewed tiles."""

    def __init__(self):
        if not settings.active_model_path.exists():
            raise FileNotFoundError(
                "Active model not found. "
                "Run warm-start training first."
            )
        self.session = SessionLocal()
        self.buffer = ExperienceReplayBuffer(
            self.session
        )
        self.audit = TrainingAuditLogger(
            self.session
        )

    def train(self, resume: bool = False) -> None:
        try:
            data_yaml = self.buffer.assemble_dataset()
            stats = self.buffer.get_buffer_stats()
            self.audit.start_run(
                stats["new_tile_count"],
                stats["target_replay_count"],
                {
                    "epochs": settings.max_epochs,
                    "imgsz": settings.warm_start_imgsz,
                    "batch": settings.batch_size,
                    "mode": "hitl",
                    "resume": resume,
                },
            )

            active_model = YOLO(
                str(settings.active_model_path)
            )
            baseline = active_model.val(
                data=str(data_yaml),
                imgsz=settings.warm_start_imgsz,
                batch=settings.batch_size,
                verbose=False,
            )
            baseline_map50 = _metric_map50(
                baseline
            )

            if resume:
                model = YOLO(
                    str(
                        _latest_checkpoint(
                            "hitl_train_"
                        )
                    )
                )
                results = model.train(
                    resume=True
                )
            else:
                results = active_model.train(
                    data=str(data_yaml),
                    epochs=settings.max_epochs,
                    imgsz=settings.warm_start_imgsz,
                    batch=settings.batch_size,
                    project=str(
                        PROJECT_ROOT
                        / "runs"
                        / "detect"
                    ),
                    name=(
                        f"hitl_train_"
                        f"{int(time.time())}"
                    ),
                    exist_ok=True,
                    workers=0,
                )

            trained_map50 = _metric_map50(
                results
            )
            best = (
                Path(results.save_dir)
                / "weights"
                / "best.pt"
            )
            if not best.exists():
                raise FileNotFoundError(
                    "Training completed without "
                    f"expected checkpoint: {best}"
                )

            improvement = (
                trained_map50 - baseline_map50
            )
            if (
                improvement
                >= settings.min_map50_improvement
            ):
                shutil.copy2(
                    best,
                    settings.active_model_path,
                )
                logger.info(
                    "Promoted model: validation "
                    "mAP50 {:.4f} -> {:.4f} "
                    "(Δ {:.4f}).",
                    baseline_map50,
                    trained_map50,
                    improvement,
                )
            else:
                logger.warning(
                    "Model not promoted: validation "
                    "mAP50 {:.4f} -> {:.4f} "
                    "(Δ {:.4f}).",
                    baseline_map50,
                    trained_map50,
                    improvement,
                )

            self.audit.complete_run(
                settings.active_model_path,
                trained_map50,
            )
            self.buffer.cleanup()
        except Exception:
            self.audit.fail_run()
            raise
        finally:
            self.session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Blast detection training pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["warm", "hitl"],
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    args = parser.parse_args()
    init_db()
    if args.mode == "warm":
        WarmStartTrainer().train(
            resume=args.resume
        )
    else:
        HitlTrainer().train(
            resume=args.resume
        )
