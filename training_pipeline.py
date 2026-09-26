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


def _is_cell_box(parts: list[str]) -> bool:
    """Return False for YOLO label rows that are not real cell boxes.

    ``parts`` is one label row split on whitespace:
    [class_id, x_center, y_center, width, height], coordinates normalized 0-1.

    Myeloblast-fbliw tags 465 of its 472 "RBC" boxes as near-full-frame
    rectangles (width * height > 0.8). These are image-level tags, and they
    currently enter training and validation as RBC targets.
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

                    shutil.copy2(
                        img_file,
                        self.unified_train_dir
                        / split
                        / "images"
                        / f"{ds_name}_{img_file.name}",
                    )
                    self._remap_and_copy_label(
                        label,
                        self.unified_train_dir
                        / split
                        / "labels"
                        / f"{ds_name}_{label.name}",
                        class_map,
                    )
                    copied += 1
        return self._generate_yaml()

    def _remap_and_copy_label(
        self,
        src: Path,
        dest: Path,
        class_map: dict,
    ) -> None:
        lines = []
        for line in src.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                original = int(parts[0])
            except ValueError:
                logger.warning(
                    "Skipping malformed class id in {}",
                    src,
                )
                continue
            if original in class_map and _is_cell_box(parts):
                parts[0] = str(class_map[original])
                lines.append(" ".join(parts))

        dest.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

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
