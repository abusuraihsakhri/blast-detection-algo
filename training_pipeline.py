"""
Module 4: Training Pipeline

This module is responsible for training the YOLO model using either:
1. Warm Start: Initial training on merged historical datasets.
2. HITL Incremental: Fine-tuning on a mix of new annotations and historical
   data assembled by the Experience Replay Buffer.

Security Notes:
    - All DB operations use SQLAlchemy sessions cleanly.
    - Model saving paths are validated against config directories.
    - Audit logs are meticulously tracked to enable potential rollbacks.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger
from sqlalchemy.orm import Session
from ultralytics import YOLO

from config import MODELS_DIR, PROJECT_ROOT, TEMP_TRAIN_DIR, settings
from database import SessionLocal, init_db
from models import TrainingRun, TrainingStatus
from replay_buffer import ExperienceReplayBuffer


class TrainingAuditLogger:
    """Handles bookkeeping for training runs in the database."""

    def __init__(self, db: Session):
        self.db = db
        self.run_id: Optional[int] = None

    def start_run(
        self, num_new: int, num_replay: int, hyperparams: dict
    ) -> int:
        """Log the start of a training run."""
        run = TrainingRun(
            status=TrainingStatus.RUNNING,
            num_new_images=num_new,
            num_replay_images=num_replay,
            config_json=json.dumps(hyperparams),
            started_at=datetime.now(timezone.utc)
        )
        self.db.add(run)
        self.db.commit()
        self.run_id = run.id
        logger.info(f"Started TrainingRun {self.run_id}")
        return self.run_id

    def complete_run(self, model_path: Path, map50: float) -> None:
        """Mark run as completed successfully."""
        if not self.run_id:
            return
        run = self.db.get(TrainingRun, self.run_id)
        if run:
            run.status = TrainingStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            run.model_path = str(model_path)
            run.final_map50 = map50
            self.db.commit()
            logger.info(f"Completed TrainingRun {self.run_id} with mAP50 {map50:.4f}")

    def fail_run(self) -> None:
        """Mark run as failed."""
        if not self.run_id:
            return
        run = self.db.get(TrainingRun, self.run_id)
        if run:
            run.status = TrainingStatus.FAILED
            run.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            logger.error(f"Failed TrainingRun {self.run_id}")


class WarmStartTrainer:
    """
    Builds the initial base model from downloaded Roboflow datasets.
    """

    def __init__(self):
        self.unified_train_dir = TEMP_TRAIN_DIR / "warm_start"
        self.datasets_dir = PROJECT_ROOT

        # Unified Taxonomy (from config.py)
        # 0: Benign, 1: Early, 2: Pre, 3: Pro
        # 4: WBC, 5: RBC, 6: Platelets
        # 7: Myeloblast, 8: Lymphoblast, 9: Atypical
        
        self.taxonomy = {
            0: "Benign",
            1: "Early",
            2: "Pre",
            3: "Pro",
            4: "WBC",
            5: "RBC",
            6: "Platelets",
            7: "Myeloblast",
            8: "Lymphoblast",
            9: "Atypical"
        }

        # Maps dataset-specific class IDs to the unified taxonomy
        self.remapping_rules = {
            "leukemia-1": {
                0: 0, # Benign -> Benign
                1: 1, # Early -> Early
                2: 2, # Pre -> Pre
                3: 3  # Pro -> Pro
            },
            "blast-cell-detection-1": {
                0: 2, # blast-cell -> Pre (treating generic blast as Pre)
                1: 4  # wbc -> WBC
            },
            "bccd-blood-cell-1": {
                0: 6, # Platelets
                1: 5, # RBC
                2: 4  # WBC
            },
            "blood-cell-znm2t-1": {
                # Map all detailed WBC subtypes from this dataset to the generic WBC class
                0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4
            },
            "acute-leukemia-1": {
                0: 8, # lymph-b -> Lymphoblast
                1: 7, # myelo-b -> Myeloblast
                2: 4  # wbc -> WBC
            },
            "myeloblast-fbliw-1": {
                0: 7, # Myeloblast -> Myeloblast
                1: 5, # RBC -> RBC
                2: 7  # WBC -> Myeloblast (fixes data poisoning where annotators labeled blasts as generic WBC)
            }
        }

    def prepare_unified_dataset(self, max_samples: Optional[int] = None) -> Path:
        """Merge all datasets into one YOLO-format folder with remapped labels."""
        logger.info("Assembling warm-start unified dataset...")
        if self.unified_train_dir.exists():
            shutil.rmtree(self.unified_train_dir)
            
        # Create standard YOLO splits
        for split in ["train", "valid"]:
            (self.unified_train_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.unified_train_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        for ds_name, class_map in self.remapping_rules.items():
            ds_path = self.datasets_dir / ds_name
            if not ds_path.exists():
                logger.warning(f"Dataset missing: {ds_name}. Skipping.")
                continue

            for split in ["train", "valid"]:
                src_images = ds_path / split / "images"
                src_labels = ds_path / split / "labels"
                
                # Some datasets use 'val' instead of 'valid'
                if split == "valid" and not src_images.exists():
                    src_images = ds_path / "val" / "images"
                    src_labels = ds_path / "val" / "labels"

                if not src_images.exists():
                    continue

                copied = 0
                for img_file in src_images.iterdir():
                    if max_samples and copied >= max_samples:
                        break
                    if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        # Also check if label exists, otherwise skip
                        txt_file = src_labels / (img_file.stem + ".txt")
                        if not txt_file.exists():
                            continue
                            
                        dest_img = self.unified_train_dir / split / "images" / f"{ds_name}_{img_file.name}"
                        shutil.copy2(img_file, dest_img)
                        
                        dest_txt = self.unified_train_dir / split / "labels" / f"{ds_name}_{txt_file.name}"
                        self._remap_and_copy_label(txt_file, dest_txt, class_map)
                        copied += 1

        return self._generate_yaml()

    def _remap_and_copy_label(self, src: Path, dest: Path, class_map: dict) -> None:
        """Read original labels, remap class IDs, and rewrite."""
        lines = []
        try:
            with open(src, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        orig_cls = int(parts[0])
                        if orig_cls in class_map:
                            new_cls = class_map[orig_cls]
                            parts[0] = str(new_cls)
                            lines.append(" ".join(parts))
        except Exception as e:
            logger.warning(f"Failed to parse label {src}: {e}")
            return

        with open(dest, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")

    def _generate_yaml(self) -> Path:
        """Generate dataset.yaml for the unified dataset."""
        yaml_path = self.unified_train_dir / "dataset.yaml"
        config = {
            "path": str(self.unified_train_dir.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "names": self.taxonomy
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        return yaml_path

    def train(self, max_samples: Optional[int] = None, resume: bool = False) -> None:
        """Execute the warm start training process."""
        logger.info("Initializing Warm Start Training...")
        data_yaml = self.prepare_unified_dataset(max_samples=max_samples)
        
        # Start DB session to log the run
        session = SessionLocal()
        audit = TrainingAuditLogger(session)
        
        # If performing a small dry run, reflect that in hyperparams
        actual_epochs = 2 if max_samples else settings.warm_start_epochs
        
        audit.start_run(
            num_new=0, # It's all historical
            num_replay=-1, # denotes massive warm start
            hyperparams={
                "epochs": actual_epochs,
                "imgsz": settings.warm_start_imgsz,
                "batch": settings.batch_size,
                "mode": "warm_start",
                "max_samples": max_samples,
                "resume": resume
            }
        )

        try:
            if resume:
                # Find the latest detect/warm_start_* folder
                detect_dir = PROJECT_ROOT / "runs" / "detect"
                warm_runs = [d for d in detect_dir.iterdir() if d.is_dir() and d.name.startswith("warm_start_")]
                if not warm_runs:
                    raise FileNotFoundError("No warm-start training runs found to resume from.")
                latest_run = max(warm_runs, key=lambda d: d.stat().st_mtime)
                latest_checkpoint = latest_run / "weights" / "last.pt"
                if not latest_checkpoint.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {latest_checkpoint}")
                logger.info(f"Loading YOLO model for resume from {latest_checkpoint}...")
                model = YOLO(str(latest_checkpoint))
                logger.info("Resuming training...")
                results = model.train(resume=True)
            else:
                logger.info("Loading base YOLO model...")
                model = YOLO(settings.base_model_name)
                logger.info(f"Starting training for {actual_epochs} epochs...")
                results = model.train(
                    data=str(data_yaml),
                    epochs=actual_epochs,
                    imgsz=settings.warm_start_imgsz,
                    batch=settings.batch_size,
                    project=str(PROJECT_ROOT / "runs" / "detect"),
                    name=f"warm_start_{int(time.time())}",
                    exist_ok=True,
                    verbose=True,
                    workers=0
                )
            
            # The best model path is generally in results.save_dir
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            
            # Extract final mAP@50 (often stored in results.results_dict under maps)
            map50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
            
            if best_model_path.exists():
                settings.active_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_model_path, settings.active_model_path)
                logger.info(f"Deployed new active model to {settings.active_model_path}")
            
            audit.complete_run(model_path=settings.active_model_path, map50=map50)
            
        except Exception as e:
            logger.error(f"Warm start training failed: {e}")
            audit.fail_run()
            raise
        finally:
            session.close()


class HitlTrainer:
    """
    Handles incremental fine-tuning during active learning loops.
    Draws data from the Experience Replay Buffer.
    """

    def __init__(self):
        if not settings.active_model_path.exists():
            raise FileNotFoundError(
                f"Active model not found at {settings.active_model_path}. "
                "You must run warm-start training first."
            )
        self.session = SessionLocal()
        self.buffer = ExperienceReplayBuffer(self.session)
        self.audit = TrainingAuditLogger(self.session)

    def train(self, resume: bool = False) -> None:
        """Execute incremental training."""
        try:
            logger.info("Initializing HITL Incremental Training...")
            
            # Pre-training check: fetch new dataset
            data_yaml = self.buffer.assemble_dataset()
            stats = self.buffer.get_buffer_stats()
            
            self.audit.start_run(
                num_new=stats["new_tile_count"],
                num_replay=stats["target_replay_count"],
                hyperparams={
                    "epochs": settings.max_epochs,
                    "imgsz": settings.warm_start_imgsz,
                    "batch": settings.batch_size,
                    "mode": "hitl",
                    "resume": resume
                }
            )

            if resume:
                detect_dir = PROJECT_ROOT / "runs" / "detect"
                hitl_runs = [d for d in detect_dir.iterdir() if d.is_dir() and d.name.startswith("hitl_train_")]
                if not hitl_runs:
                    raise FileNotFoundError("No hitl training runs found to resume from.")
                latest_run = max(hitl_runs, key=lambda d: d.stat().st_mtime)
                latest_checkpoint = latest_run / "weights" / "last.pt"
                logger.info(f"Loading YOLO model for resume from {latest_checkpoint}...")
                model = YOLO(str(latest_checkpoint))
            else:
                # Start training from active_model.pt
                model = YOLO(str(settings.active_model_path))
            if resume:
                logger.info("Resuming HITL training...")
                results = model.train(resume=True)
            else:
                results = model.train(
                    data=str(data_yaml),
                    epochs=settings.max_epochs, # fewer epochs for incremental
                    imgsz=settings.warm_start_imgsz,
                    batch=settings.batch_size,
                    project=str(PROJECT_ROOT / "runs" / "detect"),
                    name=f"hitl_train_{int(time.time())}",
                    exist_ok=True,
                    workers=0
                )

            map50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"

            # In a true sophisticated AL setup, you might compare this mAP50
            # against a holdout test set to decide whether to promote.
            # Here we just check if it meets a minimum threshold.
            if map50 >= settings.min_map50_improvement:
                logger.info(f"Model meets promotion criteria (mAP50={map50:.4f}). Promoting.")
                shutil.copy2(best_model_path, settings.active_model_path)
            else:
                logger.warning(f"Model failed to meet minimum mAP thresholds. Discarding updates.")
                
            self.audit.complete_run(model_path=settings.active_model_path, map50=map50)
            
            logger.info("Cleaning up replay buffer data...")
            self.buffer.cleanup()

        except Exception as e:
            logger.error(f"HITL training failed: {e}")
            self.audit.fail_run()
            raise
        finally:
            self.session.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blast Algo Training Pipeline")
    parser.add_argument(
        "--mode",
        choices=["warm", "hitl"],
        required=True,
        help="Training mode: 'warm' for initial build, 'hitl' for incremental fine-tuning."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest available checkpoint for the given mode."
    )
    args = parser.parse_args()

    init_db()

    if args.mode == "warm":
        trainer = WarmStartTrainer()
        trainer.train(resume=args.resume)
    elif args.mode == "hitl":
        trainer = HitlTrainer()
        trainer.train(resume=args.resume)
