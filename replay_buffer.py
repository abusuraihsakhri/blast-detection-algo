"""Experience replay dataset assembly for reviewed HITL tiles."""

from __future__ import annotations

import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from loguru import logger
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from config import DATA_DIR, RAW_TILES_DIR, TEMP_TRAIN_DIR, settings, validate_path_within
from models import Annotation, ClassRegistry, Tile, TrainingRun, TrainingStatus


class ReplayBufferError(Exception):
    pass


class ExperienceReplayBuffer:
    def __init__(
        self,
        db: Session,
        replay_ratio: float | None = None,
        train_val_split: float | None = None,
        max_dataset_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.db = db
        self.replay_ratio = settings.replay_ratio if replay_ratio is None else replay_ratio
        self.train_val_split = settings.train_val_split if train_val_split is None else train_val_split
        self.max_dataset_size = settings.max_replay_dataset_size if max_dataset_size is None else max_dataset_size
        self._rng = random.Random(seed)
        self._class_map = self._load_class_map()

    def assemble_dataset(self) -> Path:
        new_tiles = self.get_new_tiles()
        if not new_tiles:
            raise ReplayBufferError("No newly annotated tiles are available for retraining.")
        if len(new_tiles) > self.max_dataset_size:
            raise ReplayBufferError(
                f"New reviewed tiles ({len(new_tiles)}) exceed max dataset size "
                f"({self.max_dataset_size}); increase BLAST_MAX_REPLAY_DATASET_SIZE."
            )

        replay_tiles = self._get_replay_tiles(len(new_tiles))
        seen: Set[int] = {tile.id for tile in new_tiles}
        combined = list(new_tiles)
        for tile in replay_tiles:
            if tile.id not in seen:
                seen.add(tile.id)
                combined.append(tile)

        if len(combined) > self.max_dataset_size:
            new_ids = {tile.id for tile in new_tiles}
            replay_only = [tile for tile in combined if tile.id not in new_ids]
            self._rng.shuffle(replay_only)
            combined = list(new_tiles) + replay_only[: self.max_dataset_size - len(new_tiles)]

        train_tiles, val_tiles = self._stratified_train_val_split(combined)
        self._prepare_temp_directory()
        self._copy_tiles_and_labels(train_tiles, "train")
        self._copy_tiles_and_labels(val_tiles, "val")
        return self._generate_dataset_yaml()

    def get_new_tiles(self) -> List[Tile]:
        last_run = self._get_last_completed_run()
        cutoff = last_run.completed_at if last_run else datetime.min.replace(tzinfo=timezone.utc)
        return (
            self.db.query(Tile)
            .filter(Tile.is_annotated.is_(True), Tile.updated_at > cutoff)
            .all()
        )

    def get_class_distribution(self, before: Optional[datetime] = None) -> Dict[str, int]:
        last_run = self._get_last_completed_run()
        cutoff = before or (
            last_run.completed_at if last_run else datetime.max.replace(tzinfo=timezone.utc)
        )
        rows = (
            self.db.query(Annotation.class_label, sa_func.count(sa_func.distinct(Annotation.tile_id)))
            .join(Tile, Annotation.tile_id == Tile.id)
            .filter(Tile.is_annotated.is_(True), Tile.updated_at <= cutoff)
            .group_by(Annotation.class_label)
            .all()
        )
        return {label: count for label, count in rows}

    def cleanup(self) -> None:
        if TEMP_TRAIN_DIR.exists():
            shutil.rmtree(TEMP_TRAIN_DIR)

    def _load_class_map(self) -> Dict[str, int]:
        mapping = {row.name: row.yolo_index for row in self.db.query(ClassRegistry).all()}
        if mapping:
            return mapping
        for idx, name in enumerate(settings.default_classes):
            self.db.add(ClassRegistry(name=name, yolo_index=idx))
            mapping[name] = idx
        self.db.flush()
        return mapping

    def _get_last_completed_run(self) -> Optional[TrainingRun]:
        return (
            self.db.query(TrainingRun)
            .filter(TrainingRun.status == TrainingStatus.COMPLETED)
            .order_by(TrainingRun.completed_at.desc())
            .first()
        )

    def _get_replay_tiles(self, num_new: int) -> List[Tile]:
        target_count = int(math.ceil(num_new * self.replay_ratio))
        class_dist = self.get_class_distribution()
        if not class_dist or target_count <= 0:
            return []
        quotas = self._compute_class_quotas(class_dist, target_count)
        last_run = self._get_last_completed_run()
        cutoff = last_run.completed_at if last_run else datetime.max.replace(tzinfo=timezone.utc)
        sampled_ids: Set[int] = set()
        sampled: List[Tile] = []

        for class_label, quota in quotas.items():
            rows = (
                self.db.query(Annotation.tile_id)
                .join(Tile, Annotation.tile_id == Tile.id)
                .filter(
                    Annotation.class_label == class_label,
                    Tile.is_annotated.is_(True),
                    Tile.updated_at <= cutoff,
                )
                .distinct()
                .all()
            )
            candidates = [tid for (tid,) in rows if tid not in sampled_ids]
            for tid in self._rng.sample(candidates, min(quota, len(candidates))):
                tile = self.db.get(Tile, tid)
                if tile is not None and tid not in sampled_ids:
                    sampled_ids.add(tid)
                    sampled.append(tile)
        return sampled

    def _compute_class_quotas(self, class_dist: Dict[str, int], target_total: int) -> Dict[str, int]:
        if not class_dist or target_total <= 0:
            return {label: 0 for label in class_dist}
        counts = sorted(class_dist.values())
        median = counts[len(counts) // 2]
        base = target_total / len(class_dist)
        quotas = {
            label: min(
                available,
                max(1, int(math.ceil(base * (settings.underrepresented_boost if available <= median else 0.5)))),
            )
            for label, available in class_dist.items()
        }
        while sum(quotas.values()) > target_total:
            reducible = [label for label, value in quotas.items() if value > 0]
            if not reducible:
                break
            label = max(reducible, key=lambda key: quotas[key])
            quotas[label] -= 1
        return quotas

    def _stratified_train_val_split(self, tiles: List[Tile]) -> Tuple[List[Tile], List[Tile]]:
        groups: Dict[str, List[Tile]] = defaultdict(list)
        for tile in tiles:
            if tile.annotations:
                counts: Dict[str, int] = defaultdict(int)
                for ann in tile.annotations:
                    counts[ann.class_label] += 1
                groups[max(counts, key=counts.get)].append(tile)
            else:
                groups["_negative"].append(tile)

        train: List[Tile] = []
        val: List[Tile] = []
        for group in groups.values():
            self._rng.shuffle(group)
            if len(group) == 1:
                train.extend(group)
                continue
            split_idx = min(len(group) - 1, max(1, int(len(group) * self.train_val_split)))
            train.extend(group[:split_idx])
            val.extend(group[split_idx:])
        if not val and len(train) > 1:
            val.append(train.pop())
        return train, val

    def _prepare_temp_directory(self) -> None:
        if TEMP_TRAIN_DIR.exists():
            shutil.rmtree(TEMP_TRAIN_DIR)
        for split in ("train", "val"):
            (TEMP_TRAIN_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (TEMP_TRAIN_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _resolve_image_path(self, tile: Tile) -> Path:
        stored = Path(tile.file_path)
        candidate = stored if stored.is_absolute() else DATA_DIR / stored
        candidate = validate_path_within(candidate, DATA_DIR)
        if candidate.exists():
            return candidate
        if not stored.is_absolute() and len(stored.parts) == 1:
            legacy = validate_path_within(RAW_TILES_DIR / stored.name, RAW_TILES_DIR)
            if legacy.exists():
                return legacy
        return candidate

    def _copy_tiles_and_labels(self, tiles: List[Tile], split: str) -> None:
        images_dir = TEMP_TRAIN_DIR / "images" / split
        labels_dir = TEMP_TRAIN_DIR / "labels" / split
        for tile in tiles:
            try:
                image_src = self._resolve_image_path(tile)
            except ValueError as exc:
                logger.error("Skipping tile {}: {}", tile.id, exc)
                continue
            if not image_src.exists():
                logger.warning("Tile image not found, skipping: {}", image_src)
                continue
            shutil.copy2(image_src, images_dir / image_src.name)
            lines = self._generate_yolo_labels(tile)
            label_path = labels_dir / f"{image_src.stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _generate_yolo_labels(self, tile: Tile) -> List[str]:
        lines: List[str] = []
        for ann in tile.annotations:
            class_idx = self._class_map.get(ann.class_label)
            if class_idx is None:
                class_idx = self._register_new_class(ann.class_label)
            lines.append(ann.to_yolo_line(class_idx))
        return lines

    def _register_new_class(self, class_label: str) -> int:
        max_idx = self.db.query(sa_func.max(ClassRegistry.yolo_index)).scalar()
        new_idx = 0 if max_idx is None else max_idx + 1
        self.db.add(ClassRegistry(name=class_label, yolo_index=new_idx))
        self.db.flush()
        self._class_map[class_label] = new_idx
        return new_idx

    def _generate_dataset_yaml(self) -> Path:
        names = {idx: name for name, idx in sorted(self._class_map.items(), key=lambda item: item[1])}
        config = {
            "path": str(TEMP_TRAIN_DIR.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        yaml_path = TEMP_TRAIN_DIR / "dataset.yaml"
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return yaml_path

    def get_buffer_stats(self) -> Dict:
        new_tiles = self.get_new_tiles()
        target_replay = int(math.ceil(len(new_tiles) * self.replay_ratio))
        return {
            "new_tile_count": len(new_tiles),
            "replay_ratio": self.replay_ratio,
            "target_replay_count": target_replay,
            "total_estimated": min(len(new_tiles) + target_replay, self.max_dataset_size),
            "max_dataset_size": self.max_dataset_size,
            "historical_class_distribution": self.get_class_distribution(),
            "registered_classes": list(self._class_map.keys()),
        }
