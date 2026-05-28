"""
Module 3: Experience Replay Buffer for HITL Active Learning.

Prevents catastrophic forgetting by constructing balanced training
datasets that mix newly annotated tiles with a stratified sample of
historical annotations.

Algorithm Overview:
    1. Pull 100 % of newly annotated tiles (since last training run).
    2. Calculate a replay quota = len(new) × replay_ratio.
    3. Distribute the quota evenly across all historical classes,
       boosting underrepresented classes and capping overrepresented ones.
    4. Randomly sample tiles per-class, deduplicate, and assemble a
       temporary training directory with the Ultralytics dataset.yaml.

Security:
    - Every file path read from the database is validated against
      DATA_DIR before any copy operation (path-traversal prevention).
    - All DB access uses parameterised SQLAlchemy queries.
    - Disk usage is capped via ``max_replay_dataset_size``.
"""

from __future__ import annotations

import json
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

from config import (
    ANNOTATED_IMAGES_DIR,
    ANNOTATED_LABELS_DIR,
    DATA_DIR,
    TEMP_TRAIN_DIR,
    settings,
    validate_path_within,
)
from models import Annotation, ClassRegistry, Tile, TrainingRun, TrainingStatus


class ReplayBufferError(Exception):
    """Raised when the replay buffer encounters a non-recoverable error."""


class ExperienceReplayBuffer:
    """
    Constructs balanced training datasets for incremental fine-tuning.

    Parameters
    ----------
    db : sqlalchemy.orm.Session
        An active database session.
    replay_ratio : float, optional
        Ratio of historical-to-new images (default from config).
    train_val_split : float, optional
        Fraction allocated to training vs. validation (default 0.9).
    max_dataset_size : int, optional
        Hard cap on total images to prevent disk exhaustion.
    seed : int | None, optional
        Random seed for reproducibility in sampling.
    """

    def __init__(
        self,
        db: Session,
        replay_ratio: float | None = None,
        train_val_split: float | None = None,
        max_dataset_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.db = db
        self.replay_ratio = replay_ratio or settings.replay_ratio
        self.train_val_split = train_val_split or settings.train_val_split
        self.max_dataset_size = max_dataset_size or settings.max_replay_dataset_size
        self._rng = random.Random(seed)

        # Eagerly load the class registry into memory for fast lookups.
        self._class_map: Dict[str, int] = self._load_class_map()

    # ── Public API ────────────────────────────────────────────

    def assemble_dataset(self) -> Path:
        """
        Main entry point.  Constructs the temporary training dataset.

        Returns
        -------
        Path
            Absolute path to the generated ``dataset.yaml``.

        Raises
        ------
        ReplayBufferError
            If no new tiles are available or the class registry is empty.
        """
        logger.info("──── Experience Replay Buffer: assembling dataset ────")

        # 1 ─ Gather new tiles
        new_tiles = self.get_new_tiles()
        if not new_tiles:
            raise ReplayBufferError(
                "No newly annotated tiles found since the last training run. "
                "Have the pathologist annotate more tiles before retraining."
            )
        logger.info(f"New tiles since last training: {len(new_tiles)}")

        # 2 ─ Gather historical replay tiles (stratified)
        replay_tiles = self._get_replay_tiles(len(new_tiles))
        logger.info(f"Historical replay tiles sampled: {len(replay_tiles)}")

        # 3 ─ Merge and deduplicate
        all_tile_ids: Set[int] = {t.id for t in new_tiles}
        combined_tiles: List[Tile] = list(new_tiles)
        for tile in replay_tiles:
            if tile.id not in all_tile_ids:
                all_tile_ids.add(tile.id)
                combined_tiles.append(tile)

        # 4 ─ Enforce hard cap
        if len(combined_tiles) > self.max_dataset_size:
            logger.warning(
                f"Dataset size ({len(combined_tiles)}) exceeds cap "
                f"({self.max_dataset_size}). Truncating replay data."
            )
            # Keep ALL new tiles, trim only historical
            replay_only = [t for t in combined_tiles if t not in new_tiles]
            self._rng.shuffle(replay_only)
            remaining_budget = self.max_dataset_size - len(new_tiles)
            combined_tiles = list(new_tiles) + replay_only[:remaining_budget]

        logger.info(f"Total dataset size: {len(combined_tiles)}")

        # 5 ─ Train / val split (stratified by class presence)
        train_tiles, val_tiles = self._stratified_train_val_split(combined_tiles)
        logger.info(f"Train: {len(train_tiles)} | Val: {len(val_tiles)}")

        # 6 ─ Write files to temp_train/
        self._prepare_temp_directory()
        self._copy_tiles_and_labels(train_tiles, split="train")
        self._copy_tiles_and_labels(val_tiles, split="val")

        # 7 ─ Generate dataset.yaml
        yaml_path = self._generate_dataset_yaml()
        logger.info(f"dataset.yaml written to: {yaml_path}")

        return yaml_path

    def get_new_tiles(self) -> List[Tile]:
        """
        Query all tiles annotated since the last completed training run.

        Returns every tile where ``is_annotated=True`` and whose
        ``updated_at`` timestamp is later than the most recent
        completed training run.
        """
        last_run = self._get_last_completed_run()
        cutoff = last_run.completed_at if last_run else datetime.min.replace(
            tzinfo=timezone.utc
        )

        tiles = (
            self.db.query(Tile)
            .filter(
                Tile.is_annotated.is_(True),
                Tile.updated_at > cutoff,
            )
            .all()
        )
        return tiles

    def get_class_distribution(
        self, before: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Return a histogram of distinct tiles per class label in
        historical annotations.

        Parameters
        ----------
        before : datetime, optional
            Only consider annotations created before this timestamp.
            Defaults to the cutoff of the last training run.

        Returns
        -------
        dict
            Mapping of class_label → count of distinct tiles.
        """
        last_run = self._get_last_completed_run()
        cutoff = before or (
            last_run.completed_at
            if last_run
            else datetime.max.replace(tzinfo=timezone.utc)
        )

        rows = (
            self.db.query(
                Annotation.class_label,
                sa_func.count(sa_func.distinct(Annotation.tile_id)),
            )
            .join(Tile, Annotation.tile_id == Tile.id)
            .filter(
                Tile.is_annotated.is_(True),
                Tile.updated_at <= cutoff,
            )
            .group_by(Annotation.class_label)
            .all()
        )
        return {label: count for label, count in rows}

    def cleanup(self) -> None:
        """
        Remove the temporary training directory to reclaim disk space.

        Safe to call even if the directory does not exist.
        """
        if TEMP_TRAIN_DIR.exists():
            shutil.rmtree(TEMP_TRAIN_DIR)
            logger.info(f"Cleaned up temporary training directory: {TEMP_TRAIN_DIR}")

    # ── Private Helpers ───────────────────────────────────────

    def _load_class_map(self) -> Dict[str, int]:
        """Load the class_name → yolo_index mapping from the database."""
        rows = self.db.query(ClassRegistry).all()
        mapping = {row.name: row.yolo_index for row in rows}
        if not mapping:
            logger.warning(
                "ClassRegistry is empty.  Seeding with default classes."
            )
            mapping = self._seed_default_classes()
        return mapping

    def _seed_default_classes(self) -> Dict[str, int]:
        """Insert default classes from config into the registry."""
        mapping: Dict[str, int] = {}
        for idx, name in enumerate(settings.default_classes):
            entry = ClassRegistry(name=name, yolo_index=idx)
            self.db.add(entry)
            mapping[name] = idx
        self.db.flush()
        logger.info(f"Seeded ClassRegistry with {len(mapping)} default classes.")
        return mapping

    def _get_last_completed_run(self) -> Optional[TrainingRun]:
        """Return the most recently completed training run, or None."""
        return (
            self.db.query(TrainingRun)
            .filter(TrainingRun.status == TrainingStatus.COMPLETED)
            .order_by(TrainingRun.completed_at.desc())
            .first()
        )

    def _get_replay_tiles(self, num_new: int) -> List[Tile]:
        """
        Perform stratified sampling of historical tiles.

        Steps:
            1. Compute total replay budget = num_new × replay_ratio.
            2. Get per-class tile counts for historical data.
            3. Distribute the budget across classes with boosting for
               underrepresented classes.
            4. Sample tiles per class.
        """
        target_count = int(math.ceil(num_new * self.replay_ratio))
        class_dist = self.get_class_distribution()

        if not class_dist:
            logger.warning("No historical annotations found for replay.")
            return []

        # ── Compute per-class quotas ──────────────────────────
        quotas = self._compute_class_quotas(class_dist, target_count)
        logger.debug(f"Per-class replay quotas: {quotas}")

        # ── Sample tiles per class ────────────────────────────
        sampled_tile_ids: Set[int] = set()
        sampled_tiles: List[Tile] = []

        last_run = self._get_last_completed_run()
        cutoff = last_run.completed_at if last_run else datetime.max.replace(
            tzinfo=timezone.utc
        )

        for class_label, quota in quotas.items():
            if quota <= 0:
                continue

            # Get tile IDs that contain this class in historical data
            tile_ids_for_class = (
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
            candidate_ids = [
                tid for (tid,) in tile_ids_for_class
                if tid not in sampled_tile_ids
            ]

            # Randomly sample up to quota
            sample_size = min(quota, len(candidate_ids))
            selected_ids = self._rng.sample(candidate_ids, sample_size)

            # Fetch actual Tile objects
            if selected_ids:
                tiles = (
                    self.db.query(Tile)
                    .filter(Tile.id.in_(selected_ids))
                    .all()
                )
                for tile in tiles:
                    if tile.id not in sampled_tile_ids:
                        sampled_tile_ids.add(tile.id)
                        sampled_tiles.append(tile)

        return sampled_tiles

    def _compute_class_quotas(
        self, class_dist: Dict[str, int], target_total: int
    ) -> Dict[str, int]:
        """
        Distribute the replay budget across classes with rebalancing.

        Underrepresented classes (below median count) receive a boosted
        quota; overrepresented classes are dampened. The final quotas
        are capped at the available tile count per class.

        Parameters
        ----------
        class_dist : dict
            Mapping of class_label → number of available tiles.
        target_total : int
            Total number of replay tiles to sample.

        Returns
        -------
        dict
            Mapping of class_label → number of tiles to sample.
        """
        num_classes = len(class_dist)
        if num_classes == 0:
            return {}

        base_quota = target_total / num_classes

        # Identify the median count to decide what is under/over represented
        sorted_counts = sorted(class_dist.values())
        median_count = sorted_counts[len(sorted_counts) // 2]

        quotas: Dict[str, int] = {}
        boost = settings.underrepresented_boost

        for label, available in class_dist.items():
            if available <= median_count:
                # Underrepresented → boost quota (but cap at what's available)
                raw_quota = base_quota * boost
            else:
                # Overrepresented → dampen to half of base, minimum 1
                raw_quota = max(base_quota * 0.5, 1)

            quotas[label] = min(int(math.ceil(raw_quota)), available)

        # Normalise so the sum doesn't wildly exceed target_total
        total_allocated = sum(quotas.values())
        if total_allocated > target_total * 1.5:
            scale = target_total / total_allocated
            quotas = {
                label: max(1, int(math.floor(q * scale)))
                for label, q in quotas.items()
            }

        return quotas

    def _stratified_train_val_split(
        self, tiles: List[Tile]
    ) -> Tuple[List[Tile], List[Tile]]:
        """
        Split tiles into train and val sets, preserving class ratios.

        Uses a simple approach: group tiles by their dominant class
        (the class with most annotations on that tile), then split
        each group proportionally.
        """
        # Determine dominant class per tile
        class_groups: Dict[str, List[Tile]] = defaultdict(list)
        for tile in tiles:
            if tile.annotations:
                # Count annotations per class for this tile
                class_counts: Dict[str, int] = defaultdict(int)
                for ann in tile.annotations:
                    class_counts[ann.class_label] += 1
                dominant = max(class_counts, key=class_counts.get)  # type: ignore[arg-type]
                class_groups[dominant].append(tile)
            else:
                class_groups["_unknown"].append(tile)

        train_set: List[Tile] = []
        val_set: List[Tile] = []

        for label, group in class_groups.items():
            self._rng.shuffle(group)
            split_idx = max(1, int(len(group) * self.train_val_split))
            train_set.extend(group[:split_idx])
            val_set.extend(group[split_idx:])

        # Ensure val set is never empty
        if not val_set and len(train_set) > 1:
            val_set.append(train_set.pop())

        return train_set, val_set

    def _prepare_temp_directory(self) -> None:
        """Create a clean temporary training directory structure."""
        # Remove any stale data from a previous failed run
        if TEMP_TRAIN_DIR.exists():
            shutil.rmtree(TEMP_TRAIN_DIR)

        for split in ("train", "val"):
            (TEMP_TRAIN_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (TEMP_TRAIN_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        logger.debug(f"Prepared temp training directory: {TEMP_TRAIN_DIR}")

    def _copy_tiles_and_labels(
        self, tiles: List[Tile], split: str
    ) -> None:
        """
        Copy tile images and generate YOLO label files into the
        temp training directory.

        Parameters
        ----------
        tiles : list[Tile]
            Tiles to process.
        split : str
            One of 'train' or 'val'.
        """
        images_dir = TEMP_TRAIN_DIR / "images" / split
        labels_dir = TEMP_TRAIN_DIR / "labels" / split

        for tile in tiles:
            # ── Resolve and validate image path ───────────────
            image_src = Path(tile.file_path)
            if not image_src.is_absolute():
                image_src = DATA_DIR / image_src

            try:
                image_src = validate_path_within(image_src, DATA_DIR)
            except ValueError as e:
                logger.error(f"Skipping tile {tile.id}: {e}")
                continue

            if not image_src.exists():
                logger.warning(
                    f"Tile image not found, skipping: {image_src}"
                )
                continue

            # ── Copy image ────────────────────────────────────
            dest_image = images_dir / image_src.name
            shutil.copy2(image_src, dest_image)

            # ── Generate YOLO label file ──────────────────────
            label_lines = self._generate_yolo_labels(tile)
            if label_lines:
                label_file = labels_dir / (image_src.stem + ".txt")
                label_file.write_text("\n".join(label_lines) + "\n")

    def _generate_yolo_labels(self, tile: Tile) -> List[str]:
        """
        Convert a tile's DB annotations into YOLO-format label lines.

        Returns
        -------
        list[str]
            YOLO lines: ``<class_idx> <x_c> <y_c> <w> <h>``
            Returns empty list if the tile has no annotations.
        """
        lines: List[str] = []
        for ann in tile.annotations:
            class_idx = self._class_map.get(ann.class_label)
            if class_idx is None:
                # Register previously unseen class dynamically
                class_idx = self._register_new_class(ann.class_label)

            lines.append(ann.to_yolo_line(class_idx))
        return lines

    def _register_new_class(self, class_label: str) -> int:
        """
        Dynamically register a new class discovered during annotation.

        Appends the class to ``ClassRegistry`` with the next available
        YOLO index.
        """
        max_idx = self.db.query(
            sa_func.max(ClassRegistry.yolo_index)
        ).scalar()
        new_idx = (max_idx or -1) + 1

        entry = ClassRegistry(name=class_label, yolo_index=new_idx)
        self.db.add(entry)
        self.db.flush()

        self._class_map[class_label] = new_idx
        logger.info(
            f"Registered new class: '{class_label}' → YOLO index {new_idx}"
        )
        return new_idx

    def _generate_dataset_yaml(self) -> Path:
        """
        Write the Ultralytics ``dataset.yaml`` configuration file.

        Returns the path to the generated YAML.
        """
        # Build ordered class names (sorted by YOLO index)
        sorted_classes = sorted(self._class_map.items(), key=lambda x: x[1])
        names = {idx: name for name, idx in sorted_classes}

        dataset_config = {
            "path": str(TEMP_TRAIN_DIR.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }

        yaml_path = TEMP_TRAIN_DIR / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                dataset_config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        return yaml_path

    # ── Reporting ─────────────────────────────────────────────

    def get_buffer_stats(self) -> Dict:
        """
        Return a summary of the current replay buffer state.

        Useful for the React UI to display dataset composition
        before the pathologist triggers training.
        """
        new_tiles = self.get_new_tiles()
        class_dist = self.get_class_distribution()
        target_replay = int(math.ceil(len(new_tiles) * self.replay_ratio))

        return {
            "new_tile_count": len(new_tiles),
            "replay_ratio": self.replay_ratio,
            "target_replay_count": target_replay,
            "total_estimated": len(new_tiles) + target_replay,
            "max_dataset_size": self.max_dataset_size,
            "historical_class_distribution": class_dist,
            "registered_classes": list(self._class_map.keys()),
        }
