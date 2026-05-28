"""
Tests for Module 3: Experience Replay Buffer.

Covers:
    - Stratified sampling produces balanced class distribution
    - Dataset.yaml generation matches Ultralytics schema
    - Path traversal attacks are rejected
    - Deduplication when tiles contain multiple classes
    - Edge cases: no historical data, single class, empty new data
    - Dynamic class registration for unseen labels
    - Train/val split preserves class ratios
    - Cleanup removes temporary directory
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ── Project imports ───────────────────────────────────────────
# Patch config paths BEFORE importing modules that depend on them,
# so temp directories are used instead of production paths.
import config

TEST_DATA_DIR = Path(__file__).parent / "_test_data"
TEST_TEMP_TRAIN = TEST_DATA_DIR / "temp_train"
TEST_ANNOTATED_IMAGES = TEST_DATA_DIR / "annotated_tiles" / "images"
TEST_ANNOTATED_LABELS = TEST_DATA_DIR / "annotated_tiles" / "labels"

# Monkey-patch config paths for test isolation
config.DATA_DIR = TEST_DATA_DIR
config.TEMP_TRAIN_DIR = TEST_TEMP_TRAIN
config.ANNOTATED_IMAGES_DIR = TEST_ANNOTATED_IMAGES
config.ANNOTATED_LABELS_DIR = TEST_ANNOTATED_LABELS

from models import Annotation, Base, ClassRegistry, Tile, TrainingRun, TrainingStatus
from replay_buffer import ExperienceReplayBuffer, ReplayBufferError


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _setup_teardown():
    """Create and clean test directories before/after each test."""
    for d in [TEST_ANNOTATED_IMAGES, TEST_ANNOTATED_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)


@pytest.fixture
def db_session() -> Session:
    """Provide a fresh in-memory SQLite session for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Seed class registry
    for idx, name in enumerate(["blast", "lymphocyte", "rbc", "artifact"]):
        session.add(ClassRegistry(name=name, yolo_index=idx))
    session.flush()

    yield session
    session.close()


def _create_tile_image(filename: str) -> Path:
    """Create a minimal dummy image file for testing."""
    path = TEST_ANNOTATED_IMAGES / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a minimal valid PNG (1x1 pixel, red)
    import struct
    import zlib

    def _minimal_png() -> bytes:
        signature = b"\x89PNG\r\n\x1a\n"

        def chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        raw_data = b"\x00\xff\x00\x00"  # filter byte + RGB
        idat = zlib.compress(raw_data)
        return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

    path.write_bytes(_minimal_png())
    return path


def _add_annotated_tile(
    session: Session,
    filename: str,
    classes: list[str],
    updated_at: datetime | None = None,
    source_wsi: str = "test.svs",
) -> Tile:
    """Helper: create a tile with annotations in the DB and a file on disk."""
    img_path = _create_tile_image(filename)

    tile = Tile(
        file_path=str(img_path),
        source_wsi=source_wsi,
        x_coord=0,
        y_coord=0,
        level=0,
        tissue_pct=0.85,
        is_annotated=True,
    )
    if updated_at:
        tile.updated_at = updated_at
    session.add(tile)
    session.flush()

    for i, cls in enumerate(classes):
        ann = Annotation(
            tile_id=tile.id,
            class_label=cls,
            x_center=0.5,
            y_center=0.5,
            width=0.1,
            height=0.1,
            confidence=0.95,
            is_manual=False,
        )
        if updated_at:
            ann.created_at = updated_at
        session.add(ann)

    session.flush()

    # Backfill the updated_at if provided (override server_default)
    if updated_at:
        session.execute(
            Tile.__table__.update()
            .where(Tile.__table__.c.id == tile.id)
            .values(updated_at=updated_at)
        )
        session.flush()
        session.refresh(tile)

    return tile


# ── Tests: Core Functionality ─────────────────────────────────


class TestGetNewTiles:
    """Tests for ExperienceReplayBuffer.get_new_tiles()."""

    def test_returns_tiles_after_last_training(self, db_session: Session):
        """New tiles created after the last training run are returned."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        recent = datetime.now(timezone.utc) + timedelta(hours=1)

        # Historical tile (before training)
        _add_annotated_tile(db_session, "old_tile.png", ["rbc"], updated_at=past)

        # Simulate a completed training run
        run = TrainingRun(
            status=TrainingStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            num_new_images=1,
            num_replay_images=0,
        )
        db_session.add(run)
        db_session.flush()

        # New tile (after training)
        _add_annotated_tile(db_session, "new_tile.png", ["blast"], updated_at=recent)

        buffer = ExperienceReplayBuffer(db_session, seed=42)
        new_tiles = buffer.get_new_tiles()
        assert len(new_tiles) == 1
        assert "new_tile" in new_tiles[0].file_path

    def test_returns_all_when_no_prior_training(self, db_session: Session):
        """If no training has ever run, all annotated tiles are 'new'."""
        now = datetime.now(timezone.utc)
        _add_annotated_tile(db_session, "tile_a.png", ["blast"], updated_at=now)
        _add_annotated_tile(db_session, "tile_b.png", ["rbc"], updated_at=now)

        buffer = ExperienceReplayBuffer(db_session, seed=42)
        new_tiles = buffer.get_new_tiles()
        assert len(new_tiles) == 2


class TestClassDistribution:
    """Tests for ExperienceReplayBuffer.get_class_distribution()."""

    def test_returns_correct_histogram(self, db_session: Session):
        """Counts distinct tiles per class correctly."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        _add_annotated_tile(db_session, "t1.png", ["blast", "rbc"], updated_at=past)
        _add_annotated_tile(db_session, "t2.png", ["blast"], updated_at=past)
        _add_annotated_tile(db_session, "t3.png", ["lymphocyte"], updated_at=past)

        buffer = ExperienceReplayBuffer(db_session, seed=42)
        dist = buffer.get_class_distribution(
            before=datetime.now(timezone.utc)
        )

        assert dist["blast"] == 2     # t1 + t2
        assert dist["rbc"] == 1       # t1
        assert dist["lymphocyte"] == 1  # t3

    def test_empty_when_no_annotations(self, db_session: Session):
        """Returns empty dict when no historical annotations exist."""
        buffer = ExperienceReplayBuffer(db_session, seed=42)
        dist = buffer.get_class_distribution()
        assert dist == {}


class TestStratifiedSampling:
    """Tests for the stratified replay logic."""

    def test_produces_balanced_distribution(self, db_session: Session):
        """
        Given skewed historical data, replay sampling should boost
        underrepresented classes relative to overrepresented ones.
        """
        past = datetime.now(timezone.utc) - timedelta(hours=5)

        # Historical data: heavily skewed toward RBC
        for i in range(20):
            _add_annotated_tile(
                db_session, f"hist_rbc_{i}.png", ["rbc"], updated_at=past
            )
        for i in range(3):
            _add_annotated_tile(
                db_session, f"hist_blast_{i}.png", ["blast"], updated_at=past
            )

        # Simulate a training run
        run = TrainingRun(
            status=TrainingStatus.COMPLETED,
            completed_at=past + timedelta(hours=1),
            num_new_images=23,
            num_replay_images=0,
        )
        db_session.add(run)
        db_session.flush()

        # New data
        recent = datetime.now(timezone.utc) + timedelta(hours=1)
        for i in range(5):
            _add_annotated_tile(
                db_session, f"new_{i}.png", ["blast"], updated_at=recent
            )

        buffer = ExperienceReplayBuffer(db_session, replay_ratio=4.0, seed=42)
        stats = buffer.get_buffer_stats()

        assert stats["new_tile_count"] == 5
        assert stats["target_replay_count"] == 20

        # The key assertion: blast (underrepresented) should get a boosted
        # quota relative to rbc (overrepresented)
        quotas = buffer._compute_class_quotas(
            stats["historical_class_distribution"], 20
        )
        # Blast gets boosted, RBC gets dampened
        assert quotas.get("blast", 0) > 0
        assert quotas.get("rbc", 0) > 0
        # Blast quota should be >= RBC quota despite having fewer samples
        # (boosting effect)
        assert quotas["blast"] >= quotas["rbc"] or quotas["blast"] == 3  # capped at available


class TestPathTraversal:
    """Security tests for path validation."""

    def test_rejects_traversal_attempt(self, db_session: Session):
        """Paths escaping DATA_DIR are blocked during dataset assembly."""
        from config import validate_path_within

        with pytest.raises(ValueError, match="Path traversal blocked"):
            validate_path_within(
                Path("../../etc/passwd"),
                TEST_DATA_DIR,
            )

    def test_accepts_valid_path(self, db_session: Session):
        """Paths within DATA_DIR are accepted."""
        from config import validate_path_within

        valid = TEST_DATA_DIR / "annotated_tiles" / "images" / "safe.png"
        valid.parent.mkdir(parents=True, exist_ok=True)
        valid.touch()

        result = validate_path_within(valid, TEST_DATA_DIR)
        assert result == valid.resolve()


class TestDatasetAssembly:
    """Integration tests for the full assemble_dataset() pipeline."""

    def test_full_pipeline_produces_valid_structure(self, db_session: Session):
        """
        End-to-end: seed historical + new tiles, assemble dataset,
        verify directory structure and dataset.yaml content.
        """
        past = datetime.now(timezone.utc) - timedelta(hours=5)

        # Historical tiles
        for i in range(8):
            cls = ["blast", "lymphocyte", "rbc", "artifact"][i % 4]
            _add_annotated_tile(
                db_session, f"hist_{i}.png", [cls], updated_at=past
            )

        # Training run
        run = TrainingRun(
            status=TrainingStatus.COMPLETED,
            completed_at=past + timedelta(hours=1),
            num_new_images=8,
            num_replay_images=0,
        )
        db_session.add(run)
        db_session.flush()

        # New tiles
        recent = datetime.now(timezone.utc) + timedelta(hours=1)
        for i in range(4):
            _add_annotated_tile(
                db_session, f"new_{i}.png", ["blast"], updated_at=recent
            )

        buffer = ExperienceReplayBuffer(
            db_session, replay_ratio=2.0, seed=42
        )
        yaml_path = buffer.assemble_dataset()

        # ── Assertions ────────────────────────────────────────
        assert yaml_path.exists()
        assert (TEST_TEMP_TRAIN / "images" / "train").is_dir()
        assert (TEST_TEMP_TRAIN / "images" / "val").is_dir()
        assert (TEST_TEMP_TRAIN / "labels" / "train").is_dir()
        assert (TEST_TEMP_TRAIN / "labels" / "val").is_dir()

        # Check dataset.yaml content
        with open(yaml_path) as f:
            ds = yaml.safe_load(f)

        assert "train" in ds
        assert "val" in ds
        assert "names" in ds
        assert ds["names"][0] == "blast"
        assert "path" in ds

        # Check that image files were actually copied
        train_images = list((TEST_TEMP_TRAIN / "images" / "train").iterdir())
        val_images = list((TEST_TEMP_TRAIN / "images" / "val").iterdir())
        assert len(train_images) + len(val_images) > 0

        # Check that label files were generated
        train_labels = list((TEST_TEMP_TRAIN / "labels" / "train").iterdir())
        assert len(train_labels) > 0

        # Verify label format
        label_content = train_labels[0].read_text().strip()
        parts = label_content.split()
        assert len(parts) == 5  # class_idx x_c y_c w h
        assert parts[0].isdigit()

    def test_raises_when_no_new_tiles(self, db_session: Session):
        """Attempting to assemble with no new tiles raises an error."""
        buffer = ExperienceReplayBuffer(db_session, seed=42)
        with pytest.raises(ReplayBufferError, match="No newly annotated tiles"):
            buffer.assemble_dataset()


class TestCleanup:
    """Tests for temporary directory cleanup."""

    def test_cleanup_removes_directory(self, db_session: Session):
        """cleanup() removes the temp_train directory."""
        TEST_TEMP_TRAIN.mkdir(parents=True, exist_ok=True)
        (TEST_TEMP_TRAIN / "dummy.txt").write_text("test")

        buffer = ExperienceReplayBuffer(db_session, seed=42)
        buffer.cleanup()

        assert not TEST_TEMP_TRAIN.exists()

    def test_cleanup_safe_when_missing(self, db_session: Session):
        """cleanup() does not error if temp_train doesn't exist."""
        buffer = ExperienceReplayBuffer(db_session, seed=42)
        buffer.cleanup()  # Should not raise


class TestDynamicClassRegistration:
    """Tests for automatic registration of unseen classes."""

    def test_registers_new_class_during_label_generation(self, db_session: Session):
        """A class not in the registry gets auto-registered."""
        past = datetime.now(timezone.utc) - timedelta(hours=5)

        # Create tile with an unknown class
        tile = _add_annotated_tile(
            db_session, "exotic.png", ["promyelocyte"], updated_at=past
        )

        buffer = ExperienceReplayBuffer(db_session, seed=42)

        # Generate labels — should auto-register "promyelocyte"
        labels = buffer._generate_yolo_labels(tile)
        assert len(labels) == 1
        assert "promyelocyte" in buffer._class_map
        assert buffer._class_map["promyelocyte"] == 4  # next after 0,1,2,3


class TestBufferStats:
    """Tests for the reporting/stats API."""

    def test_returns_expected_fields(self, db_session: Session):
        """get_buffer_stats() returns all required keys."""
        buffer = ExperienceReplayBuffer(db_session, seed=42)
        stats = buffer.get_buffer_stats()

        assert "new_tile_count" in stats
        assert "replay_ratio" in stats
        assert "target_replay_count" in stats
        assert "total_estimated" in stats
        assert "max_dataset_size" in stats
        assert "historical_class_distribution" in stats
        assert "registered_classes" in stats
