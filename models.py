"""
SQLAlchemy ORM models for the HITL Active Learning Pipeline.

Tables:
    - Tile: Extracted WSI tiles and their metadata.
    - Annotation: Individual bounding box annotations per tile.
    - TrainingRun: Audit log of every fine-tuning cycle.
    - ClassRegistry: Canonical class-to-YOLO-index mapping.

Security Notes:
    - All string fields have explicit length limits to prevent oversized payloads.
    - Foreign keys enforce referential integrity (enabled via PRAGMA in database.py).
    - Timestamps use server-side defaults to prevent client-side manipulation.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ── Base ──────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# ── Enums ─────────────────────────────────────────────────────
class TrainingStatus(str, enum.Enum):
    """Status of a training run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Tile ──────────────────────────────────────────────────────
class Tile(Base):
    """
    Represents a single tile extracted from a Whole Slide Image.

    Each tile is a fixed-size crop (e.g., 512x512) at a specific
    magnification level, with metadata for tissue content and
    annotation status.
    """

    __tablename__ = "tiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(
        String(512),
        unique=True,
        nullable=False,
        index=True,
        comment="Relative path from DATA_DIR to tile image file.",
    )
    source_wsi = Column(
        String(256),
        nullable=False,
        index=True,
        comment="Original WSI filename (e.g., 'patient_001.svs').",
    )
    x_coord = Column(
        Integer,
        nullable=False,
        comment="Tile origin X coordinate at pyramid level 0.",
    )
    y_coord = Column(
        Integer,
        nullable=False,
        comment="Tile origin Y coordinate at pyramid level 0.",
    )
    level = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Pyramid level at which tile was extracted.",
    )
    tissue_pct = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Percentage of tile area containing tissue (0.0–1.0).",
    )
    is_annotated = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="True if pathologist has reviewed and saved annotations.",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    annotations = relationship(
        "Annotation",
        back_populates="tile",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Composite index for efficient spatial queries
    __table_args__ = (
        Index("ix_tile_source_coords", "source_wsi", "x_coord", "y_coord"),
    )

    def __repr__(self) -> str:
        return (
            f"<Tile(id={self.id}, wsi='{self.source_wsi}', "
            f"pos=({self.x_coord},{self.y_coord}), "
            f"annotated={self.is_annotated})>"
        )


# ── Annotation ────────────────────────────────────────────────
class Annotation(Base):
    """
    A single bounding box annotation on a tile.

    Stores YOLO-format normalized coordinates and tracks whether
    the annotation was machine-generated or manually drawn by a
    pathologist.
    """

    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tile_id = Column(
        Integer,
        ForeignKey("tiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    class_label = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Human-readable class name (e.g., 'blast', 'rbc').",
    )
    x_center = Column(
        Float,
        nullable=False,
        comment="YOLO normalized X center (0.0–1.0).",
    )
    y_center = Column(
        Float,
        nullable=False,
        comment="YOLO normalized Y center (0.0–1.0).",
    )
    width = Column(
        Float,
        nullable=False,
        comment="YOLO normalized width (0.0–1.0).",
    )
    height = Column(
        Float,
        nullable=False,
        comment="YOLO normalized height (0.0–1.0).",
    )
    confidence = Column(
        Float,
        nullable=True,
        comment="Model confidence score (null for manual annotations).",
    )
    is_manual = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if drawn/corrected by pathologist.",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    tile = relationship("Tile", back_populates="annotations")

    def __repr__(self) -> str:
        return (
            f"<Annotation(id={self.id}, tile={self.tile_id}, "
            f"class='{self.class_label}', manual={self.is_manual})>"
        )

    def to_yolo_line(self, class_index: int) -> str:
        """
        Format this annotation as a YOLO label line.

        Args:
            class_index: Numeric class ID from ClassRegistry.

        Returns:
            String in format: ``<class_id> <x_center> <y_center> <width> <height>``
        """
        return (
            f"{class_index} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


# ── Training Run ──────────────────────────────────────────────
class TrainingRun(Base):
    """
    Audit record for each fine-tuning cycle.

    Tracks the data composition, resulting model, and performance
    metrics to enable lineage tracking and rollback decisions.
    """

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Null while training is in progress.",
    )
    status = Column(
        Enum(TrainingStatus),
        nullable=False,
        default=TrainingStatus.PENDING,
        index=True,
    )
    num_new_images = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Count of newly annotated images in this training batch.",
    )
    num_replay_images = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Count of historical replay images in this training batch.",
    )
    model_path = Column(
        String(512),
        nullable=True,
        comment="Path to best.pt produced by this run.",
    )
    final_map50 = Column(
        Float,
        nullable=True,
        comment="mAP@0.5 on validation set after training.",
    )
    config_json = Column(
        Text,
        nullable=True,
        comment="Serialized training hyperparameters (JSON string).",
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingRun(id={self.id}, status={self.status.value}, "
            f"new={self.num_new_images}, replay={self.num_replay_images})>"
        )


# ── Class Registry ────────────────────────────────────────────
class ClassRegistry(Base):
    """
    Canonical mapping of class names to YOLO numeric indices.

    This table is the single source of truth for class labels.
    New classes discovered during annotation are appended here
    with the next available index.
    """

    __tablename__ = "class_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Canonical class name (e.g., 'blast').",
    )
    yolo_index = Column(
        Integer,
        unique=True,
        nullable=False,
        comment="Numeric class ID used in YOLO .txt labels.",
    )
    first_seen = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("name", name="uq_class_name"),
        UniqueConstraint("yolo_index", name="uq_yolo_index"),
    )

    def __repr__(self) -> str:
        return f"<ClassRegistry(name='{self.name}', idx={self.yolo_index})>"
