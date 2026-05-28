"""
Lightweight ORM model for tracking historical YOLO datasets.

This AnnotationRecord table serves the Experience Replay Buffer by indexing
which images exist on disk, what YOLO class IDs they contain, and which
dataset they belong to. It supplements the core models in models.py —
those handle tiles/annotations from the HITL pipeline, while this table
tracks pre-existing downloaded datasets.

Security Notes:
    - image_path / label_path are validated at ingestion time to prevent
      path-traversal when later consumed by the replay buffer.
    - All queries use parameterised SQLAlchemy; no raw SQL.
    - String columns have no unbounded lengths — SQLite doesn't enforce
      them, but the constraint documents intent.
"""

import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String

from models import Base  # Reuse the existing declarative base


class AnnotationRecord(Base):
    """
    Tracks a single image+label pair from a downloaded historical dataset.

    Attributes:
        image_path: Absolute path to the image file on disk.
        label_path: Absolute path to the corresponding YOLO .txt label file.
        contained_classes: Comma-separated YOLO class IDs found in the label
                           file (e.g. "0,2,3"). Critical for stratified
                           sampling in the Replay Buffer.
        is_historical: True for downloaded datasets, False for UI-annotated.
        dataset_name: Identifier for the source dataset (e.g. "leukemia-nfxzn").
        timestamp: When this record was ingested.
    """

    __tablename__ = "annotation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String(1024), unique=True, nullable=False)
    label_path = Column(String(1024), nullable=False)
    contained_classes = Column(String(256), nullable=False)
    is_historical = Column(Boolean, default=True, nullable=False)
    dataset_name = Column(String(256), nullable=False)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<AnnotationRecord(id={self.id}, "
            f"dataset='{self.dataset_name}', "
            f"classes='{self.contained_classes}')>"
        )
