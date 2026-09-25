"""
Ingestion script for registering downloaded YOLO datasets into the
pathology_active_learning SQLite database.

Scans standard YOLO directory structures (train/, valid/, test/) and
inserts AnnotationRecord rows for source-dataset provenance and label
metadata. Incremental replay uses reviewed Tile/Annotation records instead.

Usage:
    python import_historical_data.py

Security Notes:
    - All file paths are resolved to absolute, canonical forms to prevent
      directory-traversal issues when later consumed.
    - Duplicate detection uses the unique constraint on image_path.
    - Label files are read in text mode with explicit encoding to avoid
      codec-based injection vectors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import SessionLocal, init_db  # noqa: E402
from historical_models import AnnotationRecord  # noqa: E402


def parse_yolo_labels(label_path: str) -> list[str]:
    """
    Read a YOLO .txt label file and return sorted unique class IDs.

    Each line in a YOLO label file has the format:
        <class_id> <x_center> <y_center> <width> <height>

    Returns:
        Sorted list of unique class ID strings (e.g. ["0", "2", "3"]).
        Empty list if the file doesn't exist or is empty.
    """
    if not os.path.exists(label_path):
        return []

    unique_classes: set[str] = set()
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = parts[0]
                    # Validate it's actually a non-negative integer
                    if class_id.isdigit():
                        unique_classes.add(class_id)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"  [WARN] Could not read label file {label_path}: {exc}")
        return []

    return sorted(unique_classes, key=int)


def ingest_dataset(dataset_root_path: str, dataset_name: str) -> None:
    """
    Scan a YOLO dataset folder and insert records into the database.

    Traverses the standard YOLO splits (train/, valid/, test/) looking
    for images and their corresponding label files. Records are only
    inserted if they don't already exist (idempotent re-runs are safe).

    Args:
        dataset_root_path: Path to the dataset root containing split folders.
        dataset_name: Human-readable name for this dataset.
    """
    session = SessionLocal()
    records_added = 0
    records_skipped = 0

    # Standard YOLO data splits
    splits = ["train", "valid", "test"]
    valid_image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    try:
        for split in splits:
            images_dir = os.path.join(dataset_root_path, split, "images")
            labels_dir = os.path.join(dataset_root_path, split, "labels")

            if not os.path.isdir(images_dir):
                print(f"  [SKIP] No images directory: {images_dir}")
                continue

            image_files = sorted(os.listdir(images_dir))
            print(f"  Scanning {split}/images/ — {len(image_files)} files found")

            for img_file in image_files:
                ext = os.path.splitext(img_file)[1].lower()
                if ext not in valid_image_extensions:
                    continue

                # Resolve to absolute canonical paths
                img_path = str(Path(os.path.join(images_dir, img_file)).resolve())

                # Find corresponding label file
                base_name = os.path.splitext(img_file)[0]
                label_path = str(
                    Path(os.path.join(labels_dir, f"{base_name}.txt")).resolve()
                )

                # Extract class IDs for stratified sampling metadata
                classes_present = parse_yolo_labels(label_path)
                class_str = ",".join(classes_present) if classes_present else ""

                # Idempotent: skip if already ingested
                existing = (
                    session.query(AnnotationRecord)
                    .filter_by(image_path=img_path)
                    .first()
                )
                if existing:
                    records_skipped += 1
                    continue

                new_record = AnnotationRecord(
                    image_path=img_path,
                    label_path=label_path,
                    contained_classes=class_str,
                    is_historical=True,
                    dataset_name=dataset_name,
                )
                session.add(new_record)
                records_added += 1

        session.commit()
        print(
            f"\n  ✅ Dataset '{dataset_name}': "
            f"{records_added} new records ingested, "
            f"{records_skipped} duplicates skipped."
        )

    except Exception:
        session.rollback()
        print(f"\n  ❌ Error ingesting '{dataset_name}'. Transaction rolled back.")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  Historical Dataset Ingestion")
    print("=" * 60)

    # Initialise the database (creates tables if they don't exist)
    init_db()

    # ── BCCD Baseline Dataset ────────────────────────────────────
    # NOTE: The BCCD-1 download failed (roboflow.zip contains an error).
    # Uncomment the line below once the dataset is re-downloaded:
    #
    # bccd_path = str(PROJECT_ROOT / "BCCD-1")
    # print(f"\n[1/2] Ingesting BCCD baseline from: {bccd_path}")
    # ingest_dataset(bccd_path, "bccd_baseline")

    # ── Leukemia Specialised Dataset ─────────────────────────────
    leukemia_path = str(PROJECT_ROOT / "leukemia-1")
    print(f"\n[1/1] Ingesting leukemia dataset from: {leukemia_path}")
    ingest_dataset(leukemia_path, "leukemia-nfxzn")

    print("\n" + "=" * 60)
    print("  Source-dataset provenance table is ready.")
    print("=" * 60)
