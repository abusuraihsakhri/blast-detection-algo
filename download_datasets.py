"""
Download and register Roboflow datasets for the warm-start build.

This script downloads the curated set of Roboflow Universe datasets
required to bootstrap the blast detection model with strong baseline
knowledge. After downloading, each dataset is automatically registered
into the SQLite database for the Experience Replay Buffer.

Datasets:
    1. leukemia-nfxzn  (v1) — Blast-subtype detection  (Benign/Early/Pre/Pro)
    2. blast-cell-detection (v1) — Binary blast vs WBC detection
    3. bccd (v3) — General blood cell detection  (WBC/RBC/Platelets)
    4. blood-cell-znm2t (v1) — Comprehensive WBC differential (12+ classes)

Usage:
    1. Set ROBOFLOW_API_KEY in your .env file
    2. Run: python download_datasets.py
    3. Optionally pass --skip-download to just ingest already-downloaded data

Security Notes:
    - API key loaded from environment (.env), never hardcoded.
    - All downloaded paths are resolved to canonical absolute paths.
    - Download locations are constrained to PROJECT_ROOT.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from database import SessionLocal, init_db  # noqa: E402
from historical_models import AnnotationRecord  # noqa: E402
from import_historical_data import ingest_dataset  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#  Dataset Registry — curated for warm-start training
# ═══════════════════════════════════════════════════════════════════
#
# Each entry maps to a Roboflow Universe project. The class_map shows
# how the dataset's YOLO class IDs relate to our project's taxonomy.
#
#   Project Taxonomy (config.py):
#       0: blast
#       1: lymphocyte
#       2: rbc
#       3: artifact
#
#   These datasets span the spectrum from baseline blood-cell morphology
#   (broad distribution) to specialized blast-subtype detection (narrow,
#   high-value distribution), which together give the model a strong
#   warm start before any HITL annotations begin.

DATASET_REGISTRY = [
    {
        "name": "leukemia-nfxzn",
        "workspace": "tawfiq-islam-hfp3h",
        "project": "leukemia-nfxzn",
        "version": 1,
        "format": "yolov8",
        "folder_name": "leukemia-1",
        "description": "Primary blast subtype dataset (Benign/Early/Pre/Pro)",
        "class_map": {
            0: "Benign",
            1: "Early (Pre-B ALL)",
            2: "Pre (B-ALL)",
            3: "Pro (B-ALL)",
        },
        "priority": "CRITICAL",
    },
    {
        "name": "blast-cell-detection",
        "workspace": "yolov4-njwoe",
        "project": "blast-cell-detection",
        # This project splits train/valid/test across separate versions:
        # v26 = Train (3x aug, 3011 images)
        # v27 = Valid (72 images)
        # v28 = Test  (39 images)
        "multi_version": True,
        "versions": {
            "train": 26,
            "valid": 27,
            "test": 28,
        },
        "format": "yolov8",
        "folder_name": "blast-cell-detection-1",
        "description": "Binary blast vs WBC — teaches the model the blast boundary",
        "class_map": {
            0: "blast-cell",
            1: "wbc",
        },
        "priority": "HIGH",
    },
    {
        "name": "bccd-baseline",
        "workspace": "bccd-vhdu3",
        "project": "blood-cell-cella-bccd-lihfn",
        "version": 1,
        "format": "yolov8",
        "folder_name": "bccd-blood-cell-1",
        "description": "Classic BCCD — WBC/RBC/Platelet detection foundation",
        "class_map": {
            0: "Platelets",
            1: "RBC",
            2: "WBC",
        },
        "priority": "HIGH",
    },
    {
        "name": "blood-cell-comprehensive",
        "workspace": "aninp",
        "project": "blood-cell-znm2t",
        "version": 1,
        "format": "yolov8",
        "folder_name": "blood-cell-znm2t-1",
        "description": "Comprehensive WBC differential (12+ subtypes)",
        "class_map": {
            # This dataset has many WBC subtypes — excellent for teaching
            # the model fine-grained morphological differences.
        },
        "priority": "MEDIUM",
    },
    {
        "name": "acute-leukemia",
        "workspace": "yolov4-njwoe",
        "project": "acute-leukemia",
        "version": 1,
        "format": "yolov8",
        "folder_name": "acute-leukemia-1",
        "description": "Myeloid vs Lymphoid blast discrimination (AML + ALL)",
        "class_map": {
            0: "lymph-b",
            1: "myelo-b",
            2: "wbc",
        },
        "priority": "CRITICAL",
    },
    {
        "name": "myeloblast",
        "workspace": "nhung-o028n",
        "project": "myeloblast-fbliw",
        "version": 1,
        "format": "yolov8",
        "folder_name": "myeloblast-fbliw-1",
        "description": "Dedicated myeloblast detection in blood smears",
        "class_map": {
            0: "Myeloblast",
            1: "RBC",
            2: "WBC",
        },
        "priority": "HIGH",
    },
]


def get_api_key() -> str:
    """
    Retrieve the Roboflow API key from environment.

    Security: Never hardcode API keys. The key is loaded from the
    ROBOFLOW_API_KEY environment variable (set via .env file).

    Raises:
        SystemExit: If the key is not found.
    """
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print(
            "\n  ❌ ROBOFLOW_API_KEY not found in environment.\n"
            "     Add it to your .env file:\n"
            "       ROBOFLOW_API_KEY=your_key_here\n"
        )
        sys.exit(1)
    return key


def download_dataset(entry: dict, api_key: str) -> Path | None:
    """
    Download a single dataset from Roboflow Universe.

    Handles both standard (single version) and multi-version datasets
    where train/valid/test are split across separate Roboflow versions.

    Args:
        entry: A dict from DATASET_REGISTRY.
        api_key: The Roboflow API key.

    Returns:
        Path to the downloaded dataset directory, or None on failure.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  ❌ 'roboflow' package not installed. Run: pip install roboflow")
        sys.exit(1)

    import shutil

    folder_path = PROJECT_ROOT / entry["folder_name"]
    if folder_path.exists() and any(folder_path.iterdir()):
        print(f"  ⏭  Already exists: {folder_path.name}/ — skipping download")
        return folder_path

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(entry["workspace"]).project(entry["project"])

        if entry.get("multi_version"):
            # ── Multi-version: download each split separately and merge ──
            print(
                f"  ⬇  Multi-version download ({len(entry['versions'])} splits)..."
            )
            folder_path.mkdir(parents=True, exist_ok=True)

            for split_name, ver_num in entry["versions"].items():
                print(f"      Downloading v{ver_num} → {split_name}/")
                temp_dir = PROJECT_ROOT / f"_temp_{entry['folder_name']}_{split_name}"

                try:
                    dataset = project.version(ver_num).download(
                        entry["format"],
                        location=str(temp_dir),
                    )

                    # The Roboflow download may create various split dirs.
                    # We need to find the images and labels and move them
                    # into our canonical structure.
                    target_images = folder_path / split_name / "images"
                    target_labels = folder_path / split_name / "labels"
                    target_images.mkdir(parents=True, exist_ok=True)
                    target_labels.mkdir(parents=True, exist_ok=True)

                    # Search for images/labels in the downloaded structure
                    for src_split in ["train", "valid", "test", "."]:
                        src_images = temp_dir / src_split / "images" if src_split != "." else temp_dir / "images"
                        src_labels = temp_dir / src_split / "labels" if src_split != "." else temp_dir / "labels"

                        if src_images.is_dir():
                            for f in src_images.iterdir():
                                shutil.copy2(str(f), str(target_images / f.name))
                        if src_labels.is_dir():
                            for f in src_labels.iterdir():
                                shutil.copy2(str(f), str(target_labels / f.name))

                finally:
                    # Clean up temp directory
                    if temp_dir.exists():
                        shutil.rmtree(str(temp_dir), ignore_errors=True)

            print(f"  ✅ Merged to: {folder_path}")
            return folder_path

        else:
            # ── Standard single-version download ──
            print(f"  ⬇  Downloading {entry['name']} (v{entry['version']})...")
            print(f"      Project: {entry['workspace']}/{entry['project']}")

            dataset = project.version(entry["version"]).download(
                entry["format"],
                location=str(folder_path),
            )
            print(f"  ✅ Downloaded to: {dataset.location}")
            return Path(dataset.location)

    except Exception as exc:
        print(f"  ❌ Failed to download {entry['name']}: {exc}")
        return None


def verify_dataset_structure(dataset_path: Path) -> bool:
    """
    Validate that a downloaded dataset has the expected YOLO structure.

    Checks for at least one of train/valid/test containing images/.
    """
    for split in ["train", "valid", "test"]:
        images_dir = dataset_path / split / "images"
        if images_dir.is_dir() and any(images_dir.iterdir()):
            return True

    print(f"  ⚠  Invalid structure: {dataset_path.name}/ — no images found")
    return False


def print_dataset_summary(dataset_path: Path, entry: dict) -> None:
    """Print a summary of what's in the dataset."""
    total_images = 0
    for split in ["train", "valid", "test"]:
        images_dir = dataset_path / split / "images"
        if images_dir.is_dir():
            count = len(list(images_dir.glob("*.*")))
            total_images += count
            print(f"      {split}: {count} images")

    print(f"      Total: {total_images} images")
    if entry.get("class_map"):
        print(f"      Classes: {entry['class_map']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & ingest Roboflow datasets for warm-start training."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloads; only ingest already-downloaded datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Specific dataset names to process (default: all).",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  🔬 Pathology Warm-Start Dataset Pipeline")
    print("=" * 65)

    # ── Filter dataset registry ──────────────────────────────────
    datasets_to_process = DATASET_REGISTRY
    if args.datasets:
        datasets_to_process = [
            d for d in DATASET_REGISTRY if d["name"] in args.datasets
        ]
        if not datasets_to_process:
            print(f"\n  ❌ No matching datasets for: {args.datasets}")
            print(
                f"     Available: {[d['name'] for d in DATASET_REGISTRY]}"
            )
            sys.exit(1)

    # ── Phase 1: Download from Roboflow ──────────────────────────
    api_key = None
    if not args.skip_download:
        api_key = get_api_key()

    downloaded_paths: dict[str, Path] = {}

    print(
        f"\n{'─' * 65}\n"
        f"  Phase 1: Download ({len(datasets_to_process)} datasets)\n"
        f"{'─' * 65}"
    )

    for i, entry in enumerate(datasets_to_process, 1):
        print(
            f"\n  [{i}/{len(datasets_to_process)}] "
            f"{entry['name']} [{entry['priority']}]"
        )
        print(f"      {entry['description']}")

        folder_path = PROJECT_ROOT / entry["folder_name"]

        if args.skip_download:
            if folder_path.exists():
                print(f"  ⏭  Using existing: {folder_path.name}/")
                downloaded_paths[entry["name"]] = folder_path
            else:
                print(f"  ⚠  Not found: {folder_path.name}/ — skipping")
        else:
            result = download_dataset(entry, api_key)
            if result:
                downloaded_paths[entry["name"]] = result

    # ── Phase 2: Verify & Summarise ──────────────────────────────
    print(
        f"\n{'─' * 65}\n"
        f"  Phase 2: Verify dataset structure\n"
        f"{'─' * 65}"
    )

    valid_datasets: dict[str, Path] = {}
    for name, path in downloaded_paths.items():
        entry = next(d for d in DATASET_REGISTRY if d["name"] == name)
        print(f"\n  📂 {name}:")
        if verify_dataset_structure(path):
            print_dataset_summary(path, entry)
            valid_datasets[name] = path
        else:
            print(f"      ❌ Skipping invalid dataset")

    # ── Phase 3: Ingest into database ────────────────────────────
    print(
        f"\n{'─' * 65}\n"
        f"  Phase 3: Database ingestion ({len(valid_datasets)} datasets)\n"
        f"{'─' * 65}"
    )

    init_db()

    for name, path in valid_datasets.items():
        print(f"\n  📥 Ingesting {name}...")
        ingest_dataset(str(path), name)

    # ── Final Summary ────────────────────────────────────────────
    session = SessionLocal()
    try:
        total_records = session.query(AnnotationRecord).count()
        datasets = (
            session.query(
                AnnotationRecord.dataset_name,
                AnnotationRecord.dataset_name,
            )
            .distinct()
            .all()
        )
        dataset_names = [d[0] for d in datasets]

        # Count per dataset
        from sqlalchemy import func

        dataset_counts = (
            session.query(
                AnnotationRecord.dataset_name,
                func.count(AnnotationRecord.id),
            )
            .group_by(AnnotationRecord.dataset_name)
            .all()
        )
    finally:
        session.close()

    print(f"\n{'═' * 65}")
    print(f"  📊 Final Database Summary")
    print(f"{'═' * 65}")
    print(f"  Total records: {total_records}")
    for ds_name, count in dataset_counts:
        print(f"    • {ds_name}: {count} images")
    print(f"\n  🚀 Warm-start data pipeline complete.")
    print(f"     The Experience Replay Buffer is ready for training.")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
