"""Central configuration for the blast-detection review and training workflow."""

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_TILES_DIR = DATA_DIR / "raw_tiles"
ANNOTATED_IMAGES_DIR = DATA_DIR / "annotated_tiles" / "images"
ANNOTATED_LABELS_DIR = DATA_DIR / "annotated_tiles" / "labels"
TEMP_TRAIN_DIR = DATA_DIR / "temp_train"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATABASE_PATH = PROJECT_ROOT / "blast_algo.db"


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables and optional .env."""

    database_url: str = Field(default=f"sqlite:///{DATABASE_PATH}")

    tile_size: int = Field(default=512, ge=64, le=2048)
    magnification_level: int = Field(default=0, ge=0)
    background_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    replay_ratio: float = Field(default=4.0, gt=0.0)
    train_val_split: float = Field(default=0.9, gt=0.0, lt=1.0)
    max_replay_dataset_size: int = Field(default=5000, ge=10)
    underrepresented_boost: float = Field(default=1.5, ge=1.0)

    max_epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    base_model_name: str = Field(default="yolov8n.pt")
    active_model_path: Path = Field(default=MODELS_DIR / "active_model.pt")

    warm_start_epochs: int = Field(default=20, ge=1, le=200)
    warm_start_imgsz: int = Field(default=640, ge=320, le=1280)
    min_map50_improvement: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum validation mAP@50 improvement required before replacing the active model.",
    )

    inference_overlap_pct: float = Field(default=0.25, ge=0.0, lt=1.0)
    inference_conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_tiles_to_save_per_wsi: int = Field(default=5000, ge=1)
    save_empty_tiles: bool = True

    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    max_upload_pixels: int = Field(default=50_000_000, ge=1_000_000)
    sandbox_retention_seconds: int = Field(default=24 * 60 * 60, ge=60)
    allowed_upload_mime_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"]
    )

    default_classes: List[str] = Field(
        default=[
            "Benign",
            "Early",
            "Pre",
            "Pro",
            "WBC",
            "RBC",
            "Platelets",
            "Myeloblast",
            "Lymphoblast",
            "Atypical",
        ]
    )

    model_config = {
        "env_prefix": "BLAST_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = AppSettings()


def validate_path_within(file_path: Path, allowed_root: Path) -> Path:
    """Resolve a path and reject paths that escape ``allowed_root``."""
    resolved = file_path.resolve()
    allowed = allowed_root.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"Path traversal blocked: '{file_path}' resolves to '{resolved}', "
            f"which is outside '{allowed}'."
        ) from exc
    return resolved


def ensure_directories() -> None:
    """Create runtime directories required by local workflows."""
    for directory in (
        RAW_TILES_DIR,
        ANNOTATED_IMAGES_DIR,
        ANNOTATED_LABELS_DIR,
        TEMP_TRAIN_DIR,
        MODELS_DIR,
        LOGS_DIR,
        DATA_DIR / "sandbox_uploads",
    ):
        directory.mkdir(parents=True, exist_ok=True)
