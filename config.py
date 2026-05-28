"""
Centralized configuration for the HITL Active Learning Pipeline.

All paths, hyperparameters, and operational constants are defined here.
Uses pydantic-settings for type safety and .env file support.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


# ── Base Paths ────────────────────────────────────────────────
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
    """Application-wide settings loaded from environment / .env file."""

    # ── Database ──────────────────────────────────────────────
    database_url: str = Field(
        default=f"sqlite:///{DATABASE_PATH}",
        description="SQLAlchemy database connection string.",
    )

    # ── Tile Extraction (Module 1) ────────────────────────────
    tile_size: int = Field(default=512, ge=64, le=2048)
    magnification_level: int = Field(
        default=0,
        ge=0,
        description="Pyramid level for tile extraction. 0 = highest (typically 40x).",
    )
    background_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Discard tile if background percentage exceeds this value.",
    )

    # ── Experience Replay Buffer (Module 3) ───────────────────
    replay_ratio: float = Field(
        default=4.0,
        gt=0.0,
        description="Ratio of historical:new images in training set.",
    )
    train_val_split: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Fraction of data allocated to training (rest is validation).",
    )
    max_replay_dataset_size: int = Field(
        default=5000,
        ge=10,
        description="Hard cap on total images in temp training set to prevent disk exhaustion.",
    )
    underrepresented_boost: float = Field(
        default=1.5,
        ge=1.0,
        description="Multiplier for per-class quota of underrepresented classes.",
    )

    # ── Training (Module 4) ───────────────────────────────────
    max_epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    base_model_name: str = Field(
        default="yolov8n.pt",
        description="Ultralytics pretrained model to use as initial checkpoint.",
    )
    active_model_path: Path = Field(
        default=MODELS_DIR / "active_model.pt",
        description="Path to the currently active model weights.",
    )

    # ── Warm-Start Training ──────────────────────────────────
    warm_start_epochs: int = Field(
        default=20, # Reduced from 30 to prevent mosaic closing degradation and overfitting
        ge=1,
        le=200,
        description="Number of epochs for initial warm-start training.",
    )
    warm_start_imgsz: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Image size for warm-start training.",
    )
    min_map50_improvement: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum mAP@50 improvement required for model promotion.",
    )
    # ── Inference (Module 5) ──────────────────────────────────
    inference_overlap_pct: float = Field(
        default=0.25,
        ge=0.0,
        lt=1.0,
        description="Overlap percentage (0-1) between sliding window tiles.",
    )
    inference_conf_threshold: float = Field(
        default=0.25, # Lowered from 0.50 to boost recall for critical medical findings
        ge=0.0,
        le=1.0,
        description="Minimum YOLO confidence threshold to record an annotation.",
    )
    nms_iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for global Non-Maximum Suppression to deduplicate cells.",
    )
    max_tiles_to_save_per_wsi: int = Field(
        default=5000,
        ge=1,
        description="Maximum number of tissue-containing tiles to save to disk per WSI to prevent disk exhaustion.",
    )
    save_empty_tiles: bool = Field(
        default=True,
        description="If True (Option A), saves all tissue tiles even if no blast is detected. Best for active learning.",
    )

    # ── Default Class Taxonomy ────────────────────────────────
    default_classes: List[str] = Field(
        default=[
            "Benign",       # 0: Benign / normal cells
            "Early",        # 1: Early Pre-B ALL blast (lymphoid)
            "Pre",          # 2: Pre B-ALL blast (lymphoid)
            "Pro",          # 3: Pro B-ALL blast (lymphoid)
            "WBC",          # 4: Normal White Blood Cells
            "RBC",          # 5: Red Blood Cells
            "Platelets",    # 6: Platelets
            "Myeloblast",   # 7: AML myeloid blast
            "Lymphoblast",  # 8: Generic lymphoid blast
            "Atypical",     # 9: Morphologically ambiguous / suspicious
        ],
        description="Seed classes for the CLASS_REGISTRY table.",
    )

    model_config = {
        "env_prefix": "BLAST_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ── Singleton instance ────────────────────────────────────────
settings = AppSettings()


# ── Security: Path Validation ─────────────────────────────────
def validate_path_within(file_path: Path, allowed_root: Path) -> Path:
    """
    Ensure that ``file_path`` resolves inside ``allowed_root``.

    Prevents directory-traversal attacks when paths sourced from the
    database are used in file-system operations (e.g., shutil.copy).

    Raises:
        ValueError: If the resolved path escapes the allowed root.

    Returns:
        The resolved, validated ``Path``.
    """
    resolved = file_path.resolve()
    allowed = allowed_root.resolve()

    # is_relative_to was introduced in Python 3.9 and strictly requires the path 
    # to be a true hierarchical child of the allowed root, defeating startswith spoofing.
    if hasattr(resolved, 'is_relative_to'):
        if not resolved.is_relative_to(allowed):
            raise ValueError(
                f"Path traversal blocked: '{file_path}' resolves to "
                f"'{resolved}', which is outside '{allowed}'."
            )
    else:
        # Fallback for Python < 3.9
        try:
            resolved.relative_to(allowed)
        except ValueError:
            raise ValueError(
                f"Path traversal blocked: '{file_path}' resolves to "
                f"'{resolved}', which is outside '{allowed}'."
            )
            
    return resolved


# ── Directory Bootstrap ───────────────────────────────────────
def ensure_directories() -> None:
    """Create all required data directories if they do not exist."""
    for directory in [
        RAW_TILES_DIR,
        ANNOTATED_IMAGES_DIR,
        ANNOTATED_LABELS_DIR,
        TEMP_TRAIN_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
