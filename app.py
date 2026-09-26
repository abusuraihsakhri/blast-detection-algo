"""Local FastAPI application for pathology review and sandbox inference."""

from __future__ import annotations

import io
import shutil
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Iterator, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from config import (
    ANNOTATED_IMAGES_DIR,
    ANNOTATED_LABELS_DIR,
    DATA_DIR,
    RAW_TILES_DIR,
    ensure_directories,
    settings,
    validate_path_within,
)
from database import SessionLocal, init_db
from models import Annotation, Tile

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    _cleanup_old_sandbox_uploads()
    yield


app = FastAPI(title="Blast Detection Review", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )
    return response


ensure_directories()
public_path = Path(__file__).parent / "public"
app.mount("/static", StaticFiles(directory=str(public_path)), name="static")
app.mount("/raw_tiles", StaticFiles(directory=str(RAW_TILES_DIR)), name="raw_tiles")

SANDBOX_UPLOADS_DIR = DATA_DIR / "sandbox_uploads"
SANDBOX_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class BoxModel(BaseModel):
    class_label: str = Field(..., min_length=1, max_length=64)
    x_center: float = Field(..., ge=0.0, le=1.0)
    y_center: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., gt=0.0, le=1.0)
    height: float = Field(..., gt=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    @field_validator("class_label")
    @classmethod
    def validate_class_label(cls, value: str) -> str:
        value = value.strip()
        if value not in settings.default_classes:
            raise ValueError("Unknown class label")
        return value


class SaveRequest(BaseModel):
    annotations: List[BoxModel] = Field(default_factory=list, max_length=5000)


class SandboxSaveRequest(BaseModel):
    sandbox_filename: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    annotations: List[BoxModel] = Field(default_factory=list, max_length=5000)


@contextmanager
def db_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _tile_disk_path(tile: Tile) -> Path:
    stored = Path(tile.file_path)
    candidate = stored if stored.is_absolute() else DATA_DIR / stored
    try:
        candidate = validate_path_within(candidate, DATA_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Stored tile path is invalid.") from exc

    if not candidate.exists() and len(stored.parts) == 1:
        legacy = validate_path_within(RAW_TILES_DIR / stored.name, RAW_TILES_DIR)
        if legacy.exists():
            candidate = legacy
    return candidate


def _write_annotation_export(
    image_src: Path,
    base_name: str,
    annotations: List[BoxModel],
) -> tuple[Path, Path]:
    image_src = validate_path_within(image_src, DATA_DIR)
    if not image_src.exists():
        raise HTTPException(status_code=409, detail="Source image is missing on disk.")

    dest_image = validate_path_within(
        ANNOTATED_IMAGES_DIR / f"{base_name}.jpg",
        ANNOTATED_IMAGES_DIR,
    )
    dest_label = validate_path_within(
        ANNOTATED_LABELS_DIR / f"{base_name}.txt",
        ANNOTATED_LABELS_DIR,
    )

    tmp_image = dest_image.with_suffix(".tmp.jpg")
    tmp_label = dest_label.with_suffix(".tmp.txt")
    try:
        shutil.copy2(image_src, tmp_image)
        lines = []
        for box in annotations:
            class_idx = settings.default_classes.index(box.class_label)
            lines.append(
                f"{class_idx} {box.x_center:.6f} {box.y_center:.6f} "
                f"{box.width:.6f} {box.height:.6f}"
            )
        tmp_label.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        tmp_image.replace(dest_image)
        tmp_label.replace(dest_label)
    except Exception:
        tmp_image.unlink(missing_ok=True)
        tmp_label.unlink(missing_ok=True)
        raise
    return dest_image, dest_label


def _cleanup_old_sandbox_uploads() -> None:
    cutoff = time.time() - settings.sandbox_retention_seconds
    for path in SANDBOX_UPLOADS_DIR.glob("sandbox_*.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = public_path / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI assets are missing.")
    return FileResponse(index_path)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": settings.active_model_path.exists()}


@app.get("/api/config")
def get_config():
    return {
        "taxonomy": settings.default_classes,
        "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
    }


@app.get("/api/tile/next")
def get_next_tile():
    with db_session() as db:
        tile = (
            db.query(Tile)
            .filter(Tile.is_annotated.is_(False))
            .order_by(Tile.id.asc())
            .first()
        )
        if not tile:
            return {"status": "empty", "message": "No tiles are waiting for review."}

        image_path = _tile_disk_path(tile)
        if not image_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Tile image for record {tile.id} is missing.",
            )

        try:
            relative = image_path.relative_to(RAW_TILES_DIR)
        except ValueError as exc:
            raise HTTPException(
                status_code=500,
                detail="Tile is outside the raw tile directory.",
            ) from exc

        annotations = db.query(Annotation).filter(Annotation.tile_id == tile.id).all()
        return {
            "status": "success",
            "tile": {"id": tile.id, "path": f"/raw_tiles/{relative.as_posix()}"},
            "annotations": [
                {
                    "id": a.id,
                    "class_label": a.class_label,
                    "x_center": a.x_center,
                    "y_center": a.y_center,
                    "width": a.width,
                    "height": a.height,
                    "score": a.confidence,
                }
                for a in annotations
            ],
        }


@app.post("/api/tile/{tile_id}/save")
def save_tile_annotations(tile_id: int, request: SaveRequest):
    with db_session() as db:
        tile = db.get(Tile, tile_id)
        if not tile:
            raise HTTPException(status_code=404, detail="Tile not found.")

        src_image = _tile_disk_path(tile)
        base_name = f"manual_{tile_id}_{src_image.stem}"
        try:
            dest_image, _ = _write_annotation_export(
                src_image,
                base_name,
                request.annotations,
            )
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Failed to write annotation export.",
            ) from exc

        db.query(Annotation).filter(Annotation.tile_id == tile.id).delete(
            synchronize_session=False
        )
        for box in request.annotations:
            db.add(
                Annotation(
                    tile_id=tile.id,
                    class_label=box.class_label,
                    x_center=box.x_center,
                    y_center=box.y_center,
                    width=box.width,
                    height=box.height,
                    confidence=1.0,
                    is_manual=True,
                )
            )
        tile.file_path = str(dest_image.relative_to(DATA_DIR))
        tile.is_annotated = True

    return {"status": "success", "message": "Review saved."}


_inf_model = None


async def _read_valid_image(file: UploadFile) -> Image.Image:
    content_type = (file.content_type or "").lower()
    if content_type not in settings.allowed_upload_mime_types:
        raise HTTPException(status_code=415, detail="Unsupported image type.")

    contents = await file.read(settings.max_upload_bytes + 1)
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the upload size limit.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    if image.width * image.height > settings.max_upload_pixels:
        raise HTTPException(
            status_code=413,
            detail="Image dimensions exceed the configured pixel limit.",
        )
    return image


@app.post("/api/test/upload")
async def test_inference(file: UploadFile = File(...)):
    global _inf_model

    # Validate uploaded file first to reject malformed, oversized, or unauthorized payloads
    image = await _read_valid_image(file)
    _cleanup_old_sandbox_uploads()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ultralytics is not installed.",
        ) from exc

    if _inf_model is None:
        if not settings.active_model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="No active model is available. Train or provide a model first.",
            )
        _inf_model = YOLO(str(settings.active_model_path))

    safe_id = uuid.uuid4().hex[:12]
    orig_stem = Path(file.filename or "upload").stem
    safe_stem = "".join(
        c if c.isalnum() or c in ("_", "-") else "_"
        for c in orig_stem
    )[:50] or "upload"
    persisted_filename = f"sandbox_{safe_id}_{safe_stem}.jpg"
    persisted_path = validate_path_within(
        SANDBOX_UPLOADS_DIR / persisted_filename,
        SANDBOX_UPLOADS_DIR,
    )
    image.save(persisted_path, "JPEG", quality=95)

    try:
        # Low floor on purpose: the UI confidence slider filters client-side.
        result = (
            await run_in_threadpool(
                _inf_model.predict, image, conf=0.05, verbose=False
            )
        )[0]
    except Exception as exc:
        persisted_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        ) from exc

    predictions = []
    for box in result.boxes:
        x_c_n, y_c_n, w_n, h_n = box.xywhn[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu())
        names = _inf_model.names
        class_name = (
            names.get(cls_id, "unknown")
            if isinstance(names, dict)
            else names[cls_id]
        )
        if class_name not in settings.default_classes:
            continue
        predictions.append(
            {
                "class_label": class_name,
                "x_center": float(x_c_n),
                "y_center": float(y_c_n),
                "width": float(w_n),
                "height": float(h_n),
                "confidence": float(box.conf[0].cpu()),
            }
        )

    return {
        "status": "success",
        "sandbox_filename": persisted_filename,
        "predictions": predictions,
    }


@app.post("/api/test/save")
def save_sandbox_corrections(request: SandboxSaveRequest):
    src_path = validate_path_within(
        SANDBOX_UPLOADS_DIR / request.sandbox_filename,
        SANDBOX_UPLOADS_DIR,
    )
    if not src_path.exists() or not src_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Sandbox image not found. Re-upload and try again.",
        )

    base_name = Path(request.sandbox_filename).stem
    try:
        dest_image, _ = _write_annotation_export(
            src_path,
            base_name,
            request.annotations,
        )
        with db_session() as db:
            tile = Tile(
                file_path=str(dest_image.relative_to(DATA_DIR)),
                source_wsi=f"sandbox:{request.sandbox_filename}",
                x_coord=0,
                y_coord=0,
                level=0,
                tissue_pct=1.0,
                is_annotated=True,
            )
            db.add(tile)
            db.flush()
            for box in request.annotations:
                db.add(
                    Annotation(
                        tile_id=tile.id,
                        class_label=box.class_label,
                        x_center=box.x_center,
                        y_center=box.y_center,
                        width=box.width,
                        height=box.height,
                        confidence=1.0,
                        is_manual=True,
                    )
                )
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save corrected annotations.",
        ) from exc

    src_path.unlink(missing_ok=True)
    return {
        "status": "success",
        "message": (
            f"Saved {len(request.annotations)} annotation(s) "
            "for the next training cycle."
        ),
    }


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    def open_browser() -> None:
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
