import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import io
from PIL import Image
from sqlalchemy.orm import Session

from database import SessionLocal, init_db
from models import Tile, Annotation
from config import (
    settings, 
    RAW_TILES_DIR, 
    ANNOTATED_IMAGES_DIR, 
    ANNOTATED_LABELS_DIR,
    ensure_directories
)

app = FastAPI(title="Pathology Active Learning UI")

# Apply security headers (CORS and explicit X-Content-Type-Options)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response

# Ensure required directories exist
ensure_directories()

# Mount the public directory for static files (React/Vanilla JS UI)
public_path = Path(__file__).parent / "public"
app.mount("/static", StaticFiles(directory=str(public_path)), name="static")

# Mount the raw_tiles directory so the UI can fetch images securely
# We mount it dynamically so the frontend can load `<img src="/raw_tiles/..." />`
app.mount("/raw_tiles", StaticFiles(directory=str(RAW_TILES_DIR)), name="raw_tiles")

class BoxModel(BaseModel):
    class_label: str = Field(..., max_length=64)
    x_center: float = Field(..., ge=0.0, le=1.0)
    y_center: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

class SaveRequest(BaseModel):
    annotations: List[BoxModel]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the main application."""
    index_path = public_path / "index.html"
    if not index_path.exists():
        return "<h1>Error: UI not built yet</h1>"
    return FileResponse(index_path)

@app.get("/api/config")
def get_config():
    """Provides the UI with required settings like taxonomy."""
    return {"taxonomy": settings.default_classes}

@app.get("/api/tile/next")
def get_next_tile():
    """Fetches the next unannotated tile from the DB alongside its machine predictions."""
    db: Session = SessionLocal()
    
    # Grab first tile that needs pathologist review
    tile = db.query(Tile).filter(Tile.is_annotated == False).first()
    
    if not tile:
        db.close()
        return {"status": "empty", "message": "No more tiles in queue!"}
        
    annotations = db.query(Annotation).filter(Annotation.tile_id == tile.id).all()
    
    response = {
        "status": "success",
        "tile": {
            "id": tile.id,
            "path": f"/raw_tiles/{Path(tile.file_path).name}"
        },
        "annotations": [
            {
                "id": a.id,
                "class_label": a.class_label,
                "x_center": a.x_center,
                "y_center": a.y_center,
                "width": a.width,
                "height": a.height,
                "score": a.confidence
            }
            for a in annotations
        ]
    }
    
    db.close()
    return response

@app.post("/api/tile/{tile_id}/save")
def save_tile_annotations(tile_id: int, request: SaveRequest):
    """
    Saves the user modifications. Emits standard YOLO exports into the
    Historical Data / Replay Buffer directories.
    """
    db: Session = SessionLocal()
    tile = db.query(Tile).filter(Tile.id == tile_id).first()
    if not tile:
        db.close()
        raise HTTPException(status_code=404, detail="Tile not found")
        
    # 1. Purge old machine annotations and insert these manual ones
    db.query(Annotation).filter(Annotation.tile_id == tile.id).delete()
    
    valid_classes = settings.default_classes
    
    yolo_lines = []
    
    for box in request.annotations:
        ann = Annotation(
            tile_id=tile.id,
            class_label=box.class_label,
            x_center=box.x_center,
            y_center=box.y_center,
            width=box.width,
            height=box.height,
            confidence=1.0,
            is_manual=True
        )
        db.add(ann)
        
        # Prepare YOLO line if the class is valid
        if box.class_label in valid_classes:
            class_idx = valid_classes.index(box.class_label)
            # YOLO format: cls_idx x_center y_center width height
            yolo_lines.append(f"{class_idx} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}")

    # Mark tile as annotated
    tile.is_annotated = True
    db.commit()
    db.close()

    # 2. Extract into YOLO AL format for the Replay Buffer Trainer
    # This directly triggers the dataset ingestion pipeline on next run
    src_image = RAW_TILES_DIR / Path(tile.file_path).name
    
    # Unique base name for AL cache
    base_name = f"manual_{tile_id}_{Path(tile.file_path).stem}"
    dest_image = ANNOTATED_IMAGES_DIR / f"{base_name}.jpg"
    dest_label = ANNOTATED_LABELS_DIR / f"{base_name}.txt"
    
    if src_image.exists():
        shutil.copy2(src_image, dest_image)
    
    with open(dest_label, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))
        
    return {"status": "success", "message": "Saved successfully!"}

# --- Testing / Sandbox Inference Endpoints ---

# Lazy-loaded model to prevent blocking app startup
_inf_model = None

# Directory to temporarily hold sandbox uploads for correction workflows
SANDBOX_UPLOADS_DIR = Path(__file__).parent / "data" / "sandbox_uploads"
SANDBOX_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/sandbox_uploads", StaticFiles(directory=str(SANDBOX_UPLOADS_DIR)), name="sandbox_uploads")

@app.post("/api/test/upload")
async def test_inference(file: UploadFile = File(...)):
    """Runs live YOLO inference on an uploaded image and persists it for correction."""
    global _inf_model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise HTTPException(status_code=500, detail="Ultralytics is not installed.")

    # Only load model once
    if _inf_model is None:
        if not settings.active_model_path.exists():
            raise HTTPException(status_code=500, detail=f"Model not found at {settings.active_model_path}")
        _inf_model = YOLO(str(settings.active_model_path))
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    # Persist the uploaded image so it can be referenced during correction/save
    safe_id = uuid.uuid4().hex[:12]
    orig_stem = Path(file.filename or "upload").stem
    # Sanitize filename — strip anything non-alphanumeric
    safe_stem = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in orig_stem)[:50]
    persisted_filename = f"sandbox_{safe_id}_{safe_stem}.jpg"
    image.save(SANDBOX_UPLOADS_DIR / persisted_filename, "JPEG", quality=95)
        
    # Run inference — use low conf so frontend slider has full control
    results = _inf_model.predict(image, conf=0.05, verbose=False)
    result = results[0]
    
    predictions = []
    if len(result.boxes) > 0:
        boxes = result.boxes
        for box in boxes:
            # Normalized center coordinates
            x_c_n, y_c_n, w_n, h_n = box.xywhn[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            class_name = _inf_model.names.get(cls_id, "unknown")
            
            predictions.append({
                "class_label": class_name,
                "x_center": float(x_c_n),
                "y_center": float(y_c_n),
                "width": float(w_n),
                "height": float(h_n),
                "confidence": conf
            })
    
    return {
        "status": "success",
        "sandbox_filename": persisted_filename,
        "predictions": predictions
    }


class SandboxSaveRequest(BaseModel):
    sandbox_filename: str = Field(..., max_length=128)
    annotations: List[BoxModel]

@app.post("/api/test/save")
def save_sandbox_corrections(request: SandboxSaveRequest):
    """
    Saves user-corrected annotations from Sandbox mode into the 
    Active Learning replay buffer (ANNOTATED_IMAGES_DIR / ANNOTATED_LABELS_DIR).
    This means the next training cycle will learn from these corrections.
    """
    src_path = SANDBOX_UPLOADS_DIR / request.sandbox_filename
    # Security: ensure no path traversal MUST happen BEFORE checking existence
    # to prevent blind path traversal / information disclosure oracles.
    try:
        from config import validate_path_within
        validate_path_within(src_path, SANDBOX_UPLOADS_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # Validate that the referenced sandbox image actually exists
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Sandbox image not found. Re-upload and try again.")
    
    valid_classes = settings.default_classes
    yolo_lines = []
    
    for box in request.annotations:
        if box.class_label in valid_classes:
            class_idx = valid_classes.index(box.class_label)
            yolo_lines.append(
                f"{class_idx} {box.x_center:.6f} {box.y_center:.6f} "
                f"{box.width:.6f} {box.height:.6f}"
            )
    
    # Copy image into the annotated training data directory
    base_name = Path(request.sandbox_filename).stem
    dest_image = ANNOTATED_IMAGES_DIR / f"{base_name}.jpg"
    dest_label = ANNOTATED_LABELS_DIR / f"{base_name}.txt"
    
    shutil.copy2(src_path, dest_image)
    
    with open(dest_label, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))
    
    return {
        "status": "success",
        "message": f"Saved {len(yolo_lines)} annotations to replay buffer.",
        "image_path": str(dest_image.name),
        "label_path": str(dest_label.name)
    }


if __name__ == "__main__":
    import webbrowser, threading

    init_db()

    def open_browser():
        """Open browser after a short delay to let uvicorn start."""
        import time
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:8000")

    print("\n" + "="*50)
    print("  PATHOLOGY UI SERVER STARTING...")
    print("  Browser will open automatically.")
    print("  If not, go to: http://127.0.0.1:8000")
    print("="*50 + "\n")

    # Launch browser opener in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start the server (this blocks)
    uvicorn.run(app, host="127.0.0.1", port=8000)
