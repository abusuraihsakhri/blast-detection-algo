import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# ensure paths
root_dir = Path(__file__).resolve().parent
mock_path = root_dir / "mock_wsi.tiff"

def create_mock_wsi(path: Path):
    # 2048x2048 covers exactly 4x4 grid of 512x512 tiles with overlaps
    size = 2048
    # Background: mostly white/glass with a slight off-white color
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    # Add "tissue" lines and regions so it's not discarded
    # We need < 80% to be >220 grayscale.
    # The whole image being 245, 245, 245 means it's current 100% glass.
    # Let's paint a huge dark block of tissue
    draw.rectangle([200, 200, 1800, 1800], fill=(200, 150, 150))
    
    # Let's draw some "blasts" -> dark purple dots
    # Put one exactly on a tile boundary
    # If overlap is 0.25 (128px), strides are 384.
    # Tile 0: 0-512. Tile 1: 384-896. Boundary is between 384 and 512.
    # Let's put a cell center at 448 (middle of the overlap).
    cell_locs = [
        (448, 448),    # On horizontal and vertical overlaps
        (800, 800),    # Inside a tile
        (1216, 448),   # Overlap
        (1500, 1500),  # Inside
    ]
    
    for cx, cy in cell_locs:
        r = 15
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(50, 0, 100))

    img.save(path, format="TIFF")
    print(f"Mock WSI created at {path}")

if __name__ == "__main__":
    create_mock_wsi(mock_path)
    
    from inference_pipeline import InferencePipeline
    from database import SessionLocal, init_db
    from models import Tile, Annotation
    from config import ensure_directories
    
    ensure_directories()
    init_db()
    
    # Clean previous mock run to make the script idempotent
    db = SessionLocal()
    db.query(Tile).filter(Tile.source_wsi == "mock_wsi.tiff").delete()
    db.commit()
    db.close()
    
    pipeline = InferencePipeline()
    print("Running inference pipeline...")
    pipeline.run(mock_path)
    
    # Verification checks
    db = SessionLocal()
    tiles = db.query(Tile).filter(Tile.source_wsi == "mock_wsi.tiff").all()
    annotations = db.query(Annotation).join(Tile).filter(Tile.source_wsi == "mock_wsi.tiff").all()
    
    print(f"\nVerification Results:")
    print(f"  Tiles generated: {len(tiles)}")
    print(f"  Annotations captured: {len(annotations)}")
    
    for ann in annotations:
        print(f"    - Class: {ann.class_label}, Confidence: {ann.confidence:.2f}, "
              f"Tile Center: ({ann.x_center:.3f}, {ann.y_center:.3f})")
              
    db.close()
