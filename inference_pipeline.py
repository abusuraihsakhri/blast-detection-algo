"""
Module 5: Sliding-Window Inference Pipeline

This module executes the trained YOLO model across enormous Whole Slide Images (WSIs).
Given the size of WSIs (often 100K x 100K pixels), direct inference is impossible.
Instead, we:
1. Extract sliding tiles (with overlap) using tiffslide.
2. Quickly discard empty glass using background thresholding.
3. Run YOLO inference on valid tissue patches.
4. Apply global Non-Maximum Suppression (NMS) to collapse duplicate detections
   caused by the overlapping window structure.
5. Save the resulting tiles and annotations to the active learning DB.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Generator, List, Tuple, Dict, Any

import numpy as np
import torch
import torchvision
from PIL import Image
from loguru import logger
from sqlalchemy.orm import Session
from ultralytics import YOLO

# Windows compatibility for WSI reading without C libraries
try:
    import tiffslide
except ImportError:
    logger.error("tiffslide not found. Run: pip install tiffslide")
    sys.exit(1)

from config import RAW_TILES_DIR, settings, validate_path_within
from database import SessionLocal, init_db
from models import Annotation, ClassRegistry, Tile


class WsiScanner:
    """Handles extracting overlapping tiles from a Whole Slide Image."""

    def __init__(self, wsi_path: str | Path):
        self.wsi_path = Path(wsi_path).resolve()
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI not found: {self.wsi_path}")

        self.slide = tiffslide.TiffSlide(self.wsi_path)
        
        # Determine scanning dimensions
        self.level = settings.magnification_level
        self.width, self.height = self.slide.level_dimensions[self.level]
        logger.info(f"Loaded WSI '{self.wsi_path.name}' — {self.width}x{self.height} at level {self.level}")

        self.tile_size = settings.tile_size
        self.stride = int(self.tile_size * (1.0 - settings.inference_overlap_pct))

    def _is_empty_tissue(self, img: Image.Image) -> bool:
        """
        Fast filter: Convert to grayscale and check ratio of bright (background) pixels.
        Avoids sending empty glass glass to the GPU.
        """
        # Convert to numpy array in grayscale
        gray = np.array(img.convert("L"))
        
        # Pixels > 220 are typically white/empty glare
        num_background_pixels = np.sum(gray > 220)
        total_pixels = self.tile_size * self.tile_size
        background_pct = float(num_background_pixels) / total_pixels
        
        # If the background exceeds threshold, consider it empty glass
        return background_pct > settings.background_threshold

    def extract_patches(self) -> Generator[Tuple[int, int, Image.Image, float], None, None]:
        """
        Yields (x_coord, y_coord, image, tissue_pct) for patches containing tissue.
        """
        total_tiles = sum(
            1 for _ in range(0, self.width - self.tile_size, self.stride)
            for _ in range(0, self.height - self.tile_size, self.stride)
        )
        logger.info(f"Scanning up to {total_tiles} sliding windows...")

        extracted_count = 0

        for y in range(0, self.height - self.tile_size, self.stride):
            for x in range(0, self.width - self.tile_size, self.stride):
                # Read specific tile region directly from disk
                # get_thumbnail is fast, but read_region is precise coordinate extraction
                img = self.slide.read_region(
                    (x, y), self.level, (self.tile_size, self.tile_size)
                ).convert("RGB")
                
                # Fast background check
                if self._is_empty_tissue(img):
                    continue
                
                # Approximate tissue pct
                tissue_pct = round(1.0 - (np.sum(np.array(img.convert("L")) > 220) / (self.tile_size**2)), 4)
                
                yield (x, y, img, tissue_pct)
                extracted_count += 1
                
                if extracted_count >= settings.max_tiles_to_save_per_wsi:
                    logger.warning(f"Reached tile limit ({settings.max_tiles_to_save_per_wsi}). Halting scan.")
                    return


class InferencePipeline:
    """Orchestrates WSI scanning, YOLO inference, NMS, and DB storage."""

    def __init__(self):
        if not settings.active_model_path.exists():
            raise FileNotFoundError(f"Active model missing at {settings.active_model_path}")
            
        logger.info("Initializing YOLO model & DB Connection...")
        self.model = YOLO(str(settings.active_model_path))
        self.db = SessionLocal()
        
        # Caches the YOLO model's class names (0: "Benign", 1: "Early", etc.)
        self.class_names = self.model.names

    def run(self, wsi_path: str | Path) -> None:
        """Execute the full end-to-end sliding window pipeline."""
        wsi = WsiScanner(wsi_path)
        wsi_name = Path(wsi_path).name

        all_global_detections = []
        valid_tiles_info = []

        logger.info("Starting inference loop...")
        start_time = time.time()

        # Phase 1: Scan and Inflate Local Bounding Boxes
        for x, y, img, tissue_pct in wsi.extract_patches():
            # Run YOLO prediction on the single tile
            results = self.model.predict(
                img,
                conf=settings.inference_conf_threshold,
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes
            
            # Save empty tiles ONLY if allowed by settings
            if len(boxes) == 0 and not settings.save_empty_tiles:
                continue

            # Generate unique filename
            img_filename = f"{wsi_name}_{x}_{y}.jpg"
            img_save_path = RAW_TILES_DIR / img_filename
            
            # Record tile memory footprint 
            # (We save immediately to avoid keeping 5000 images in RAM)
            img.save(img_save_path, "JPEG", quality=90)
            
            valid_tiles_info.append({
                "x": x,
                "y": y,
                "tissue": tissue_pct,
                "filename": img_filename
            })
            
            if len(boxes) == 0:
                continue

            # Translate YOLO local coordinates (0-1) to Global WSI Pixels
            # YOLO returns xywhn -> x_center_norm, y_center_norm, w_norm, h_norm
            for box in boxes:
                coords = box.xywhn[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())

                x_c_n, y_c_n, w_n, h_n = coords
                
                # Scale to local pixels
                lx_c = x_c_n * settings.tile_size
                ly_c = y_c_n * settings.tile_size
                lw = w_n * settings.tile_size
                lh = h_n * settings.tile_size
                
                # Transform to bounding box formats [x1, y1, x2, y2] globally
                gx1 = x + (lx_c - lw / 2)
                gy1 = y + (ly_c - lh / 2)
                gx2 = x + (lx_c + lw / 2)
                gy2 = y + (ly_c + lh / 2)
                
                all_global_detections.append({
                    "box": [gx1, gy1, gx2, gy2],
                    "score": conf,
                    "class_idx": cls_id,
                    "tile_filename": img_filename
                })

        logger.info(f"Phase 1 complete. Tissues saved: {len(valid_tiles_info)}. Total raw detections: {len(all_global_detections)}")

        if len(all_global_detections) > 0:
            # Phase 2: Global Non-Maximum Suppression (NMS)
            final_detections = self._apply_global_nms(all_global_detections)
        else:
            final_detections = []

        # Phase 3: Committing to Active Learning Database
        logger.info("Committing results to the structured database...")
        self._commit_to_db(wsi_name, valid_tiles_info, final_detections)
        
        elapsed = time.time() - start_time
        logger.success(f"Inference complete in {elapsed:.2f}s!")

    def _apply_global_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Resolve boundary overlaps using torchvision's NMS.
        Returns the subset of detections that survived suppression.
        """
        boxes_tensor = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
        scores_tensor = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
        
        # NMS natively supports running per-class or agnostic.
        # We run it agnostically across all classes here to prevent colliding predictions 
        # (e.g. if one tile says 'Blast' and the other says 'Lymphocyte' for the same cell)
        keep_indices = torchvision.ops.nms(
            boxes_tensor, 
            scores_tensor, 
            settings.nms_iou_threshold
        )
        
        survivors = [detections[i] for i in keep_indices]
        logger.info(f"Global NMS reduced redundant detections from {len(detections)} -> {len(survivors)}")
        return survivors

    def _commit_to_db(
        self, 
        wsi_name: str, 
        tiles_info: List[Dict], 
        survivor_detections: List[Dict]
    ) -> None:
        """
        Maps surviving global detections back into YOLO-normalized bounds
        against their designated parent Tile, and commits exactly to the database.
        """
        try:
            # Insert Tiles
            tile_records = {} # Dict[filename, TileORM]
            for info in tiles_info:
                tile = Tile(
                    file_path=info["filename"],
                    source_wsi=wsi_name,
                    x_coord=info["x"],
                    y_coord=info["y"],
                    level=settings.magnification_level,
                    tissue_pct=info["tissue"],
                    is_annotated=False  # Crucial: marks this as pending pathologist review
                )
                self.db.add(tile)
                tile_records[info["filename"]] = tile
            
            # Flush so tiles get primary keys assigned
            self.db.flush()

            # Insert Annotations
            for det in survivor_detections:
                filename = det["tile_filename"]
                if filename not in tile_records:
                    continue
                    
                tile = tile_records[filename]
                
                # Transform global [x1, y1, x2, y2] back to local [x_center_n, y_center_n, w_n, h_n]
                gx1, gy1, gx2, gy2 = det["box"]
                
                # Local coords inside the tile
                lx1 = gx1 - tile.x_coord
                ly1 = gy1 - tile.y_coord
                lx2 = gx2 - tile.x_coord
                ly2 = gy2 - tile.y_coord
                
                # Enclose bounding box strictly into the [0, 1] tile boundary 
                # (since cell boundary might slightly break the tile edge)
                lx1 = max(0.0, min(float(lx1) / settings.tile_size, 1.0))
                ly1 = max(0.0, min(float(ly1) / settings.tile_size, 1.0))
                lx2 = max(0.0, min(float(lx2) / settings.tile_size, 1.0))
                ly2 = max(0.0, min(float(ly2) / settings.tile_size, 1.0))
                
                w_n = lx2 - lx1
                h_n = ly2 - ly1
                xc_n = lx1 + (w_n / 2.0)
                yc_n = ly1 + (h_n / 2.0)

                # Skip invalid collapsed boxes
                if w_n <= 0.001 or h_n <= 0.001:
                    continue

                class_string = self.class_names.get(det["class_idx"], "unknown")

                ann = Annotation(
                    tile_id=tile.id,
                    class_label=class_string,
                    x_center=xc_n,
                    y_center=yc_n,
                    width=w_n,
                    height=h_n,
                    confidence=det["score"],
                    is_manual=False  # Flags this as machine-generated
                )
                self.db.add(ann)

            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to commit inference results to database: {e}")
            raise
        finally:
            self.db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Slide-Window WSI Inference")
    parser.add_argument("wsi_path", type=str, help="Absolute path to .svs or .tiff file")
    args = parser.parse_args()
    
    init_db()
    pipeline = InferencePipeline()
    pipeline.run(args.wsi_path)
