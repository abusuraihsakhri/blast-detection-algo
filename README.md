# Leukemic Blast Detection and Active Learning Pipeline

### [View the Project Documentation & Evaluation Plots →](https://abusuraihsakhri.github.io/blast-detection-algo/)

[![Domain](https://img.shields.io/badge/Domain-Digital%20Hematopathology%20%7C%20Leukemia-purple.svg)](https://github.com/abusuraihsakhri/blast-detection-algo)
[![PyTorch](https://img.shields.io/badge/Backend-PyTorch%20%7C%20CUDA%20AMP-blue.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Architecture-YOLOv8n%20(3.01M%20Params)-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Test mAP@50](https://img.shields.io/badge/Leak--free%20test%20mAP%4050-96.94%25-success.svg)](https://github.com/abusuraihsakhri/blast-detection-algo)
[![Datalake](https://img.shields.io/badge/Datalake-25%2C875%20Fields%20%7C%20115%2C548%20Cells-orange.svg)](https://github.com/abusuraihsakhri/blast-detection-algo)
[![Release](https://img.shields.io/github/v/release/abusuraihsakhri/blast-detection-algo?color=blueviolet&label=Model%20Release)](https://github.com/abusuraihsakhri/blast-detection-algo/releases/tag/v1.1.0)

A computer vision and active learning pipeline for detecting and classifying **Leukemic Blasts** in peripheral blood smears and Whole Slide Images (WSIs). Trained across **25,875 microscopy fields** (**115,548 annotated cells** across 6 curated cohorts). Features PyTorch YOLOv8 detection, sliding-window WSI tiling with boundary-aware Non-Maximum Suppression (NMS), a local review and sandbox web UI, and an Experience Replay Buffer that prevents catastrophic forgetting during incremental training.

---

## 📊 Evaluation Results (Multi-Center Dataset Benchmark)

v1.1.0 was retrained for 20 epochs after an audit of v1.0.0 found three data problems:

- **Mixed label formats.** Leukemia-NFXZN mixes box and polygon rows in 1,010 label files. Ultralytics reads such a file as all-polygon and garbles its 2,840 box rows.
- **Image-level tags stored as cell boxes.** 465 Myeloblast-fbliw "RBC" boxes and 119 Blood-Cell-znm2t polygons cover more than 80% of the image.
- **Leakage across datasets.** Several sources re-export the same micrographs, so evaluation images reappear in other datasets' training splits. Every Blast-Cell-Detection test image has a near-copy in another training split.

Both models are scored on the same **leak-free test set**: 757 test images (4,631 boxes) with no near-duplicate in any training split, and corrected labels. The checkpoint is epoch 19, the best validation epoch.

| Metric | v1.0.0 | v1.1.0 | Description |
| :--- | :---: | :---: | :--- |
| **mAP@50** | 93.96% | **96.94%** | Mean AP at IoU 0.50 over the 8 classes with test boxes |
| **mAP@50-95** | 72.57% | **77.10%** | Mean AP averaged over IoU thresholds 0.50–0.95 |
| **Precision ($P$)** | 86.82% | **94.07%** | Mean precision across classes |
| **Recall ($R$)** | 90.64% | **94.46%** | Mean recall across classes |
| **Validation mAP@50** | | 94.10% | 9 classes, including the Lymphoblast holdout |
| **Inference latency** | | ≈ 3 ms / image | 640 px, batch 16, RTX 3060 Laptop GPU, plus ≈ 1 ms NMS |
| **Weights** | | `data/models/active_model.pt` | 6.27 MB, also attached to Release v1.1.0 |

| Class | v1.0.0 AP@50 | v1.1.0 AP@50 | v1.1.0 AP@50-95 |
| :--- | :---: | :---: | :---: |
| Benign | 0.960 | **0.969** | 0.687 |
| Early | 0.988 | **0.985** | 0.702 |
| Pre | 0.987 | **0.990** | 0.795 |
| Pro | 0.994 | **0.994** | 0.870 |
| WBC | 0.966 | **0.974** | 0.730 |
| RBC | 0.909 | **0.961** | 0.849 |
| Platelets | 0.851 | **0.940** | 0.667 |
| Myeloblast | 0.862 | **0.941** | 0.869 |

Lymphoblast has no test split. On its validation holdout, v1.1.0 reaches AP@50 0.743 (AP@50-95 0.612), the weakest class. Atypical has no boxes at all, so the model has never seen an example of it.

**Whole-slide tiling.** `python evaluate_tiling.py` stitches 64 test images into a 4800×4800 pyramidal TIFF and runs the sliding-window pipeline on it. The pipeline reaches mAP@50 97.7%, against 98.0% when each image is predicted directly. This checks the tiling code, not scanner or staining shift on real slides.

**Limitations.** These are in-distribution numbers on the original per-dataset test splits, and Leukemia-NFXZN supplies 309 of the 757 leak-free test images. Platelets rest on 26 BCCD images. Lymphoblast's holdout also picked the checkpoint, so its number is slightly optimistic. The model has not been evaluated on real whole-slide scans, and tiles are not resampled to a common microns-per-pixel scale before inference.

| Training Loss & Convergence | Normalized Confusion Matrix |
| :---: | :---: |
| ![Convergence Curves](docs/assets/results.png) | ![Confusion Matrix](docs/assets/confusion_matrix_normalized.png) |

| Precision-Recall Curve | F1-Confidence Calibration Curve |
| :---: | :---: |
| ![PR Curve](docs/assets/BoxPR_curve.png) | ![F1 Curve](docs/assets/BoxF1_curve.png) |

*Trained weights are tracked at [`data/models/active_model.pt`](data/models/active_model.pt) and downloadable via [GitHub Release v1.1.0](https://github.com/abusuraihsakhri/blast-detection-algo/releases/tag/v1.1.0). v1.0.0 stays available on its release page.*

---

## 🔬 Dataset Overview (25,875 Fields / 115,548 Cells)

The training data merges 6 public research datasets into a unified 10-class taxonomy. The table shows the raw sources. Merging converts polygon rows to boxes, drops image-level tags, and removes 676 train images that duplicate validation or test images. Acute-Leukemia lends about 10% of its source images to validation. The v1.1.0 training set is 22,366 images (96,573 boxes), with 1,749 validation images.

| # | Dataset / Source | Scope / Target Classes | Train | Valid | Test | Total Images | Annotated Cells |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **1** | **Leukemia-NFXZN** (Tawfiq Islam) | B-ALL subtyping (Benign, Early, Pre, Pro) | 6,589 | 622 | 312 | 7,523 | 83,854 |
| **2** | **Blast-Cell-Detection** (YOLOv4) | Binary blast vs leukocyte (blasts remapped to Pre) | 3,011 | 72 | 39 | 3,122 | 3,365 |
| **3** | **BCCD Blood Cell Foundation** | Baseline elements (WBC, RBC, Platelets) | 377 | 110 | 53 | 540 | 8,851 |
| **4** | **Blood-Cell-znm2t Differential** | Leukocyte differential (8 subtypes, all merged into WBC) | 9,105 | 389 | 387 | 9,881 | 10,589 |
| **5** | **Acute-Leukemia AML/ALL** | Myeloblast vs Lymphoblast (train split only) | 3,809 | 0 | 0 | 3,809 | 7,626 |
| **6** | **Myeloblast-fbliw Dedicated** | Acute myeloid leukemia blasts | 700 | 200 | 100 | 1,000 | 1,263 |
| | **Grand Total** | | **23,591** | **1,393** | **891** | **25,875** | **115,548** |

### 🏷️ Unified 10-Class Taxonomy

1. **Benign (0):** Non-neoplastic mature lymphocytes and normal cells.
2. **Early (1):** Early Pre-B Acute Lymphoblastic Leukemia blasts.
3. **Pre (2):** Pre B-ALL blasts.
4. **Pro (3):** Pro B-ALL blasts.
5. **WBC (4):** General mature white blood cells (neutrophils, eosinophils, monocytes).
6. **RBC (5):** Mature erythrocytes.
7. **Platelets (6):** Thrombocytes.
8. **Myeloblast (7):** Acute Myeloid Leukemia (AML) blasts.
9. **Lymphoblast (8):** Generic acute lymphoblastic leukemia blasts. From Acute-Leukemia only; the weakest class (validation AP@50 0.743).
10. **Atypical (9):** Reserved for reviewer-assigned labels in the review UI. No source dataset maps to it, so it has no training examples.

---

## 🏛️ System Architecture

```text
[ Whole Slide Image (.svs / .ndpi / .tiff) or Microscopic Smear (.jpg / .png) ]
                                   │
                                   ▼
             [ Sliding-Window Tile Extraction Engine ]
             ├── Background Filtering (Luminance > 220, Threshold > 80%)
             └── 512×512 Sliding Window Tiling (25% Stride Overlap)
                                   │
                                   ▼
                [ Neural Detection Backbone ]
             ├── Architecture: YOLOv8n (3.01M Parameters)
             ├── 10-Class Taxonomy (ALL, AML, Normal Lineages)
             └── ≈ 3 ms / image on an RTX 3060 Laptop GPU
                                   │
                                   ▼
           [ Global Boundary Non-Maximum Suppression (NMS) ]
             ├── Drop Boxes Cut by Interior Tile Edges
             ├── Local Tile Coordinates -> Global WSI Coordinates Re-Projection
             └── Vectorized Agnostic NMS (IoU = 0.45, Conf >= 0.25)
                                   │
                                   ▼
            [ Review and Sandbox Web Interface ]
             ├── Interactive Bounding Box Refinement & Class Reassignment
             └── Sandbox Inference with Dynamic Thresholding
                                   │
                                   ▼
             [ Experience Replay Buffer (HITL Fine-Tuning) ]
             ├── Stratified Historical Sampling (Replay Ratio = 4.0)
             └── Underrepresented Class Quota Multiplier (1.5x Boost)
```

---

## 🔬 Pipeline Capabilities & Workflows

1. **Smear Tile Screening:** Scans digitized blood smear fields and identifies candidate leukemic blasts for expert inspection.
2. **Whole Slide Image (WSI) Tiling:** Slices gigapixel SVS and TIFF slides into 512×512 tiles with 25% overlap, discards empty glass tiles, maps tile coordinates into slide space, and applies global Non-Maximum Suppression (NMS).
3. **Pre-B ALL Subtyping:** Distinguishes Early, Pre, and Pro B-cell lymphoblastic leukemia morphological stages across annotated microscopic fields.
4. **Myeloid vs. Lymphoid Discrimination:** Detects and separates myeloblasts (AML) from lymphoblasts (ALL) and normal hematopoietic elements.
5. **Human-in-the-Loop Active Learning:** A local FastAPI web interface enables reviewing model bounding boxes, adjusting classes, and assembling training sets via an experience replay buffer to prevent catastrophic forgetting.

---

## 🚀 Quickstart & Inference

### 1. Environment Setup
```bash
git clone https://github.com/abusuraihsakhri/blast-detection-algo.git
cd blast-detection-algo
python -m venv .venv
```

On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux / macOS:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Pre-Loaded Model Weights
The active model weights are committed directly in the repository at `data/models/active_model.pt`. You can also download them from GitHub Releases:
```bash
# Optional: download the v1.1.0 release checkpoint
curl -L -o data/models/active_model.pt https://github.com/abusuraihsakhri/blast-detection-algo/releases/download/v1.1.0/active_model.pt
```

### 3. Launch Review & Sandbox UI
```bash
python app.py
```
Open your browser to `http://127.0.0.1:8000` to access the Active Learning Review Queue and the Sandbox Inference workspace.

### 4. Run Sliding-Window WSI Inference
```bash
python inference_pipeline.py /path/to/patient_slide.svs
```

### 5. Incremental Fine-Tuning (HITL Active Learning)
After reviewing and saving new tiles in the UI:
```bash
python training_pipeline.py --mode hitl
```

### 6. Automated Testing
```bash
pytest -q tests
python -m compileall -q -x "\.venv" .
node --check public/app.js
```

---

## 🔒 Security Hardening

- **Bounded File Uploads:** Constrained to `settings.max_upload_bytes` (default 25 MB) to prevent memory exhaustion.
- **MIME Type Whitelisting:** Strictly enforces image content-types (`image/jpeg`, `image/png`, `image/bmp`, `image/webp`).
- **Decompression Bomb Protection:** Configured with `settings.max_upload_pixels` (default 50 Megapixels) to defeat decompression bomb exploits.
- **Path Traversal Defenses:** All filesystem inputs and destinations are verified with `validate_path_within()` before reading or writing.
- **Origin Restriction & Security Headers:** Enforces `nosniff`, `DENY` frame-options, Content Security Policy, and binds CORS to local origins.
- **Private Sandbox Storage:** Temporary sandbox uploads are stored in an internal directory and not exposed via public static routes.

---

## Dataset attribution

The training data come from public Roboflow Universe datasets:

| Dataset | Source | License |
| :--- | :--- | :--- |
| Leukemia-NFXZN | [tawfiq-islam-hfp3h/leukemia-nfxzn](https://universe.roboflow.com/tawfiq-islam-hfp3h/leukemia-nfxzn/dataset/1) | CC BY 4.0 |
| Blast-Cell-Detection | [yolov4-njwoe/blast-cell-detection](https://universe.roboflow.com/yolov4-njwoe/blast-cell-detection) (versions 26–28) | Not recorded in the download; check the source page |
| BCCD | [bccd-vhdu3/blood-cell-cella-bccd-lihfn](https://universe.roboflow.com/bccd-vhdu3/blood-cell-cella-bccd-lihfn/dataset/1) | CC BY 4.0 |
| Blood-Cell-znm2t | [aninp/blood-cell-znm2t](https://universe.roboflow.com/aninp/blood-cell-znm2t/dataset/1) | CC BY 4.0 |
| Acute-Leukemia | [yolov4-njwoe/acute-leukemia](https://universe.roboflow.com/yolov4-njwoe/acute-leukemia/dataset/1) | CC BY 4.0 |
| Myeloblast-fbliw | [nhung-o028n/myeloblast-fbliw](https://universe.roboflow.com/nhung-o028n/myeloblast-fbliw/dataset/1) | CC BY 4.0 |

The detector is trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), which is why this repository uses the same license.

## ⚖️ License & Disclaimer

Distributed under the **GNU Affero General Public License v3.0** (AGPL-3.0). The pipeline is built on Ultralytics YOLO, which is AGPL-3.0, and the released checkpoint carries the same license. Using this code or the weights in a network service means offering that service's source under AGPL-3.0; commercial use without that obligation needs an Ultralytics enterprise license.

> **Research Use Only**: This software is intended for research, method evaluation, and algorithmic experimentation. It is not a certified diagnostic device and is not intended for primary clinical diagnosis without independent validation and regulatory approval.
