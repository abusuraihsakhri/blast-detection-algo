# Leukemic Blast Detection Clinical & Research Pipeline

### [Open the Live Research Portal & Evaluation Dashboard →](https://abusuraihsakhri.github.io/blast-detection-algo/)

[![Domain](https://img.shields.io/badge/Domain-Digital%20Hematopathology%20%7C%20Leukemia-purple.svg)](https://github.com/abusuraihsakhri/blast-detection-algo)
[![PyTorch](https://img.shields.io/badge/Backend-PyTorch%20%7C%20CUDA%20AMP-blue.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Architecture-YOLOv8n%20(3.01M%20Params)-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Validation](https://img.shields.io/badge/mAP%4050-92.23%25-success.svg)](https://github.com/abusuraihsakhri/blast-detection-algo)
[![Datalake](https://img.shields.io/badge/Datalake-25%2C875%20Fields%20%7C%20115%2C548%20Cells-orange.svg)](https://github.com/abusuraihsakhri/blast-detection-algo)
[![Release](https://img.shields.io/github/v/release/abusuraihsakhri/blast-detection-algo?color=blueviolet&label=Model%20Release)](https://github.com/abusuraihsakhri/blast-detection-algo/releases/tag/v1.0.0)

A clinical-grade, GPU-accelerated computer vision and active learning engine for detecting and classifying **Leukemic Blasts** in peripheral blood smears and Whole Slide Images (WSIs). Trained across **25,875 multi-center hematology microscopy fields** (**115,548 annotated cells** across 6 global cohorts). Features PyTorch YOLOv8 detection, tiled sliding-window WSI inference with boundary-aware Non-Maximum Suppression (NMS), a Human-in-the-Loop (HITL) review tool, and an Experience Replay Buffer that prevents catastrophic forgetting during incremental fine-tuning.

---

## 📊 Empirical Validation Results (Unified Multi-Center Benchmark)

The model was trained for **20 epochs** across the unified multi-center datalake and evaluated on unseen validation splits across diverse staining protocols and optical systems:

| Metric | Checkpoint Value | Clinical / Technical Description |
| :--- | :---: | :--- |
| **mAP@50** | **92.23%** (`0.9223`) | High sensitivity across heterogeneous leukemic blast subtypes |
| **Precision ($P$)** | **88.40%** (`0.8840`) | Minimizes false positive alarms on normal white and red blood cells |
| **Recall ($R$)** | **89.93%** (`0.8993`) | High capture rate on morphologically diverse immature blasts |
| **mAP@50-95** | **71.86%** (`0.7186`) | High bounding box regression accuracy on delicate cellular boundaries |
| **Inference Latency** | **1.2 – 2.8 ms / tile** | >350–500 FPS throughput enabling real-time sliding window scanning |
| **Neural Parameters** | **3.01 Million** | Compact YOLOv8n backbone deployable on edge microscopes |
| **Active Checkpoint** | `data/models/active_model.pt` | Pre-loaded 6.25 MB weight file tracked in Git and GitHub Releases |

### 📈 Convergence & Evaluation Curves

| Training Loss & Convergence | Normalized Confusion Matrix |
| :---: | :---: |
| ![Convergence Curves](docs/assets/results.png) | ![Confusion Matrix](docs/assets/confusion_matrix_normalized.png) |

| Precision-Recall Curve | F1-Confidence Calibration Curve |
| :---: | :---: |
| ![PR Curve](docs/assets/BoxPR_curve.png) | ![F1 Curve](docs/assets/BoxF1_curve.png) |

*Trained weights are committed directly at [`data/models/active_model.pt`](data/models/active_model.pt) and downloadable via [GitHub Release v1.0.0](https://github.com/abusuraihsakhri/blast-detection-algo/releases/tag/v1.0.0).*

---

## 🔬 Multi-Center Hematology Datalake (25,875 Fields / 115,548 Cells)

The model is trained on a diverse digital pathology datalake aggregating **25,875 microscopic fields** and **115,548 expert-annotated cells** across 6 curated global cohorts:

| # | Clinical Cohort / Source | Scope & Diagnostic Focus | Train | Valid | Test | Total Images | Annotated Cells |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **1** | **Leukemia-NFXZN** (Tawfiq Islam) | B-ALL subtyping (Benign, Early, Pre, Pro) | 6,589 | 622 | 312 | 7,523 | 83,854 |
| **2** | **Blast-Cell-Detection** (YOLOv4) | Binary blast vs mature leukocyte differentiation | 3,011 | 72 | 39 | 3,122 | 3,365 |
| **3** | **BCCD Blood Cell Foundation** | Baseline blood cell elements (WBC, RBC, Platelets) | 377 | 110 | 53 | 540 | 8,851 |
| **4** | **Blood-Cell-znm2t Differential** | Comprehensive leukocyte differential (12 subtypes) | 9,105 | 389 | 387 | 9,881 | 10,589 |
| **5** | **Acute-Leukemia AML/ALL** | Myeloblast vs Lymphoblast discrimination | 3,809 | 0 | 0 | 3,809 | 7,626 |
| **6** | **Myeloblast-fbliw Dedicated** | Dedicated acute myeloid leukemia (AML) blasts | 700 | 200 | 100 | 1,000 | 1,263 |
| | **Grand Total** | | **23,591** | **1,393** | **891** | **25,875** | **115,548** |

### 🏷️ Unified 10-Class Taxonomy

1. **Benign (0):** Normal mature lymphocytes and baseline non-neoplastic cells.
2. **Early (1):** Early Pre-B Acute Lymphoblastic Leukemia blasts.
3. **Pre (2):** Pre B-ALL blasts (primary lymphoid precursor population).
4. **Pro (3):** Pro B-ALL blasts (early uncommitted lymphoid blast stage).
5. **WBC (4):** General mature white blood cells (neutrophils, eosinophils, monocytes).
6. **RBC (5):** Mature erythrocytes for smear background normalization.
7. **Platelets (6):** Thrombocytes.
8. **Myeloblast (7):** Acute Myeloid Leukemia (AML) blasts (visible nucleoli, Auer rods).
9. **Lymphoblast (8):** Generic acute lymphoblastic leukemia blasts.
10. **Atypical (9):** Morphologically ambiguous / dysplastic cells flagged for review.

---

## 🏛️ End-to-End System Architecture

```text
[ Whole Slide Image (.svs / .ndpi / .tiff) or Microscopic Smear (.jpg / .png) ]
                                   │
                                   ▼
             [ Sliding-Window Tile Extraction Engine ]
             ├── Background Glass Filtering (Luminance > 220, Threshold > 80%)
             └── 512×512 Sliding Window Tiling (25% Stride Overlap)
                                   │
                                   ▼
                [ Neural Detection Backbone ]
             ├── Architecture: YOLOv8n (3.01M Parameters)
             ├── 10-Class Hematology Taxonomy (ALL, AML, Normal Lineages)
             └── CUDA AMP Inference Engine (1.2–2.8 ms / tile)
                                   │
                                   ▼
           [ Global Boundary Non-Maximum Suppression (NMS) ]
             ├── Local Tile Coordinates -> Global WSI Coordinates Re-Projection
             └── Vectorized Agnostic NMS (IoU = 0.45, Conf >= 0.25)
                                   │
                                   ▼
            [ Pathologist Review & Active Learning UI ]
             ├── Interactive Bounding Box Refinement & Class Reassignment
             └── Sandbox Instant Inference with Dynamic Thresholding
                                   │
                                   ▼
             [ Experience Replay Buffer (HITL Fine-Tuning) ]
             ├── Stratified Historical Sampling (Replay Ratio = 4.0)
             └── Underrepresented Class Quota Multiplier (1.5x Boost)
```

---

## 🩺 Clinical Use Cases

1. **High-Throughput Smear Triage:** Rapidly pre-screens peripheral blood smears to flag urgent blast proliferation (>20% diagnostic threshold) for immediate STAT bone marrow biopsy and flow cytometry.
2. **Whole Slide Imaging (WSI) Digital Hematopathology:** Scans gigapixel SVS and TIFF slides at 40× magnification, re-projecting local detections into unified slide coordinates to generate whole-slide blast density heatmaps.
3. **Pre-B ALL Staging Assistance:** Distinguishes Early, Pre, and Pro B-cell lymphoblastic leukemia stages to assist subspecialty pediatric hematology staging.
4. **Lineage Disambiguation (AML vs ALL):** Rapidly discriminates myeloid blasts (myeloblasts) from lymphoid blasts (lymphoblasts), expediting emergency induction chemotherapy protocol selection.
5. **Pathologist-in-the-Loop Active Learning:** Integrated web UI allows pathologists to correct false alarms, fine-tune bounding contours, and incrementally re-train the model without catastrophic forgetting.

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
The active model weights are committed directly in the repository at `data/models/active_model.pt`. You can also verify or re-download the release asset:
```bash
# Optional: re-download official v1.0.0 release checkpoint
curl -L -o data/models/active_model.pt https://github.com/abusuraihsakhri/blast-detection-algo/releases/download/v1.0.0/active_model.pt
```

### 3. Launch Pathologist Review & Sandbox UI
```bash
python app.py
```
Open your browser to `http://127.0.0.1:8000` to access the Active Learning Review Queue and the Drag-and-Drop Sandbox Inference workspace.

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
python -m compileall -q .
node --check public/app.js
```

---

## 🔒 Security Hardening

- **Bounded File Uploads:** Constrained to `settings.max_upload_bytes` to prevent memory exhaustion and Denial-of-Service (DoS) attacks.
- **MIME Type Whitelisting:** Strictly enforces image content-types (`image/jpeg`, `image/png`, `image/bmp`, `image/tiff`, `image/webp`).
- **Decompression Bomb Protection:** Configured with `settings.max_upload_pixels` to defeat decompression bomb exploits.
- **Path Traversal Defenses:** All filesystem inputs and destinations are verified with `validate_path_within()` before reading or writing.
- **Origin Restriction & Security Headers:** Enforces `nosniff`, `DENY` frame-options, strict CSP, and binds CORS strictly to local loopback origins.
- **Private Sandbox Storage:** Temporary sandbox uploads are not exposed via static routes to protect patient specimen privacy.

---

## ⚖️ License & Clinical Disclaimer

Distributed under the **Apache License 2.0**.

> **Research and Investigational Use Only**: This software is designed for academic research, algorithm development, and investigational hematopathology workflows. It is not an FDA/CE-IVD approved diagnostic medical device. Final clinical diagnoses must always be established by board-certified pathologists utilizing accredited laboratory standards and comprehensive clinical correlation.
