# Blast Algo - Pathology Active Learning Pipeline

An advanced, Human-In-The-Loop (HITL) Active Learning pipeline for detecting and classifying Leukemic blasts in Whole Slide Images (WSIs). Pre-trained on a robust baseline of ~25,000 blast images, the system combines Ultralytics YOLO, sliding-window inference, and an Experience Replay Buffer to iteratively assist pathologists while preventing catastrophic forgetting during continuous learning.

## 🔬 Overview

This repository orchestrates a robust end-to-end deep learning workflow using YOLO and an Experience Replay Buffer. The system allows pathologists to iteratively refine the model. As pathologists correct bounding boxes and classifications via the UI, the system leverages Active Learning to fine-tune the model, preventing catastrophic forgetting by smartly mixing new annotations with a highly curated historical dataset.

### Core Modules

1. **WSI Inference Pipeline (`inference_pipeline.py`)**: Slices massive (100K x 100K) WSIs into overlapping tiles, filters out empty glass using background thresholding, runs YOLO predictions, and uses global Non-Maximum Suppression (NMS) to collapse duplicates.
2. **Pathologist HITL UI (`app.py`)**: A FastAPI-powered web interface where pathologists review model predictions, correct bounding boxes, and save annotations.
3. **Experience Replay Buffer (`replay_buffer.py`)**: Stratifies and samples from historical annotations (~25k blasts) to dynamically construct balanced training datasets. It intelligently boosts underrepresented classes to prevent data imbalance.
4. **Training Engine (`training_pipeline.py`)**: Supports "Warm-Start" (training on baseline historical datasets) and "Incremental Fine-Tuning" (learning from pathologist corrections on the fly).
5. **Dataset Ingestion (`download_datasets.py` / `import_historical_data.py`)**: Automatically pulls and registers high-quality blast cell datasets from Roboflow Universe to bootstrap the model.

## 📊 Training Quality (Warm Start)

The initial Warm-Start model was trained on a robust amalgamation of **~25,000 blast and blood cell images**. The training spanned 20 epochs using the Ultralytics YOLO architecture. 

The model achieved excellent baseline detection capabilities, peaking at Epoch 10:
* **mAP@50**: 0.922 (92.2%)
* **mAP@50-95**: 0.718 (71.8%)
* **Precision**: 0.884 (88.4%)
* **Recall**: 0.899 (89.9%)

By the end of the 20 epochs, the model successfully generalized its bounding box regression and classification loss without severe overfitting, ensuring a solid foundation before the pathologist begins human-in-the-loop corrections.

## 🛠 Setup & Installation

### Requirements
* Python 3.10+
* Windows / Linux
* `tiffslide`, `torch`, `ultralytics`, `fastapi`, `sqlalchemy` (See `requirements.txt`)

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
ROBOFLOW_API_KEY=your_api_key_here
```

### 3. Bootstrap the Database & Datasets
Download the initial warm-start dataset (approx 25k images) and ingest it into the active learning database:
```bash
python download_datasets.py
python import_historical_data.py
```

### 4. Run Warm-Start Training
Train the initial baseline model. 
```bash
python training_pipeline.py --mode warm
```
*(This produces your initial `active_model.pt` saved to the `data/models/` directory)*

### 5. Launch the Pathologist UI
Start the FastAPI application to begin correcting WSI tiles and triggering incremental learning loops.
```bash
python app.py
```


## 🤝 Contribution
Contributions are welcome! Please ensure that any new APIs or endpoints follow the strict file validation checks laid out in `config.py` to prevent access control violations.
