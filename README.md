# Blast Detection Algorithm

Local research software for reviewing blood-cell detections, correcting bounding boxes, and preparing reviewed images for incremental YOLO training. The repository also contains a sliding-window whole-slide-image (WSI) inference pipeline and warm-start dataset utilities.

> **Research use only.** This project is not a validated diagnostic device and should not be used for clinical decision-making without appropriate validation, governance, and regulatory review.

## What is included

- **Review queue:** inspect model-generated tile annotations, move/draw/delete boxes, change classes, and save reviewed labels.
- **Sandbox inference:** upload a local image, inspect detections at an adjustable confidence threshold, correct them, and save the reviewed result for a later training cycle.
- **WSI inference:** scan SVS/TIFF pyramid images in overlapping tiles, filter background, run YOLO inference, and apply global NMS.
- **Incremental training:** combine newly reviewed tiles with a stratified sample of previously reviewed tiles.
- **Warm-start utilities:** download configured source datasets, remap labels into the project taxonomy, and train an initial detector.
- **Audit trail:** training runs are recorded in SQLite with status, configuration, model path, and validation mAP@50.

The repository does **not** contain trained model weights or benchmark artifacts that substantiate a specific accuracy figure or training-image count. Performance should be measured on an independent, versioned validation/test set appropriate to the intended use.

## Requirements

- Python 3.10–3.14; Python 3.11 is used in CI.
- A supported PyTorch environment. GPU acceleration is optional but recommended for training/inference workloads.
- `tiffslide` and its runtime requirements for WSI scanning.
- An active YOLO model at `data/models/active_model.pt` for sandbox or WSI inference.
- A Roboflow API key only if `download_datasets.py` is used to fetch source datasets.

## Setup

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional configuration can be copied from `.env.example` to `.env`. Keep API keys and credentials out of Git.

## Run the review application

```bash
python app.py
```

The server binds to `127.0.0.1:8000` by default and opens the local interface in a browser. Uploaded sandbox images are stored temporarily on the same machine; successful sandbox saves move a copy into the reviewed dataset and register it in SQLite for a later training cycle.

The browser UI is a client for the local FastAPI backend. It is **not suitable for GitHub Pages**: functional inference and review require Python, SQLite, filesystem access, model weights, PyTorch/Ultralytics, and native WSI support. Converting this workflow to Pyodide/PyScript would not preserve the current functionality or practical model-loading characteristics.

## Browser compatibility

The interface uses standard Canvas and Pointer Events APIs and is intended for current Chromium, Firefox, and Safari releases on desktop and mobile. CI syntax-checks the JavaScript and smoke-tests the FastAPI routes; it does not currently run a cross-browser end-to-end test matrix.

## Main workflows

### WSI inference

```bash
python inference_pipeline.py /path/to/slide.svs
```

Tiles are generated with overlap and stored under `data/raw_tiles/`. Tile coordinates are recorded in level-0 slide coordinates so detections from pyramid levels can be reconciled consistently.

### Warm-start training

Configure/download the source datasets first if required:

```bash
python download_datasets.py
python training_pipeline.py --mode warm
```

Dataset-to-taxonomy remapping rules are defined in `training_pipeline.py`. These mappings are domain assumptions and should be checked against the exact dataset versions and labels before scientific use.

### Incremental training

```bash
python training_pipeline.py --mode hitl
```

The replay buffer uses reviewed `Tile`/`Annotation` records from the application database. The separate `AnnotationRecord` table created by the dataset-ingestion utilities is retained as source-dataset provenance; it is not automatically mixed into incremental replay.

Before promotion, the active model and candidate are compared on the same generated validation split. This is a regression guard, not a substitute for an independent external holdout set.

## Testing and security checks

```bash
pytest -q tests
python -m compileall -q .
node --check public/app.js
pip-audit -r requirements.txt
```

GitHub Actions runs these core checks on pull requests and on `main`. The application constrains upload type/size/pixel count, validates filesystem paths, limits CORS to the local UI origin, applies browser security headers, and does not expose the temporary sandbox-upload directory as a static route.

## Data and privacy

Runtime data, databases, model weights, WSI files, generated training data, logs, and `.env` files are ignored by Git. The application is designed as a local workflow; do not expose it directly to untrusted networks without authentication, TLS, deployment hardening, and an explicit data-governance review.

## Repository layout

```text
app.py                  FastAPI server and review/sandbox API
public/                 Browser UI
inference_pipeline.py   WSI tiling, model inference, global NMS
replay_buffer.py        Reviewed-tile replay dataset assembly
training_pipeline.py    Warm-start and incremental training
models.py               SQLAlchemy data model
config.py               Paths and environment-backed settings
tests/                  Automated tests
```

## License

No license file is currently included in this repository. Unless the repository owner adds one, no open-source license grant should be assumed.
