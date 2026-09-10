# CNN-based Feature Extraction from Drone Orthophotos

Automated semantic segmentation of high-resolution drone orthophotos to extract
**building footprints**, **roads**, and **vegetation** for residential area mapping.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [1. Tiling](#1-tiling)
   - [2. Training](#2-training)
   - [3. Evaluation](#3-evaluation)
   - [4. Inference](#4-inference)
   - [5. API](#5-api-optional)
6. [Configuration Reference](#configuration-reference)
7. [Output Format](#output-format)
8. [Performance Targets](#performance-targets)
9. [QGIS Integration](#qgis-integration)
10. [Extending the Project](#extending-the-project)

---

## Overview

```
Drone imagery → GeoTIFF orthophoto
          │
          ▼
   Tiling (256×256, 15% overlap)
          │
          ▼
   U-Net / DeepLabV3+ (ResNet34, ImageNet pretrained)
          │
          ▼
   Softmax probabilities per class
          │
          ▼
   Threshold → Morphological cleanup → Rasterio vectorize
          │
          ▼
   GeoJSON per class (building, road, vegetation)
```

### Key design choices

| Concern              | Solution                                             |
|----------------------|------------------------------------------------------|
| Encoder              | ResNet34 (ImageNet pretrained, good speed/accuracy)  |
| Segmentation head    | U-Net (or DeepLabV3+ for sharper boundaries)         |
| Loss                 | Dice + CrossEntropy (handles class imbalance)        |
| Optimiser            | AdamW + cosine annealing LR schedule                 |
| Mixed precision      | `torch.cuda.amp` (fp16) for ~2× GPU speedup          |
| Tile overlap merging | Probability averaging across overlapping tile regions|
| Vectorization        | `rasterio.features.shapes` + Douglas-Peucker         |

---

## Project Structure

```
.
├── config/
│   └── config.yaml          ← all hyperparameters and paths
│
├── preprocessing/
│   ├── tiling.py            ← GeoTIFF → overlapping tile patches
│   └── dataset.py           ← PyTorch Dataset + DataLoader factory
│
├── models/
│   ├── segmentation_model.py← smp model factory + checkpoint helpers
│   └── losses.py            ← Dice + CrossEntropy combined loss
│
├── training/
│   └── train.py             ← full training loop
│
├── postprocessing/
│   └── vectorize.py         ← mask → GeoJSON pipeline
│
├── evaluation/
│   └── evaluate.py          ← per-class IoU / Dice / F1 report
│
├── inference/
│   └── predict.py           ← end-to-end CLI inference
│
├── api/
│   └── app.py               ← FastAPI REST service
│
├── utils/
│   ├── config.py            ← YAML loader with dot-access
│   ├── logger.py            ← rotating + coloured logger
│   └── metrics.py           ← SegmentationMetrics accumulator
│
├── data/                    ← (gitignored) raw + tiled data
│   ├── raw/
│   ├── tiles/
│   └── masks/
│
├── checkpoints/             ← (gitignored) model checkpoints
├── outputs/                 ← GeoJSON results + evaluation reports
├── logs/                    ← training logs + TensorBoard
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 – 3.11
- GDAL system libraries (required by `rasterio`)

**Ubuntu / Debian:**
```bash
sudo apt-get install -y gdal-bin libgdal-dev
```

**macOS (Homebrew):**
```bash
brew install gdal
```

**Windows:**  
Install GDAL via [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or use a
pre-built wheel from [Christoph Gohlke's repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

### Python environment

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install PyTorch (CPU example — replace with CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

> **GPU users:** Visit https://pytorch.org/get-started/locally/ to get the
> correct `torch` + CUDA wheel command, then install `requirements.txt`.

---

## Quick Start

### 1. Prepare data

Place your GeoTIFF orthophoto in `data/raw/` and your single-band integer
class-label mask (same CRS, same resolution) alongside it.

### 2. Tile the orthophoto

```bash
python preprocessing/tiling.py \
    data/raw/my_ortho.tif \
    --mask data/raw/my_ortho_mask.tif \
    --out-img data/tiles \
    --out-mask data/masks
```

### 3. Train

```bash
python training/train.py
```

Training logs are printed to console and saved to `logs/`. TensorBoard logs
are written to `logs/tensorboard/` — start TensorBoard with:
```bash
tensorboard --logdir logs/tensorboard
```

### 4. Evaluate

```bash
python evaluation/evaluate.py --checkpoint checkpoints/best.pt --split test
```

### 5. Predict on a new orthophoto

```bash
python inference/predict.py \
    data/raw/new_ortho.tif \
    --checkpoint checkpoints/best.pt \
    --output-dir outputs/new_ortho \
    --save-prob-tif
```

---

## Pipeline Walkthrough

### 1. Tiling

`preprocessing/tiling.py` reads a large GeoTIFF and produces overlapping
256×256 tile pairs (image + mask).  Overlap is 15 % by default (controlled
via `tiling.overlap` in `config.yaml`).  Tiles that are more than 70 %
nodata are discarded.

The affine transform of each tile is preserved in the output GeoTIFF so that
predictions can be stitched back with correct georeferencing.

### 2. Training

`training/train.py` runs the full training loop:

- **Loss**: `0.5 × Dice + 0.5 × CrossEntropy`
- **Optimiser**: AdamW, lr=1e-4, weight_decay=1e-5
- **Scheduler**: Cosine annealing over `epochs` iterations
- **Mixed precision**: `torch.cuda.amp` (enabled by default)
- **Checkpointing**: top-3 checkpoints by val IoU + `best.pt` symlink
- **Early stopping**: patience = 10 epochs (configurable)
- **TensorBoard**: optional (auto-detected)

### 3. Evaluation

`evaluation/evaluate.py` loads a checkpoint, runs inference on the test
split, and reports per-class **IoU**, **Dice**, and **F1**.

Sample output:
```
────────────────────────────────────────────────────────────
Class                    IoU      Dice        F1
────────────────────────────────────────────────────────────
background            0.9521    0.9756    0.9756
building              0.7812    0.8772    0.8772
road                  0.8134    0.8971    0.8971
vegetation            0.8423    0.9145    0.9145
────────────────────────────────────────────────────────────
Mean                  0.8473    0.9161    0.9161
────────────────────────────────────────────────────────────
```

### 4. Inference

`inference/predict.py`:

1. Tiles the input orthophoto (same settings as training)
2. Runs batched CNN inference on all tiles
3. Merges tile probability maps using **probability averaging** (configurable
   to max-vote)
4. Passes the merged probability volume through the vectorization pipeline
5. Writes one GeoJSON per class to the output directory

### 5. API (optional)

Start the FastAPI server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

#### Endpoints

| Method | Path         | Description                               |
|--------|--------------|-------------------------------------------|
| GET    | `/`          | Health check                              |
| GET    | `/classes`   | List configured classes                   |
| GET    | `/model/info`| Model architecture from config            |
| POST   | `/predict`   | Upload GeoTIFF → ZIP of GeoJSON files     |

#### Example `curl` call

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@my_ortho.tif" \
     -o features.zip
unzip features.zip
```

Interactive API docs are available at `http://localhost:8000/docs`.

---

## Configuration Reference

All settings live in `config/config.yaml`.  The most important knobs:

| Key | Default | Description |
|-----|---------|-------------|
| `tiling.tile_size` | 256 | Tile width/height in pixels |
| `tiling.overlap` | 0.15 | Fractional overlap between tiles |
| `model.architecture` | unet | `unet` or `deeplabv3plus` |
| `model.encoder` | resnet34 | Any smp-supported encoder |
| `training.epochs` | 50 | Max training epochs |
| `training.batch_size` | 16 | Adjust to GPU memory |
| `training.learning_rate` | 1e-4 | AdamW initial LR |
| `training.mixed_precision` | true | fp16 training |
| `postprocessing.binary_threshold` | 0.5 | Mask binarisation threshold |
| `postprocessing.min_polygon_area` | 25 | Min polygon area in sq metres |

---

## Output Format

Each class GeoJSON follows this schema:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [[...]] },
      "properties": {
        "feature_id": "uuid4-string",
        "class_id": 1,
        "class": "building",
        "area_sq_m": 142.5,
        "centroid_lon": 77.594123,
        "centroid_lat": 12.971598
      }
    }
  ]
}
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Building IoU | ≥ 0.75 |
| Road F1 | ≥ 0.80 |
| Inference time | < 5 min per sq km tile set (GPU) |

---

## QGIS Integration

1. In QGIS, go to **Layer → Add Layer → Add Vector Layer**
2. Select the output `building.geojson` / `road.geojson` / `vegetation.geojson`
3. Style each layer by class colour
4. Use the **Field Calculator** to compute additional statistics

If you ran inference with `--save-prob-tif`, you can also load
`outputs/probabilities.tif` as a raster and style it per band for QA.

---

## Extending the Project

### Add a new class

1. Extend `dataset.classes` in `config.yaml` (add class name at end)
2. Re-label your mask GeoTIFFs to include the new class index
3. Increase `dataset.num_classes` and `model.num_classes`
4. Re-train from scratch (or fine-tune from a checkpoint with `--resume`)

### Switch to DeepLabV3+

Change `model.architecture` to `deeplabv3plus` in `config.yaml` — no code
changes needed.

### Use EfficientNet-B0 encoder

Set `model.encoder` to `efficientnet-b0`.  Make sure to install the
`timm` package: `pip install timm`.

### Add Mask R-CNN (instance segmentation)

Create `models/mask_rcnn.py` following the same `SegmentationModel` interface
and register it in `models/segmentation_model.py`'s `_ARCH_MAP`.
