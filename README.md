# CNN-based Feature Extraction from Drone Orthophotos

Automated semantic segmentation of high-resolution drone orthophotos to extract
**building footprints**, **roads**, and **vegetation** for residential area mapping.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Collection & Benchmark Datasets](#data-collection--benchmark-datasets)
3. [Data Preprocessing & Augmentation Pipeline](#data-preprocessing--augmentation-pipeline)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [1. Tiling](#1-tiling)
   - [2. Training](#2-training)
   - [3. Evaluation](#3-evaluation)
   - [4. Inference](#4-inference)
   - [5. API](#5-api-optional)
8. [Accuracy Benchmarks: Existing vs. Proposed Models](#accuracy-benchmarks-existing-vs-proposed-models)
9. [Configuration Reference](#configuration-reference)
10. [Output Format](#output-format)
11. [Performance Targets](#performance-targets)
12. [QGIS Integration](#qgis-integration)
13. [Extending the Project](#extending-the-project)

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

## Data Collection & Benchmark Datasets

This system supports raw drone flights, photogrammetric orthomosaics, and standard remote sensing benchmarks.

### 1. Data Collection Specifications (UAV Flight Parameters)

For custom drone surveys over residential areas, the recommended data acquisition protocol is:

| Parameter | Specification | Purpose / Notes |
| :--- | :--- | :--- |
| **Platform** | Quadcopter / Hexacopter (e.g., DJI Matrice 300 / Phantom 4 RTK) | Stable hovering and automated waypoint flight |
| **Camera Sensor** | 20–45 MP RGB Sensor (Global Shutter preferred) | Prevents rolling shutter distortion at flight speed |
| **Flight Altitude** | 80 m – 120 m AGL (Above Ground Level) | Balances spatial coverage and footprint resolution |
| **Ground Sampling Distance (GSD)** | 2.5 cm – 5.0 cm / pixel | Resolves precise rooftop edges, road curbs, and tree boundaries |
| **Forward / Side Overlap** | 75% – 85% Front, 70% – 80% Side | Ensures dense point cloud generation in photogrammetry (ODM / Metashape) |
| **GCPs / RTK-PPK** | 5+ Ground Control Points or RTK GNSS | Sub-centimetre georeferencing accuracy (EPSG:4326 / UTM zones) |
| **Radiometric Conditions** | Clear sky / diffuse overcast, solar noon ($\pm 2$ hrs) | Minimises deep shadow casting across narrow residential alleys |

### 2. Supported Public Benchmark Datasets

| Dataset | Sensor / Platform | Classes | Resolution (GSD) | Spatial Coverage | Access Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UAVid** | 4K UAV Oblique & Nadir | 8 (Building, Road, Tree, Low Veg, Static/Moving Car, Clutter) | ~2.5 cm/px ($3840 \times 2160$) | Urban & Residential Corridors | [uavid.nl](https://uavid.nl/) |
| **OpenDroneMap (Aukerman)** | Drone Orthomosaic (RGB) | 4 (Building, Road, Vegetation, Background) | ~3.0 cm/px ($4000 \times 3000$) | Rural & Residential Sites | [OpenDroneMap GitHub](https://github.com/OpenDroneMap/DroneDataset) |
| **ISPRS Potsdam** | Airborne True Orthophoto (TOP) | 6 (Imp. Surfaces, Building, Low Veg, Tree, Car, Clutter) | 5.0 cm/px ($6000 \times 6000$) | Dense Urban & Suburban | [ISPRS Benchmark](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) |
| **ISPRS Vaihingen** | Airborne Orthomosaic (IR-R-G) | 6 (Imp. Surfaces, Building, Low Veg, Tree, Car, Clutter) | 9.0 cm/px | Historic & Suburban Town | [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) |
| **LandCover.ai** | Aerial Orthophotos (Poland) | 4 (Building, Woodland, Water, Road) | 25 cm – 50 cm/px | $216.27\text{ km}^2$ Multi-region | [landcover.ai](https://landcover.ai.linuxp1.com/) |
| **Massachusetts Buildings & Roads** | Aerial Orthophotos (RGB) | Building Footprints / Road Network | 100 cm/px ($1500 \times 1500$) | Urban & Residential Boston | [Toronto GeoData](https://www.cs.toronto.edu/~vmnih/data/) |

---

## Data Preprocessing & Augmentation Pipeline

```
Raw Drone GeoTIFF + Label Mask
              │
              ▼  Step 1: Georeferenced Sliding Tiling (256×256 or 512×512, 15% overlap)
   Preserve Affine Transform & Spatial Metadata per Tile
              │
              ▼  Step 2: NoData & Valid Content Filtering (Discard if NoData > 70%)
   Clean Tile Database
              │
              ▼  Step 3: Radiometric & Channel Normalisation (ImageNet µ & σ)
   Normalized Tensor [C, H, W]
              │
              ▼  Step 4: Spatial & Color Augmentations (Albumentations)
   - Random Horizontal & Vertical Flips (p=0.5)
   - Random Affine Rotation (±45°, p=0.5)
   - Color Jitter: Brightness (0.2), Contrast (0.2), Saturation (0.2), Hue (0.05)
   - Random Crop (224×224) & Gaussian Blur
              │
              ▼  Step 5: Stratified Dataset Split (75% Train, 15% Val, 10% Test)
   Feed into Model Training Loop
```

### Preprocessing Steps Detail

1. **Sliding Window Tiling**:
   - Drone orthophotos are typically gigapixel rasters ($10,000 \times 10,000+$ pixels).
   - `preprocessing/tiling.py` slices the orthophoto and raster label mask into $256 \times 256$ patches with **15% stride overlap** to eliminate edge boundary distortion.
   - Each tile retains its geospatial bounding box, affine matrix, and EPSG projection.

2. **NoData Handling & Filtering**:
   - Tiles outside the flight survey polygon or containing $>70\%$ NoData/black border pixels are automatically purged using `min_valid_ratio: 0.3`.

3. **Data Augmentation Strategy**:
   - Applied via **Albumentations** on the fly during training:
     - Geometric: `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45)`.
     - Photometric: `ColorJitter`, `RandomBrightnessContrast`, `RGBShift`.
     - Regularization: `RandomCrop(224, 224)` for scale invariance.

4. **Multi-class Mask Encoding**:
   - Ground truth masks are mapped to discrete integer class IDs (`0`: Background, `1`: Building, `2`: Road, `3`: Vegetation) and converted to PyTorch `torch.LongTensor` targets.

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

## Accuracy Benchmarks

Live accuracy is **not hardcoded**. The dashboard reads:

1. `outputs/<run>/metrics.json` after inference (IoU/Dice if a paired `*_mask` was uploaded; otherwise softmax confidence)
2. `outputs/evaluation_report.json` from `python evaluation/evaluate.py` on the labelled test split
3. `outputs/multi_model_accuracy_report.json` only after a real tiled-dataset train (synthetic fallback scores were removed)

Latest checkpoint eval in this repo (`checkpoints/best.pt`, test split):

| Class | IoU | Dice / F1 |
| :--- | :---: | :---: |
| Background | 66.6% | 80.0% |
| Building | 54.8% | 70.8% |
| Road | 35.1% | 51.9% |
| Vegetation | 80.5% | 89.2% |
| **Mean** | **59.3%** | **73.0%** |

Upload a GeoTIFF + mask, or re-run evaluation, to refresh these numbers.

---

## Performance Targets & SLA

| Metric | Target | How it is measured |
| :--- | :---: | :--- |
| **Building Footprint IoU** | $\ge 0.75$ | Live IoU vs ground-truth mask / test split |
| **Road Network F1-Score** | $\ge 0.80$ | Live Dice/F1 vs ground-truth mask / test split |
| **Vegetation IoU** | $\ge 0.70$ | Live IoU vs ground-truth mask / test split |
| **Inference Throughput** | $< 5\text{ min / km}^2$ | Timed tiling + CNN + vectorize |
| **Vector Geometry Validity** | 100% valid OGC polygons | Vectorize + topological repair |

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

---

## Cloud Deployment (Render & Docker)

### Option 1: One-Click Render Blueprint (Recommended)

This repository includes a [`render.yaml`](render.yaml) blueprint specification that preconfigures the Python environment, CPU PyTorch dependencies, dynamic port binding (`$PORT`), and Uvicorn server:

1. Push your repository to GitHub: `git push origin main`
2. Open your [Render Dashboard](https://dashboard.render.com/)
3. Click **New +** → **Blueprint**
4. Connect this GitHub repository (`saimedh/CNN-based-Feature-Extraction-from-Drone-Orthophotos-for-Residential-Areas`)
5. Click **Apply**. Render will automatically:
   - Read `render.yaml`
   - Install CPU-optimized PyTorch and all dependencies
   - Launch the FastAPI API and responsive Leaflet web UI at your `https://<service-name>.onrender.com` URL

### Option 2: Render Manual Web Service

1. On [Render](https://dashboard.render.com/), click **New +** → **Web Service**
2. Choose **Build and deploy from a Git repository** and select your repository
3. Set the following settings:
   - **Environment / Runtime**: `Python`
   - **Build Command**:
     ```bash
     pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn api.app:app --host 0.0.0.0 --port $PORT
     ```
   - **Environment Variables**:
     - `PYTHON_VERSION`: `3.11.9`
     - `HOST`: `0.0.0.0`

### Option 3: Docker Deployment

The repository includes a containerized [`Dockerfile`](Dockerfile) based on `python:3.11-slim`:

```bash
# Build the Docker image
docker build -t orthoseg-ai .

# Run the container locally
docker run -p 8000:8000 orthoseg-ai
```
Navigate to `http://localhost:8000` to interact with the full web app.

