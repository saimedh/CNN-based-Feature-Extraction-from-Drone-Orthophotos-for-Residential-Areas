# CNN Based Feature Extraction from Drone Orthophotos for Residential Areas

---

## 1. PRD (Product Requirements Document)

### Problem
Manual digitizing of buildings, roads, roofs, land parcels from drone ortho images — slow, error-prone, costly. Need automated system.

### Goal
Build system that ingests drone orthophoto, outputs vectorized features (building footprints, roads, vegetation, parcels) for residential area mapping.

### Users
- Urban planners
- Municipal/property tax bodies
- Surveying firms
- Disaster assessment teams

### Core Features
1. Upload/ingest orthophoto (GeoTIFF)
2. Auto-tile large image
3. Run CNN segmentation → building/road/vegetation masks
4. Convert masks to vector (GeoJSON/Shapefile)
5. Visualize overlay on map (web viewer or QGIS export)
6. Export report — feature count, area stats, coverage %

### Success Metrics
- IoU ≥ 0.75 on building class
- F1 ≥ 0.80 on road class
- Processing time < 5 min per sq km tile set
- Vector output usable directly in QGIS/GIS pipeline

### Out of Scope (v1)
- Real-time drone streaming
- 3D reconstruction
- Multi-spectral/LiDAR fusion

### Milestones
1. Data pipeline + labeling set ready
2. Baseline U-Net trained, IoU baseline established
3. Post-processing + vectorization working
4. Web viewer / QGIS export integration
5. Evaluation report, deployment package

---

## 2. TRD (Technical Requirements Document)

### Architecture
Drone imagery → Orthophoto stitch → Preprocessing (tiling/aug) → CNN segmentation model → Postprocessing (polygonize) → Vector output + evaluation.

### Data
- Input: GeoTIFF orthophoto, georeferenced (EPSG:4326 or local UTM)
- Tile size: 256x256 or 512x512, overlap 10-20% to avoid edge artifacts
- Labels: per-pixel masks, classes = building, road, vegetation, background (extendable)

### Model
- Primary: U-Net (ResNet34/EfficientNet-B0 encoder, ImageNet pretrained)
- Alt: DeepLabV3+ for better boundary handling
- Alt: Mask R-CNN if instance-level separation of individual buildings needed
- Framework: PyTorch + segmentation-models-pytorch

### Training
- Loss: Dice + BCE combined (handles class imbalance)
- Optimizer: AdamW, lr 1e-4, cosine decay
- Augmentation: flip, rotate, color jitter, random crop (albumentations)
- Batch size: per GPU memory, mixed precision (fp16) recommended
- Validation split: 15-20%, stratify by tile density/class balance

### Postprocessing
- Threshold mask → binary
- Morphological open/close to clean noise
- Raster → vector: `rasterio.features.shapes` or GDAL polygonize
- Simplify polygons (Douglas-Peucker) for cleaner GIS output
- Merge tile predictions back to full ortho footprint (handle overlap via max-vote or averaging)

### Output Format
- GeoJSON / Shapefile per class
- Attributes: feature_id, class, area_sq_m, centroid_lat, centroid_lon

### Evaluation
- Pixel-wise IoU, Dice, F1 per class
- Visual QA: overlay predictions vs ground truth in QGIS

### Infra
- Training: GPU instance (T4/A100 minimum viable)
- Inference: can run CPU for small orthophotos, GPU for large batch
- Storage: object storage for orthophotos + model checkpoints
- Optional API: FastAPI wrapper, endpoint accepts GeoTIFF → returns GeoJSON

### Tech Stack
Python, PyTorch, segmentation-models-pytorch, albumentations, rasterio, GDAL, QGIS, FastAPI (optional), Leaflet/Mapbox (optional viewer)

---

## 3. Prompt to Build (paste into Claude/Cursor/Copilot etc.)

```
Build a Python project for CNN-based semantic segmentation of drone orthophotos to extract residential area features (buildings, roads, vegetation).

Requirements:
- Use PyTorch + segmentation-models-pytorch, U-Net architecture with ResNet34 encoder (ImageNet pretrained)
- Input: GeoTIFF orthophoto. Tile it into 256x256 patches with rasterio, 15% overlap
- Dataset class: load image tiles + corresponding mask tiles, apply albumentations (flip, rotate, color jitter, random crop)
- Loss function: combined Dice + BCE loss
- Training script: AdamW optimizer, lr 1e-4, cosine annealing scheduler, mixed precision, checkpoint best model by validation IoU
- Postprocessing: threshold predicted mask, morphological cleanup (opening/closing), convert raster mask to vector polygons using rasterio.features.shapes, simplify geometry, merge tile results into single GeoJSON with correct georeferencing
- Evaluation script: compute per-class IoU, Dice, F1 on validation/test set
- Project structure: separate modules for preprocessing, models, training, postprocessing, utils (config-driven via YAML)
- Include requirements.txt and a README with setup + usage instructions

Deliver clean, modular, well-commented code ready to run on a sample orthophoto dataset.
```
