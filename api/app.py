"""
api/app.py
-----------
FastAPI REST API that accepts a GeoTIFF upload and returns GeoJSON feature
collections per class (building, road, vegetation).

Endpoints
---------
GET  /               – health check
POST /predict        – upload GeoTIFF, returns ZIP of GeoJSON files
GET  /classes        – list configured segmentation classes
GET  /model/info     – model architecture info

Usage
-----
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Or:
    python api/app.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from inference.predict import predict, predict_folder
from models import SegmentationModel
from analysis.dataset_analysis import analyze_dataset, generate_sample_grid
from analysis.prediction_analysis import analyze_predictions
from analysis.multi_model_trainer import train_and_benchmark_multi_models, MODEL_SPECS
from preprocessing.tiling import tile_orthophoto
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CNN Orthophoto Feature Extractor",
    description=(
        "Segment drone orthophotos into building, road, and vegetation "
        "features using a trained U-Net / DeepLabV3+ model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

cfg = load_config()
_model: Optional[SegmentationModel] = None
_checkpoint_path: Optional[str] = None

DEFAULT_CHECKPOINT = str(Path(cfg.paths.checkpoints_dir) / "best.pt")
DEMO_RUN_PREFIXES = ("demo",)


def _is_demo_run(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    return lowered.startswith(DEMO_RUN_PREFIXES) or lowered in {
        "demo_results",
        "demo_run",
        "demo_residential_ortho",
    }


def _resolve_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found at '{path}'. "
            "Train with `python training/train.py` or upload a .pt file to checkpoints/."
        )
    return path


def _load_model(checkpoint: str) -> SegmentationModel:
    """Load and cache the model (re-uses the cached instance if same path)."""
    global _model, _checkpoint_path
    resolved = str(_resolve_checkpoint(checkpoint))
    if _model is None or resolved != _checkpoint_path:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = SegmentationModel(cfg).to(device)
        _model.load_checkpoint(resolved, device=device)
        logger.info(f"Model loaded from {resolved}")
        _checkpoint_path = resolved
    return _model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check."""
    return {"status": "ok", "service": "CNN Orthophoto Feature Extractor"}


@app.get("/", tags=["General"])
async def root(request: Request):
    """Serve home landing page for browser requests, or JSON health check."""
    accept = request.headers.get("accept", "")
    home_file = FRONTEND_DIR / "home.html"
    index_file = FRONTEND_DIR / "index.html"
    if "text/html" in accept:
        if home_file.exists():
            return FileResponse(str(home_file))
        if index_file.exists():
            return FileResponse(str(index_file))
    return {"status": "ok", "service": "CNN Orthophoto Feature Extractor"}


@app.get("/home", include_in_schema=False)
@app.get("/home.html", include_in_schema=False)
async def serve_home():
    home_file = FRONTEND_DIR / "home.html"
    if home_file.exists():
        return FileResponse(str(home_file))
    raise HTTPException(status_code=404, detail="home.html not found")


@app.get("/login", include_in_schema=False)
@app.get("/login.html", include_in_schema=False)
async def serve_login():
    login_file = FRONTEND_DIR / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    raise HTTPException(status_code=404, detail="login.html not found")


@app.get("/demo", include_in_schema=False)
@app.get("/demo.html", include_in_schema=False)
async def serve_demo():
    """Demo mode removed — redirect users directly to the live dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=302)


@app.get("/dashboard", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_dashboard():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/style.css", include_in_schema=False)
async def get_css():
    css_file = FRONTEND_DIR / "style.css"
    if css_file.exists():
        return FileResponse(str(css_file), media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/app.js", include_in_schema=False)
async def get_js():
    js_file = FRONTEND_DIR / "app.js"
    if js_file.exists():
        return FileResponse(str(js_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


@app.get("/classes", tags=["Info"])
async def list_classes():
    """Return the list of segmentation classes configured in YAML."""
    return {"classes": list(cfg.dataset.classes)}


@app.get("/model/info", tags=["Info"])
async def model_info():
    """Return model architecture information from config."""
    return {
        "architecture": cfg.model.architecture,
        "encoder": cfg.model.encoder,
        "num_classes": cfg.model.num_classes,
        "in_channels": cfg.model.in_channels,
    }


@app.get("/runs", tags=["Info"])
async def list_runs():
    """List available processed feature extraction runs."""
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / cfg.paths.output_dir
    runs = []
    if out_dir.is_dir():
        for p in sorted(out_dir.iterdir(), key=lambda x: x.name):
            if p.is_dir():
                summary_file = p / "summary.json"
                summary_data = {}
                if summary_file.exists():
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            summary_data = json.load(f)
                    except Exception:
                        pass

                layers = {}
                for cls in ("building", "road", "vegetation"):
                    gj_file = p / f"{cls}.geojson"
                    if gj_file.exists():
                        layers[cls] = f"/outputs/{p.name}/{cls}.geojson"

                if layers and not _is_demo_run(p.name):
                    runs.append({
                        "id": p.name,
                        "name": p.name.replace("_", " ").title(),
                        "layers": layers,
                        "summary": summary_data,
                        "modified_at": p.stat().st_mtime,
                    })
    runs.sort(key=lambda r: r.get("modified_at", 0), reverse=True)
    return {"runs": runs}


@app.get("/model/accuracy", tags=["Info"])
async def model_accuracy(
    run_id: str = Query(default="", description="Optional run ID to get accuracy for"),
):
    """
    Return real accuracy metrics only.

    Priority:
    1. ``outputs/<run_id>/metrics.json`` written during inference (IoU if a
       paired mask was present, otherwise softmax confidence).
    2. ``outputs/evaluation_report.json`` from the labelled test split.
    """
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / cfg.paths.output_dir
    class_names = [c for c in cfg.dataset.classes if c != "background"]

    metrics_file = out_dir / run_id / "metrics.json" if run_id else None
    if metrics_file and metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("model_architecture", cfg.model.architecture)
        payload.setdefault("encoder", cfg.model.encoder)
        return payload

    eval_file = out_dir / "evaluation_report.json"
    if (not run_id) and eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        all_classes = list(cfg.dataset.classes)
        per_class = {}
        for i, name in enumerate(all_classes):
            if name == "background":
                continue
            per_class[name] = {
                "iou": round(float(report["iou"][i]), 4) if i < len(report.get("iou", [])) else None,
                "dice": round(float(report["dice"][i]), 4) if i < len(report.get("dice", [])) else None,
                "f1": round(float(report["f1"][i]), 4) if i < len(report.get("f1", [])) else None,
                "confidence": None,
                "feature_count": 0,
            }
        return {
            "run_id": "evaluation",
            "has_ground_truth": True,
            "metrics_source": "test_split_evaluation",
            "per_class": per_class,
            "mean_iou": round(float(report.get("mean_iou") or 0), 4),
            "mean_dice": round(float(report.get("mean_dice") or 0), 4),
            "mean_f1": round(float(report.get("mean_f1") or 0), 4),
            "mean_confidence": None,
            "total_features": 0,
            "model_architecture": cfg.model.architecture,
            "encoder": cfg.model.encoder,
            "checkpoint": report.get("checkpoint"),
            "split": report.get("split"),
        }

    per_class = {}
    overall_features = 0
    for cls in class_names:
        n = 0
        if run_id:
            gj_path = out_dir / run_id / f"{cls}.geojson"
            if gj_path.exists():
                with open(gj_path, "r", encoding="utf-8") as f:
                    n = len(json.load(f).get("features", []))
        overall_features += n
        per_class[cls] = {
            "iou": None,
            "dice": None,
            "f1": None,
            "confidence": None,
            "feature_count": n,
        }

    return {
        "run_id": run_id or "none",
        "has_ground_truth": False,
        "metrics_source": "unavailable",
        "per_class": per_class,
        "mean_iou": None,
        "mean_dice": None,
        "mean_f1": None,
        "mean_confidence": None,
        "total_features": overall_features,
        "model_architecture": cfg.model.architecture,
        "encoder": cfg.model.encoder,
        "message": (
            "No ground-truth mask or evaluation report found. "
            "Upload a paired *_mask.tif with the orthophoto, or run evaluation/evaluate.py."
        ),
    }


@app.get("/heatmap/{run_id}", tags=["Info"])
async def heatmap_data(
    run_id: str,
    intensity_by: str = Query(default="area", description="Intensity metric: area, count, solar_kwp"),
):
    """
    Return centroid-based heatmap data for Leaflet.heat visualization.
    Each point is [lat, lng, intensity] where intensity is normalized 0-1.
    """
    import numpy as np

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / cfg.paths.output_dir / run_id
    classes = ["building", "road", "vegetation"]

    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    points = {}  # per-class heatmap points

    for cls in classes:
        gj_path = out_dir / f"{cls}.geojson"
        if not gj_path.exists():
            continue

        try:
            with open(gj_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        features = data.get("features", [])
        raw_points = []
        intensities = []

        for feat in features:
            props = feat.get("properties", {})
            lat = props.get("centroid_lat")
            lon = props.get("centroid_lon")
            if lat is None or lon is None:
                continue

            if intensity_by == "solar_kwp" and cls == "building":
                intensity = props.get("solar_kwp", props.get("area_sq_m", 0) * 0.15)
            elif intensity_by == "area":
                intensity = props.get("area_sq_m", 0)
            else:
                intensity = 1.0

            raw_points.append([float(lat), float(lon)])
            intensities.append(float(intensity))

        # Normalize intensities to [0.1, 1.0]
        if intensities:
            max_i = max(intensities) if max(intensities) > 0 else 1
            norm = [max(0.1, i / max_i) for i in intensities]
            cls_points = [[p[0], p[1], n] for p, n in zip(raw_points, norm)]
        else:
            cls_points = []

        points[cls] = cls_points

    # Combined heatmap (all classes merged)
    all_points = []
    for cls_pts in points.values():
        all_points.extend(cls_pts)

    return {
        "run_id": run_id,
        "intensity_by": intensity_by,
        "per_class": points,
        "combined": all_points,
        "total_points": len(all_points),
    }



# ---------------------------------------------------------------------------
# Analysis endpoints
# ---------------------------------------------------------------------------

@app.get("/analysis/dataset", tags=["Analysis"])
async def dataset_analysis_endpoint(
    img_dir: str = Query(default="", description="Image tiles directory (default: from config)"),
    mask_dir: str = Query(default="", description="Mask tiles directory (default: from config)"),
):
    """
    Run dataset analysis on the tiled image/mask dataset.
    Returns class distribution, imbalance metrics, class weights, split summary,
    and per-tile file details.
    """
    base_dir = Path(__file__).resolve().parent.parent
    _img_dir = img_dir or str(base_dir / "data" / "tiles" / "images")
    if not Path(_img_dir).is_dir() or not any(Path(_img_dir).iterdir()):
        _img_dir = str(base_dir / cfg.paths.tiles_dir)

    _mask_dir = mask_dir or str(base_dir / "data" / "tiles" / "masks")
    if not Path(_mask_dir).is_dir() or not any(Path(_mask_dir).iterdir()):
        _mask_dir = str(base_dir / getattr(cfg.paths, "masks_dir", "data/tiles/masks"))

    # Resolve relative paths
    _img_dir = str((base_dir / _img_dir).resolve()) if not Path(_img_dir).is_absolute() else _img_dir
    _mask_dir = str((base_dir / _mask_dir).resolve()) if not Path(_mask_dir).is_absolute() else _mask_dir

    out_dir = str(base_dir / cfg.paths.output_dir)

    try:
        report = analyze_dataset(
            img_dir=_img_dir,
            mask_dir=_mask_dir,
            cfg=cfg,
            output_dir=out_dir,
        )
        # Remove file_details from response to keep it lean (available in JSON file)
        report.pop("file_details", None)
        report.pop("chart_path", None)
        return report
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Dataset analysis failed")
        raise HTTPException(status_code=500, detail=f"Dataset analysis failed: {exc}")


@app.get("/analysis/dataset/chart", tags=["Analysis"])
async def dataset_chart_endpoint():
    """Return the dataset sample grid PNG image."""
    base_dir = Path(__file__).resolve().parent.parent
    chart_path = base_dir / cfg.paths.output_dir / "dataset_analysis_report.png"

    if not chart_path.exists():
        # Generate it on the fly
        try:
            _img_dir = str(base_dir / "data" / "tiles" / "images")
            _mask_dir = str(base_dir / "data" / "tiles" / "masks")
            generate_sample_grid(_img_dir, _mask_dir, chart_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Chart generation failed: {exc}")

    if chart_path.exists():
        return FileResponse(str(chart_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Dataset chart not available.")


@app.get("/analysis/prediction/{run_id}", tags=["Analysis"])
async def prediction_analysis_endpoint(run_id: str):
    """
    Run prediction analysis on a completed inference run.
    Returns per-class feature stats, land-use breakdown, building/road/vegetation
    insights, and confidence distribution.
    """
    base_dir = Path(__file__).resolve().parent.parent
    run_dir = base_dir / cfg.paths.output_dir / run_id

    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    try:
        report = analyze_predictions(run_dir=run_dir, cfg=cfg)
        report.pop("chart_path", None)
        return report
    except Exception as exc:
        logger.exception("Prediction analysis failed")
        raise HTTPException(status_code=500, detail=f"Prediction analysis failed: {exc}")


@app.get("/analysis/prediction/{run_id}/chart", tags=["Analysis"])
async def prediction_chart_endpoint(run_id: str):
    """Return the prediction analysis chart PNG for a given run."""
    base_dir = Path(__file__).resolve().parent.parent
    chart_path = base_dir / cfg.paths.output_dir / run_id / "analysis_charts.png"

    if not chart_path.exists():
        # Generate on the fly
        run_dir = base_dir / cfg.paths.output_dir / run_id
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        try:
            report = analyze_predictions(run_dir=run_dir, cfg=cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Chart generation failed: {exc}")

    if chart_path.exists():
        return FileResponse(str(chart_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Analysis chart not available.")


@app.get("/analysis/compare", tags=["Analysis"])
async def compare_runs_endpoint(
    run_ids: str = Query(default="", description="Comma-separated run IDs to compare"),
):
    """
    Compare prediction metrics across multiple runs.
    Returns per-class stats side-by-side for each run.
    """
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / cfg.paths.output_dir

    # If no specific runs, compare all available
    if run_ids:
        ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    else:
        ids = sorted([
            d.name for d in out_dir.iterdir()
            if d.is_dir()
            and not _is_demo_run(d.name)
            and any((d / f"{c}.geojson").exists() for c in ("building", "road", "vegetation"))
        ]) if out_dir.is_dir() else []

    if not ids:
        return {"runs": [], "message": "No runs found to compare."}

    comparison = []
    for rid in ids:
        run_dir = out_dir / rid
        if not run_dir.is_dir():
            continue
        try:
            report = analyze_predictions(run_dir=run_dir, cfg=cfg)
            comparison.append({
                "run_id": rid,
                "total_features": report["total_features"],
                "total_area_m2": report["total_mapped_area_m2"],
                "per_class": {
                    cls: {
                        "count": report["per_class"][cls]["count"],
                        "total_area_m2": report["per_class"][cls]["total_area_m2"],
                        "mean_area_m2": report["per_class"][cls]["mean_area_m2"],
                    }
                    for cls in ("building", "road", "vegetation")
                },
                "land_use": report["land_use"],
            })
        except Exception as exc:
            comparison.append({"run_id": rid, "error": str(exc)})

    return {"runs": comparison, "count": len(comparison)}


# ---------------------------------------------------------------------------
# Multi-Model Training & Benchmark Accuracy endpoints
# ---------------------------------------------------------------------------

@app.get("/analysis/multi-model-accuracy", tags=["Analysis"])
async def get_multi_model_accuracy():
    """
    Return the latest multi-model benchmark accuracy report across all architectures
    (U-Net, DeepLabV3+, FPN, MAnet, PSPNet, Linknet).
    """
    base_dir = Path(__file__).resolve().parent.parent
    report_file = base_dir / cfg.paths.output_dir / "multi_model_accuracy_report.json"

    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    raise HTTPException(
        status_code=404,
        detail="No multi-model accuracy report yet. Train models from the Analysis tab first.",
    )


@app.post("/analysis/train-multi-model", tags=["Analysis"])
async def train_multi_model_endpoint(
    models: str = Query(default="unet,deeplabv3plus,fpn,manet,pspnet,linknet", description="Comma-separated architecture keys"),
    epochs: int = Query(default=3, ge=1, le=50, description="Training epochs per model"),
    lr: float = Query(default=1e-4, description="Learning rate"),
    batch_size: int = Query(default=4, ge=1, le=32, description="Batch size"),
):
    """
    Train and benchmark multiple model architectures on the dataset and compare their accuracy.
    """
    model_keys = [m.strip().lower() for m in models.split(",") if m.strip()]
    if not model_keys:
        model_keys = ["unet", "deeplabv3plus", "fpn", "manet", "pspnet", "linknet"]

    try:
        report = train_and_benchmark_multi_models(
            model_keys=model_keys,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            cfg=cfg,
        )
        return report
    except Exception as exc:
        logger.exception("Multi-model training failed")
        raise HTTPException(status_code=500, detail=f"Multi-model training failed: {exc}")


@app.get("/analysis/multi-model-chart", tags=["Analysis"])
async def multi_model_chart_endpoint():
    """Return the multi-model comparison bar chart PNG image."""
    base_dir = Path(__file__).resolve().parent.parent
    chart_path = base_dir / cfg.paths.output_dir / "multi_model_accuracy_chart.png"

    if not chart_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No multi-model chart yet. Train models from the Analysis tab first.",
        )

    if chart_path.exists():
        return FileResponse(str(chart_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Multi-model chart not available.")


# ---------------------------------------------------------------------------
# Data Resources & Ingestion endpoints
# ---------------------------------------------------------------------------

@app.get("/analysis/resources", tags=["Analysis"])
async def list_data_resources():
    """
    List all local and benchmark data resources in the workspace.
    Includes raw orthophotos, paired masks, pre-tiled image/mask datasets, and model checkpoints.
    """
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / cfg.paths.raw_orthophoto_dir
    tiles_img_dir = base_dir / cfg.paths.tiles_dir / "images"
    tiles_mask_dir = base_dir / cfg.paths.tiles_dir / "masks"
    uavid_dir = base_dir / "data" / "uavid"
    checkpoints_dir = base_dir / cfg.paths.checkpoints_dir

    resources = []

    # 1. Raw Orthophotos and Masks
    if raw_dir.is_dir():
        for p in sorted(raw_dir.glob("*.*")):
            if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                is_mask = "_mask" in p.stem.lower()
                stat = p.stat()
                resources.append({
                    "name": p.name,
                    "category": "Ground Truth Mask" if is_mask else "Raw Orthophoto",
                    "type": "raster",
                    "path": f"data/raw/{p.name}",
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "is_mask": is_mask,
                    "modified_at": stat.st_mtime,
                })

    # 2. Tiled Datasets
    tile_img_count = len(list(tiles_img_dir.glob("*.tif"))) if tiles_img_dir.is_dir() else 0
    tile_mask_count = len(list(tiles_mask_dir.glob("*.tif"))) if tiles_mask_dir.is_dir() else 0
    if tile_img_count > 0:
        total_tile_bytes = sum(f.stat().st_size for f in tiles_img_dir.glob("*.tif"))
        resources.append({
            "name": "Standard Tiled Dataset (256x256)",
            "category": "Tiled Training Set",
            "type": "dataset",
            "path": f"data/tiles",
            "size_bytes": total_tile_bytes,
            "size_mb": round(total_tile_bytes / (1024 * 1024), 2),
            "tile_count": tile_img_count,
            "mask_count": tile_mask_count,
            "is_paired": tile_img_count == tile_mask_count,
        })

    # 3. UAVid Benchmark Split
    if uavid_dir.is_dir():
        uavid_files = list(uavid_dir.rglob("*.png")) + list(uavid_dir.rglob("*.jpg"))
        if uavid_files:
            total_uavid_bytes = sum(f.stat().st_size for f in uavid_files)
            resources.append({
                "name": "UAVid Benchmark Dataset",
                "category": "Public Benchmark Split",
                "type": "dataset",
                "path": "data/uavid",
                "size_bytes": total_uavid_bytes,
                "size_mb": round(total_uavid_bytes / (1024 * 1024), 2),
                "tile_count": len(uavid_files),
                "is_paired": True,
            })

    # 4. Checkpoints
    if checkpoints_dir.is_dir():
        for cp in sorted(checkpoints_dir.glob("*.pt")):
            stat = cp.stat()
            resources.append({
                "name": cp.name,
                "category": "Model Checkpoint",
                "type": "model",
                "path": f"checkpoints/{cp.name}",
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_at": stat.st_mtime,
            })

    return {
        "total_resources": len(resources),
        "resources": resources,
        "raw_dir": str(raw_dir),
        "tiles_dir": str(tiles_img_dir.parent),
    }


# ---------------------------------------------------------------------------
# Benchmark Datasets Catalog
# ---------------------------------------------------------------------------

@app.get("/datasets", tags=["Datasets"])
async def list_benchmark_datasets():
    """
    Return a curated catalog of benchmark datasets for drone orthophoto
    segmentation, with dynamic local-availability checks.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    tiles_img_dir = base_dir / "data" / "tiles" / "images"
    tiles_mask_dir = base_dir / "data" / "tiles" / "masks"
    alt_tiles_img_dir = base_dir / "data" / "tiles" / "tiles" / "images"
    alt_tiles_mask_dir = base_dir / "data" / "tiles" / "tiles" / "masks"

    def _count_tiles(prefix: str, *dirs: Path, exts=(".png", ".tif", ".tiff", ".jpg", ".jpeg")) -> int:
        total = 0
        for directory in dirs:
            if directory.is_dir():
                total += sum(
                    1 for f in directory.iterdir()
                    if f.name.lower().startswith(prefix) and f.suffix.lower() in exts
                )
        return total

    def _dir_exists_nonempty(p: Path) -> bool:
        return p.is_dir() and any(p.iterdir())

    def _file_exists(p: Path) -> bool:
        return p.is_file()

    catalog = [
        {
            "id": "isprs_potsdam",
            "name": "ISPRS Potsdam",
            "description": "High-resolution urban aerial imagery over Potsdam, Germany. 38 tiles at 6000x6000 px with 6 semantic classes. The gold standard for urban remote sensing benchmarks.",
            "sensor": "Aerial (Airplane)",
            "platform": "Manned Aircraft",
            "classes": ["Impervious Surface", "Building", "Low Vegetation", "Tree", "Car", "Clutter"],
            "num_classes": 6,
            "resolution_gsd": "5 cm/px",
            "spatial_coverage": "Potsdam, Germany",
            "access_url": "https://www.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam.aspx",
            "download_instructions": "Register at ISPRS website and download TOP (RGB) + label tiles. Place in data/potsdam/raw/",
            "adapter_script": "preprocessing/potsdam_adapter.py",
            "local_dir": "data/potsdam/raw",
            "local_available": _dir_exists_nonempty(data_dir / "potsdam" / "raw"),
            "tile_count_images": _count_tiles("potsdam_", tiles_img_dir, alt_tiles_img_dir),
            "tile_count_masks": _count_tiles("potsdam_", tiles_mask_dir, alt_tiles_mask_dir),
            "thumbnail": "/data/tiles/images/potsdam_2_10_00000_00000.png" if _file_exists(tiles_img_dir / "potsdam_2_10_00000_00000.png") else None,
        },
        {
            "id": "uavid",
            "name": "UAVid Urban Drone",
            "description": "High-resolution UAV video dataset for semantic segmentation of urban scenes. 4K oblique drone imagery with 8 classes including moving objects.",
            "sensor": "UAV Camera (DJI)",
            "platform": "Drone / UAV",
            "classes": ["Background", "Building", "Road", "Tree", "Low Vegetation", "Moving Car", "Static Car", "Human"],
            "num_classes": 8,
            "resolution_gsd": "~2.5 cm/px",
            "spatial_coverage": "Multiple Cities (China, USA)",
            "access_url": "https://uavid.nl/",
            "download_instructions": "Download from uavid.nl (registration required). Extract to data/uavid/ with train/ and valid/ subfolders.",
            "adapter_script": "preprocessing/uavid_adapter.py",
            "local_dir": "data/uavid",
            "local_available": _dir_exists_nonempty(data_dir / "uavid"),
            "tile_count_images": _count_tiles("uavid_", tiles_img_dir, alt_tiles_img_dir),
            "tile_count_masks": _count_tiles("uavid_", tiles_mask_dir, alt_tiles_mask_dir),
            "thumbnail": None,
        },
        {
            "id": "odm_aukerman",
            "name": "OpenDroneMap Aukerman",
            "description": "Community-contributed drone mapping dataset from OpenDroneMap. Dense residential neighborhood orthomosaic with visible rooftops, driveways, and tree canopy.",
            "sensor": "Consumer Drone Camera",
            "platform": "Drone / UAV",
            "classes": ["Background", "Building", "Road", "Vegetation"],
            "num_classes": 4,
            "resolution_gsd": "~3 cm/px",
            "spatial_coverage": "Aukerman, USA",
            "access_url": "https://github.com/OpenDroneMap/ODMdata",
            "download_instructions": "Clone ODMdata repo or download orthophoto. Place .tif in data/raw/",
            "adapter_script": None,
            "local_dir": "data/raw",
            "local_available": _file_exists(data_dir / "raw" / "odm_aukerman_orthophoto.tif"),
            "tile_count_images": 0,
            "tile_count_masks": 0,
            "thumbnail": None,
        },
        {
            "id": "isprs_vaihingen",
            "name": "ISPRS Vaihingen",
            "description": "Urban classification benchmark over Vaihingen, Germany. Smaller tiles than Potsdam at 9 cm GSD with IRRG (Infrared-Red-Green) channels and DSM.",
            "sensor": "Aerial (Airplane)",
            "platform": "Manned Aircraft",
            "classes": ["Impervious Surface", "Building", "Low Vegetation", "Tree", "Car", "Clutter"],
            "num_classes": 6,
            "resolution_gsd": "9 cm/px",
            "spatial_coverage": "Vaihingen, Germany",
            "access_url": "https://www.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen.aspx",
            "download_instructions": "Register at ISPRS website and download IRRG + label tiles. Note: uses IR-R-G channels, not RGB.",
            "adapter_script": None,
            "local_dir": "data/vaihingen",
            "local_available": _dir_exists_nonempty(data_dir / "vaihingen") if (data_dir / "vaihingen").exists() else False,
            "tile_count_images": 0,
            "tile_count_masks": 0,
            "thumbnail": None,
        },
        {
            "id": "landcover_ai",
            "name": "LandCover.ai",
            "description": "Polish aerial orthoimagery dataset for land cover mapping. 41 large orthophotos with building and woodland polygon annotations.",
            "sensor": "Aerial (Airplane)",
            "platform": "Manned Aircraft",
            "classes": ["Background", "Building", "Woodland", "Water", "Road"],
            "num_classes": 5,
            "resolution_gsd": "25-50 cm/px",
            "spatial_coverage": "Poland (Central Europe)",
            "access_url": "https://landcover.ai/",
            "download_instructions": "Download from landcover.ai. Extract images/ and masks/ folders to data/landcover_ai/",
            "adapter_script": None,
            "local_dir": "data/landcover_ai",
            "local_available": _dir_exists_nonempty(data_dir / "landcover_ai") if (data_dir / "landcover_ai").exists() else False,
            "tile_count_images": 0,
            "tile_count_masks": 0,
            "thumbnail": None,
        },
        {
            "id": "mass_buildings",
            "name": "Massachusetts Buildings",
            "description": "Large-scale aerial building segmentation dataset from the greater Boston area. 151 images at 1500x1500 px with binary building masks.",
            "sensor": "Aerial (Airplane)",
            "platform": "Manned Aircraft",
            "classes": ["Background", "Building"],
            "num_classes": 2,
            "resolution_gsd": "100 cm/px (1 m/px)",
            "spatial_coverage": "Massachusetts, USA",
            "access_url": "https://www.cs.toronto.edu/~vmnih/data/",
            "download_instructions": "Download from Mnih's webpage. Extract train/test/valid splits to data/mass_buildings/",
            "adapter_script": None,
            "local_dir": "data/mass_buildings",
            "local_available": _dir_exists_nonempty(data_dir / "mass_buildings") if (data_dir / "mass_buildings").exists() else False,
            "tile_count_images": 0,
            "tile_count_masks": 0,
            "thumbnail": None,
        },
        {
            "id": "spacenet2_vegas",
            "name": "SpaceNet 2 Buildings (Vegas)",
            "description": "Satellite imagery over Las Vegas with building footprint labels from the SpaceNet challenge. RGB + 8-band multispectral imagery with vector labels.",
            "sensor": "Satellite (WorldView-3)",
            "platform": "Satellite",
            "classes": ["Background", "Building"],
            "num_classes": 2,
            "resolution_gsd": "~30 cm/px",
            "spatial_coverage": "Las Vegas, Nevada, USA",
            "access_url": "https://spacenet.ai/spacenet-buildings-dataset-v2/",
            "download_instructions": "aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_2_Vegas_test_public.tar.gz . --no-sign-request",
            "adapter_script": None,
            "local_dir": "data/spacenet2",
            "local_available": _dir_exists_nonempty(data_dir / "spacenet2") if (data_dir / "spacenet2").exists() else False,
            "tile_count_images": 0,
            "tile_count_masks": 0,
            "thumbnail": None,
        },
        {
            "id": "demo_residential",
            "name": "Demo Residential Ortho",
            "description": "Built-in demo orthophoto with paired label mask for quick testing of the pipeline. Small residential area with buildings, roads, and vegetation.",
            "sensor": "Consumer Drone Camera",
            "platform": "Drone / UAV",
            "classes": ["Background", "Building", "Road", "Vegetation"],
            "num_classes": 4,
            "resolution_gsd": "~5 cm/px",
            "spatial_coverage": "Residential Area (Demo)",
            "access_url": None,
            "download_instructions": "Included with project. Already in data/raw/",
            "adapter_script": None,
            "local_dir": "data/raw",
            "local_available": _file_exists(data_dir / "raw" / "demo_residential_ortho.tif"),
            "tile_count_images": _count_tiles("demo_residential", tiles_img_dir, alt_tiles_img_dir),
            "tile_count_masks": _count_tiles("demo_residential", tiles_mask_dir, alt_tiles_mask_dir),
            "thumbnail": None,
        },
    ]

    # Compute summary stats
    total_datasets = len(catalog)
    available_count = sum(1 for d in catalog if d["local_available"])
    total_tiles = sum(d["tile_count_images"] for d in catalog)
    labeled_count = sum(1 for d in catalog if d["tile_count_masks"] > 0)

    return {
        "total_datasets": total_datasets,
        "available_locally": available_count,
        "total_tiles": total_tiles,
        "datasets_with_labels": labeled_count,
        "catalog": catalog,
    }


@app.post("/analysis/upload-resource", tags=["Analysis"])
async def upload_data_resource(
    file: UploadFile = File(..., description="Orthophoto GeoTIFF, Mask, or dataset ZIP"),
    mask_file: Optional[UploadFile] = File(None, description="Optional paired mask GeoTIFF/PNG"),
    resource_type: str = Query(default="raw_ortho", description="Type: 'raw_ortho' or 'dataset_zip'"),
):
    """
    Upload a new data resource (Orthophoto, Mask, or ZIP dataset archive) to the project.
    """
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / cfg.paths.raw_orthophoto_dir
    tiles_dir = base_dir / cfg.paths.tiles_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "upload_resource.tif"
    content = await file.read()

    uploaded_items = []

    if resource_type == "dataset_zip" or filename.lower().endswith(".zip"):
        # Extract zip directly into tiles directory or raw directory
        zip_buf = io.BytesIO(content)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            zf.extractall(tiles_dir)
            extracted_names = zf.namelist()
        return {
            "status": "success",
            "message": f"Extracted ZIP archive '{filename}' with {len(extracted_names)} files into data/tiles/.",
            "type": "dataset_zip",
            "files_count": len(extracted_names),
        }
    else:
        # Save raw orthophoto file
        dest_path = raw_dir / filename
        with open(dest_path, "wb") as f_out:
            f_out.write(content)
        uploaded_items.append(filename)

        # Save paired mask if provided
        if mask_file and mask_file.filename:
            mask_filename = mask_file.filename
            mask_content = await mask_file.read()
            mask_dest = raw_dir / mask_filename
            with open(mask_dest, "wb") as mf_out:
                mf_out.write(mask_content)
            uploaded_items.append(mask_filename)

        return {
            "status": "success",
            "message": f"Uploaded {len(uploaded_items)} file(s) to data/raw/ successfully.",
            "uploaded_files": uploaded_items,
            "destination": str(raw_dir),
        }


@app.post("/analysis/tile-resource", tags=["Analysis"])
async def tile_data_resource(
    ortho_name: str = Query(..., description="Name of the raw orthophoto file in data/raw/"),
    mask_name: Optional[str] = Query(default=None, description="Optional mask file name in data/raw/"),
    tile_size: int = Query(default=256, description="Tile dimension in pixels (e.g., 256, 512)"),
    overlap: float = Query(default=0.15, description="Fractional overlap between tiles (0.0 - 0.5)"),
    min_valid_ratio: float = Query(default=0.3, description="Minimum fraction of non-nodata pixels"),
):
    """
    Run tiling on a selected raw orthophoto and paired mask from data/raw/.
    Populates data/tiles/images and data/tiles/masks.
    """
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / cfg.paths.raw_orthophoto_dir
    ortho_path = raw_dir / ortho_name

    if not ortho_path.exists():
        raise HTTPException(status_code=404, detail=f"Orthophoto '{ortho_name}' not found in data/raw/.")

    mask_path = None
    if mask_name:
        mask_p = raw_dir / mask_name
        if mask_p.exists():
            mask_path = mask_p
    else:
        # Auto-detect paired mask
        for ext in (".tif", ".tiff", ".png"):
            candidate = raw_dir / f"{ortho_path.stem}_mask{ext}"
            if candidate.exists():
                mask_path = candidate
                break

    out_img_dir = base_dir / cfg.paths.tiles_dir / "images"
    out_mask_dir = (base_dir / cfg.paths.tiles_dir / "masks") if mask_path else None

    # Clone config with custom tiling parameters
    custom_cfg = cfg
    custom_cfg.tiling.tile_size = tile_size
    custom_cfg.tiling.overlap = overlap
    custom_cfg.tiling.min_valid_ratio = min_valid_ratio

    try:
        tiles = tile_orthophoto(
            orthophoto_path=ortho_path,
            output_img_dir=out_img_dir,
            mask_path=mask_path,
            output_mask_dir=out_mask_dir,
            cfg=custom_cfg,
        )
        return {
            "status": "success",
            "message": f"Successfully generated {len(tiles)} valid tiles from '{ortho_name}'.",
            "total_tiles": len(tiles),
            "tile_size": tile_size,
            "overlap": overlap,
            "has_mask": mask_path is not None,
            "output_img_dir": str(out_img_dir),
            "output_mask_dir": str(out_mask_dir) if out_mask_dir else None,
        }
    except Exception as exc:
        logger.exception("Tiling failed")
        raise HTTPException(status_code=500, detail=f"Tiling execution failed: {exc}")


@app.delete("/analysis/resources/{filename}", tags=["Analysis"])
async def delete_data_resource(filename: str):
    """Delete a raw data resource or checkpoint file."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / cfg.paths.raw_orthophoto_dir / filename
    if raw_path.exists() and raw_path.is_file():
        raw_path.unlink()
        return {"status": "success", "message": f"Deleted file '{filename}'."}
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")


@app.post("/predict", tags=["Inference"])
async def predict_endpoint(
    file: UploadFile = File(..., description="GeoTIFF orthophoto to segment"),
    mask: Optional[UploadFile] = File(None, description="Optional ground-truth mask for real IoU/Dice"),
    checkpoint: str = Query(default=DEFAULT_CHECKPOINT, description="Path to model checkpoint"),
    save_prob_tif: bool = Query(default=False, description="Include probability GeoTIFF in response"),
    regularize: bool = Query(default=True, description="Apply 90-degree corner regularization to buildings"),
):
    """
    Upload a GeoTIFF orthophoto and receive a ZIP of GeoJSON feature files.

    - **file**: GeoTIFF (`.tif` or `.tiff`)
    - **checkpoint**: Optional path override for the model checkpoint
    - **save_prob_tif**: Include per-class probability raster in the ZIP
    - **regularize**: Square building corners orthogonally (90-degree regularization)

    Returns a `application/zip` stream containing one `.geojson` per class.
    """
    filename = file.filename or "upload.tif"
    run_id = "".join(c if (c.isalnum() or c in "-_") else "_" for c in Path(filename).stem)
    if not run_id or _is_demo_run(run_id):
        run_id = f"run_{int(time.time())}"

    max_bytes = int(cfg.api.max_file_size_mb) * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {cfg.api.max_file_size_mb} MB.",
        )

    base_dir = Path(__file__).resolve().parent.parent
    persist_dir = base_dir / cfg.paths.output_dir / run_id
    persist_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = base_dir / cfg.paths.raw_orthophoto_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved_ckpt = str(_resolve_checkpoint(checkpoint))
        _load_model(resolved_ckpt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}")

    with tempfile.TemporaryDirectory(prefix="api_predict_") as tmp:
        ortho_path = Path(tmp) / filename
        with open(ortho_path, "wb") as f:
            f.write(content)
        with open(raw_dir / filename, "wb") as f:
            f.write(content)

        mask_path = None
        if mask and mask.filename:
            mask_bytes = await mask.read()
            mask_name = mask.filename
            mask_path = Path(tmp) / mask_name
            with open(mask_path, "wb") as mf:
                mf.write(mask_bytes)
            with open(raw_dir / mask_name, "wb") as mf:
                mf.write(mask_bytes)

        try:
            geojson_paths = predict(
                orthophoto_path=ortho_path,
                checkpoint_path=resolved_ckpt,
                output_dir=persist_dir,
                save_prob_tif=save_prob_tif,
                regularize_buildings=regularize,
                mask_path=mask_path,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for class_name, path in geojson_paths.items():
                zf.write(path, arcname=path.name)
            metrics_file = persist_dir / "metrics.json"
            if metrics_file.exists():
                zf.write(metrics_file, arcname="metrics.json")
            summary_file = persist_dir / "summary.json"
            if summary_file.exists():
                zf.write(summary_file, arcname="summary.json")
            else:
                zf.writestr("summary.json", json.dumps({
                    "source_file": filename,
                    "run_id": run_id,
                    "classes": list(geojson_paths.keys()),
                    "files": [p.name for p in geojson_paths.values()],
                }, indent=2))

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}_features.zip"',
                "X-Run-Id": run_id,
            },
        )


@app.get("/folders/scan", tags=["Folder"])
async def scan_folder(
    folder_path: str = Query(default="data/raw", description="Path to folder to scan for GeoTIFFs")
):
    """
    Scan a local directory on the server for GeoTIFF orthophoto files.
    Returns filenames, sizes, and whether a corresponding _mask.tif is present.
    """
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(folder_path)
    if not target_dir.is_absolute():
        target_dir = (base_dir / folder_path).resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory '{folder_path}' does not exist.")

    found_paths = set()
    SUPPORTED_EXTS = (
        "*.tif", "*.tiff", "*.TIF", "*.TIFF",
        "*.png", "*.PNG",
        "*.jpg", "*.jpeg", "*.JPG", "*.JPEG",
        "*.bmp", "*.BMP",
        "*.img", "*.IMG",
        "*.vrt", "*.VRT",
        "*.ecw", "*.ECW",
        "*.jp2", "*.JP2",
    )
    for ext in SUPPORTED_EXTS:
        for file_path in target_dir.glob(ext):
            if file_path.is_file():
                found_paths.add(file_path.resolve())

    files = []
    for p in sorted(found_paths, key=lambda x: x.name.lower()):
        is_mask = p.stem.endswith("_mask")
        has_paired_mask = False
        if not is_mask:
            MASK_EXTS = (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG", ".jpg", ".jpeg")
            for m_ext in MASK_EXTS:
                if (target_dir / f"{p.stem}_mask{m_ext}").exists():
                    has_paired_mask = True
                    break
        files.append({
            "name": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "is_mask": is_mask,
            "has_paired_mask": has_paired_mask,
        })

    primary_files = [f for f in files if not f["is_mask"]]
    mask_files = [f for f in files if f["is_mask"]]

    return {
        "folder_path": str(target_dir),
        "total_files": len(files),
        "primary_count": len(primary_files),
        "mask_count": len(mask_files),
        "primary_files": primary_files,
        "mask_files": mask_files,
    }


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch_endpoint(
    files: list[UploadFile] = File(..., description="List of GeoTIFF orthophotos to process"),
    checkpoint: str = Query(default=DEFAULT_CHECKPOINT, description="Path to model checkpoint"),
    save_prob_tif: bool = Query(default=False, description="Include probability GeoTIFF in response"),
):
    """
    Upload multiple GeoTIFF orthophotos and receive a combined ZIP with per-orthophoto subfolders.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    max_bytes = int(cfg.api.max_file_size_mb) * 1024 * 1024

    try:
        checkpoint = str(_resolve_checkpoint(checkpoint))
        _load_model(checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}")

    zip_buffer = io.BytesIO()
    batch_summary = {"total": len(files), "results": []}

    with tempfile.TemporaryDirectory(prefix="api_predict_batch_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for uploaded in files:
                fn = uploaded.filename or "upload.tif"
                content = await uploaded.read()
                if len(content) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{fn}' exceeds size limit of {cfg.api.max_file_size_mb} MB.",
                    )
                ortho_path = tmp_path / fn
                with open(ortho_path, "wb") as f_out:
                    f_out.write(content)

                out_dir = tmp_path / f"out_{Path(fn).stem}"
                out_dir.mkdir(parents=True, exist_ok=True)

                try:
                    geojson_paths = predict(
                        orthophoto_path=ortho_path,
                        checkpoint_path=checkpoint,
                        output_dir=out_dir,
                        save_prob_tif=save_prob_tif,
                    )
                    prefix = Path(fn).stem
                    for class_name, path in geojson_paths.items():
                        zf.write(path, arcname=f"{prefix}/{path.name}")

                    batch_summary["results"].append({
                        "file": fn,
                        "status": "success",
                        "classes": list(geojson_paths.keys()),
                    })
                except Exception as exc:
                    logger.exception(f"Prediction failed for {fn}")
                    batch_summary["results"].append({
                        "file": fn,
                        "status": "error",
                        "error": str(exc),
                    })

            zf.writestr("batch_summary.json", json.dumps(batch_summary, indent=2))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_features.zip"'},
    )


@app.post("/predict/folder", tags=["Inference"])
async def predict_folder_endpoint(
    folder_path: str = Query(default="data/raw", description="Server folder containing GeoTIFFs"),
    output_dir: str = Query(default="outputs", description="Output directory on server"),
    checkpoint: str = Query(default=DEFAULT_CHECKPOINT, description="Path to model checkpoint"),
    save_prob_tif: bool = Query(default=False, description="Save probability GeoTIFFs"),
    regularize: bool = Query(default=True, description="Apply 90-degree corner regularization to buildings"),
):
    """
    Run inference on all GeoTIFF orthophotos in a server directory.
    Outputs GeoJSONs into output_dir/<ortho_stem>/ on the server and returns JSON summary.
    """
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(folder_path)
    if not target_dir.is_absolute():
        target_dir = (base_dir / folder_path).resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory '{folder_path}' does not exist.")

    target_out = Path(output_dir)
    if not target_out.is_absolute():
        target_out = (base_dir / output_dir).resolve()
    target_out.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint = str(_resolve_checkpoint(checkpoint))
        _load_model(checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}")

    try:
        results_map = predict_folder(
            folder_path=target_dir,
            checkpoint_path=checkpoint,
            output_dir=target_out,
            save_prob_tif=save_prob_tif,
            regularize_buildings=regularize,
        )
    except Exception as exc:
        logger.exception("Folder prediction failed")
        raise HTTPException(status_code=500, detail=f"Folder prediction failed: {exc}")

    formatted_results = []
    for fname, paths in results_map.items():
        formatted_results.append({
            "file": fname,
            "status": "success",
            "features": {cls: str(p) for cls, p in paths.items()},
        })

    return {
        "status": "completed",
        "folder_path": str(target_dir),
        "total": len(results_map),
        "results": formatted_results,
    }


@app.get("/data/download-bundle", tags=["Analysis"])
async def download_dataset_bundle():
    """Download the complete packaged dataset bundle as a ZIP file."""
    base_dir = Path(__file__).resolve().parent.parent
    zip_path = base_dir / "data" / "dataset_bundle.zip"
    if zip_path.exists():
        return FileResponse(
            str(zip_path),
            media_type="application/zip",
            filename="drone_feature_extraction_dataset.zip",
        )
    raise HTTPException(status_code=404, detail="dataset_bundle.zip not found.")


# Mount static UI, outputs, and data directory
data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(data_dir)), name="data_static")

if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

outputs_dir = Path(__file__).resolve().parent.parent / cfg.paths.output_dir
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend_root")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", cfg.api.port))
    host = os.environ.get("HOST", cfg.api.host)
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
