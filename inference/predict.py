"""
inference/predict.py
---------------------
End-to-end inference pipeline:
  1. Accept a GeoTIFF orthophoto path
  2. Tile it (using the same tiling as training)
  3. Run the trained CNN on each tile (batched, GPU if available)
  4. Reconstruct the full probability map via overlap merging
  5. Vectorize → GeoJSON per class
  6. (Optional) write per-class probability GeoTIFF for QA

Usage
-----
    python inference/predict.py \
        --orthophoto path/to/ortho.tif \
        --checkpoint checkpoints/best.pt \
        [--output-dir outputs/run_01]
        [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import torch
from torch.amp import autocast

from models import SegmentationModel
from postprocessing.vectorize import merge_tile_predictions, vectorize_predictions
from preprocessing.tiling import tile_orthophoto, _stride
from preprocessing.dataset import build_val_transforms, _rescale_to_uint8
from preprocessing.image_enhancement import enhance_orthophoto_image
from utils.config import load_config, _Config
from utils.logger import get_logger
from utils.metrics import numpy_segmentation_metrics, softmax_confidence_by_class

logger = get_logger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tile-level inference
# ---------------------------------------------------------------------------

def _preprocess_tile(img_path: str, tile_size: int, cfg: Optional[_Config] = None) -> np.ndarray:
    """
    Load a tile GeoTIFF, apply radiometric enhancement (CLAHE, bilateral filter, shadow recovery),
    normalise to [0,1], apply ImageNet normalisation, and return a float32 numpy array [3, H, W].
    """
    with rasterio.open(img_path) as src:
        img = src.read()  # (C, H, W)

    if img.dtype != np.uint8:
        img = _rescale_to_uint8(img)

    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    elif img.shape[0] > 3:
        img = img[:3]

    # Apply radiometric & image enhancement (CLAHE, bilateral filter, shadow reduction)
    if cfg is None or getattr(getattr(cfg, "image_processing", None), "enabled", True):
        img_rgb = np.transpose(img, (1, 2, 0))
        img_rgb = enhance_orthophoto_image(img_rgb, cfg=cfg)
        img = np.transpose(img_rgb, (2, 0, 1))

    img = img.astype(np.float32) / 255.0   # [3, H, W]
    img = (img - _MEAN[:, None, None]) / _STD[:, None, None]
    return img


def _batch_inference(
    tile_paths: list[str],
    model: SegmentationModel,
    device: torch.device,
    batch_size: int,
    tile_size: int,
    mixed_precision: bool,
    cfg: Optional[_Config] = None,
) -> list[np.ndarray]:
    """
    Run batched inference on a list of tile paths.

    Returns
    -------
    List of [C, H, W] softmax probability arrays (float32).
    """
    model.eval()
    all_probs: list[np.ndarray] = []

    for i in range(0, len(tile_paths), batch_size):
        batch_paths = tile_paths[i:i + batch_size]
        imgs = np.stack(
            [_preprocess_tile(p, tile_size, cfg=cfg) for p in batch_paths], axis=0
        )  # [B, 3, H, W]
        tensor = torch.from_numpy(imgs).to(device, non_blocking=True)

        with torch.no_grad(), autocast(device_type=str(device).split(':')[0], enabled=mixed_precision):
            logits = model(tensor)
            probs = torch.softmax(logits.float(), dim=1)

        all_probs.extend([p.cpu().numpy() for p in probs])

    return all_probs


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def _find_paired_mask(orthophoto_path: Path) -> Optional[Path]:
    """Return sibling ``<stem>_mask.<ext>`` if it exists."""
    parent = orthophoto_path.parent
    stem = orthophoto_path.stem
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".TIF", ".TIFF", ".PNG"):
        candidate = parent / f"{stem}_mask{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_mask_map(mask_path: Path, height: int, width: int) -> np.ndarray:
    """Load a single-band integer mask and resize to the orthophoto grid."""
    from scipy.ndimage import zoom

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    if mask.shape != (height, width):
        zy = height / mask.shape[0]
        zx = width / mask.shape[1]
        mask = zoom(mask, (zy, zx), order=0)
    return mask.astype(np.int64)


def _write_run_metrics(
    output_dir: Path,
    full_probs: np.ndarray,
    geojson_paths: dict[str, Path],
    cfg: _Config,
    source_file: str,
    checkpoint_path: str | Path,
    mask_path: Optional[Path],
    elapsed: float,
) -> dict:
    """Write ``metrics.json`` and ``summary.json`` for a completed run."""
    import json

    class_names = list(cfg.dataset.classes)
    conf_by_class = softmax_confidence_by_class(full_probs, class_names)
    pred = np.argmax(full_probs, axis=0)

    has_gt = mask_path is not None and Path(mask_path).exists()
    gt_metrics = None
    if has_gt:
        target = _load_mask_map(Path(mask_path), pred.shape[0], pred.shape[1])
        gt_metrics = numpy_segmentation_metrics(
            pred, target, int(cfg.model.num_classes), class_names=class_names
        )

    per_class = {}
    feature_classes = [c for c in class_names if c != "background"]
    total_features = 0
    for cls in feature_classes:
        gj_path = geojson_paths.get(cls)
        n = 0
        if gj_path and Path(gj_path).exists():
            with open(gj_path, "r", encoding="utf-8") as f:
                n = len(json.load(f).get("features", []))
        total_features += n
        conf = conf_by_class.get(cls, {})
        entry = {
            "confidence": conf.get("confidence", 0.0),
            "pixel_count": conf.get("pixel_count", 0),
            "feature_count": n,
            "iou": None,
            "dice": None,
            "f1": None,
        }
        if gt_metrics and cls in gt_metrics["per_class"]:
            entry.update(gt_metrics["per_class"][cls])
        per_class[cls] = entry

    active_conf = [v["confidence"] for v in per_class.values() if v["feature_count"] > 0 or v["pixel_count"] > 0]
    payload = {
        "run_id": Path(output_dir).name,
        "source_file": source_file,
        "checkpoint": str(checkpoint_path),
        "has_ground_truth": has_gt,
        "metrics_source": "ground_truth" if has_gt else "softmax_confidence",
        "per_class": per_class,
        "mean_iou": gt_metrics["mean_iou"] if gt_metrics else None,
        "mean_dice": gt_metrics["mean_dice"] if gt_metrics else None,
        "mean_f1": gt_metrics["mean_f1"] if gt_metrics else None,
        "mean_confidence": round(float(np.mean(active_conf)), 4) if active_conf else 0.0,
        "total_features": total_features,
        "model_architecture": cfg.model.architecture,
        "encoder": cfg.model.encoder,
        "elapsed_seconds": round(elapsed, 2),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": source_file,
                "classes": list(geojson_paths.keys()),
                "files": [Path(p).name for p in geojson_paths.values()],
                "has_ground_truth": has_gt,
                "total_features": total_features,
                "mean_iou": payload["mean_iou"],
                "mean_confidence": payload["mean_confidence"],
            },
            f,
            indent=2,
        )
    return payload


def predict(
    orthophoto_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path = "outputs",
    cfg_path: Optional[str | Path] = None,
    save_prob_tif: bool = False,
    regularize_buildings: bool = True,
    mask_path: Optional[str | Path] = None,
) -> dict[str, Path]:
    """
    Full inference pipeline for a single orthophoto.

    Parameters
    ----------
    orthophoto_path  : Path to the input GeoTIFF.
    checkpoint_path  : Path to the trained model checkpoint.
    output_dir       : Directory to write GeoJSON outputs.
    cfg_path         : Path to YAML config (None → default).
    save_prob_tif    : If True, also write a probability GeoTIFF for QA.

    Returns
    -------
    Dict mapping class name → output GeoJSON path.
    """
    t_start = time.time()
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on: {orthophoto_path}  (device={device})")

    # ---- Model ----
    ckpt = Path(checkpoint_path) if checkpoint_path else Path()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found: '{checkpoint_path}'. "
            "Train a model first (python training/train.py) or place best.pt in checkpoints/."
        )
    model = SegmentationModel(cfg).to(device)
    model.load_checkpoint(str(ckpt), device=device)

    ortho_path = Path(orthophoto_path)
    resolved_mask = Path(mask_path) if mask_path else _find_paired_mask(ortho_path)
    if resolved_mask and not resolved_mask.exists():
        resolved_mask = None

    # ---- Tile into a temp directory ----
    with tempfile.TemporaryDirectory(prefix="ortho_tiles_") as tmp_dir:
        tile_img_dir = Path(tmp_dir) / "imgs"
        tile_img_dir.mkdir()

        tile_info = tile_orthophoto(
            orthophoto_path=orthophoto_path,
            output_img_dir=tile_img_dir,
            cfg=cfg,
        )

        if not tile_info:
            raise ValueError("No valid tiles produced — check nodata/tile settings.")

        tile_paths = [t["img_path"] for t in tile_info]

        # ---- Inference ----
        logger.info(f"Running inference on {len(tile_paths)} tiles…")
        probs_list = _batch_inference(
            tile_paths=tile_paths,
            model=model,
            device=device,
            batch_size=int(cfg.training.batch_size),
            tile_size=int(cfg.tiling.tile_size),
            mixed_precision=bool(cfg.training.mixed_precision),
            cfg=cfg,
        )

        # ---- Reconstruct full ortho from tiles ----
        tile_size = int(cfg.tiling.tile_size)
        overlap = float(cfg.tiling.overlap)
        stride = _stride(tile_size, overlap)

        # Read source ortho metadata (CRS + transform)
        with rasterio.open(orthophoto_path) as src:
            src_transform = src.transform
            src_crs = src.crs
            full_h, full_w = src.height, src.width

        # Build tile_results list for merge_tile_predictions
        tile_results = []
        for ti, prob in zip(tile_info, probs_list):
            win = ti["window"]
            tile_results.append({
                "probs": prob,               # [C, tile_h, tile_w]
                "row_off": win.row_off,
                "col_off": win.col_off,
                "height": min(tile_size, full_h - win.row_off),
                "width": min(tile_size, full_w - win.col_off),
            })

        logger.info("Merging tile predictions…")
        merged_probs = merge_tile_predictions(
            tile_results,
            strategy=str(cfg.postprocessing.merge_overlap_strategy),
        )  # dict: class_id → [H, W]

        # Stack into [C, H, W] (only up to the source ortho size)
        num_classes = int(cfg.model.num_classes)
        full_probs = np.stack(
            [
                merged_probs.get(c, np.zeros((full_h, full_w), dtype=np.float32))[
                    :full_h, :full_w
                ]
                for c in range(num_classes)
            ],
            axis=0,
        )  # [C, full_h, full_w]

        # ---- (Optional) Save probability GeoTIFF ----
        if save_prob_tif:
            prob_tif_path = Path(output_dir) / "probabilities.tif"
            os.makedirs(output_dir, exist_ok=True)
            with rasterio.open(
                prob_tif_path,
                "w",
                driver="GTiff",
                count=num_classes,
                dtype=np.float32,
                height=full_h,
                width=full_w,
                crs=src_crs,
                transform=src_transform,
                compress="lzw",
            ) as dst:
                dst.write(full_probs)
            logger.info(f"Probability GeoTIFF saved → {prob_tif_path}")

        # ---- Vectorize ----
        output_paths = vectorize_predictions(
            probs=full_probs,
            transform=src_transform,
            crs=src_crs,
            output_dir=output_dir,
            cfg=cfg,
            regularize_buildings=regularize_buildings,
        )

        elapsed = time.time() - t_start
        _write_run_metrics(
            output_dir=Path(output_dir),
            full_probs=full_probs,
            geojson_paths=output_paths,
            cfg=cfg,
            source_file=ortho_path.name,
            checkpoint_path=ckpt,
            mask_path=resolved_mask,
            elapsed=elapsed,
        )

    logger.info(f"Prediction complete in {elapsed:.1f}s  →  {output_dir}")
    return output_paths


def predict_folder(
    folder_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path = "outputs",
    cfg_path: Optional[str | Path] = None,
    save_prob_tif: bool = False,
    regularize_buildings: bool = True,
) -> dict[str, dict[str, Path]]:
    """
    Run inference on all GeoTIFF orthophotos in a folder.

    Parameters
    ----------
    folder_path      : Path to directory containing GeoTIFF orthophotos.
    checkpoint_path  : Path to trained model checkpoint.
    output_dir       : Base directory to write per-ortho GeoJSON folders.
    cfg_path         : Path to YAML config.
    save_prob_tif    : If True, also write a probability GeoTIFF.
    regularize_buildings: If True, square building corners orthogonally.

    Returns
    -------
    Dict mapping filename -> (class name -> output GeoJSON path)
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder_path}")

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
    orthos = []
    for ext in SUPPORTED_EXTS:
        for f in folder_path.glob(ext):
            if "_mask" not in f.name.lower():
                orthos.append(f)
    orthos = sorted(set(orthos))

    if not orthos:
        logger.warning(f"No GeoTIFF orthophotos found in directory: {folder_path}")
        return {}

    out_base = Path(output_dir)
    results = {}
    logger.info(f"Batch inference on {len(orthos)} orthophotos in folder: {folder_path}")
    for o in orthos:
        sub_out = out_base / o.stem
        sub_out.mkdir(parents=True, exist_ok=True)
        results[o.name] = predict(
            orthophoto_path=o,
            checkpoint_path=checkpoint_path,
            output_dir=sub_out,
            cfg_path=cfg_path,
            save_prob_tif=save_prob_tif,
            regularize_buildings=regularize_buildings,
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CNN inference on a drone orthophoto (or folder of orthophotos) and export GeoJSON."
    )
    parser.add_argument("orthophoto", help="Path to input GeoTIFF orthophoto OR directory containing GeoTIFFs")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--save-prob-tif", action="store_true",
                        help="Also write a probability GeoTIFF for QA in QGIS")
    args = parser.parse_args()

    input_p = Path(args.orthophoto)
    if not input_p.exists():
        raw_candidates = list(Path("data/raw").glob("*.tif"))
        valid_candidates = [c.name for c in raw_candidates if not c.name.endswith("_mask.tif")]
        hint = f" Available orthophotos in data/raw/: {valid_candidates}" if valid_candidates else ""
        logger.error(f"Input orthophoto path not found: '{args.orthophoto}'.{hint}")
        sys.exit(1)

    if input_p.is_dir():
        predict_folder(
            folder_path=input_p,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            cfg_path=args.config,
            save_prob_tif=args.save_prob_tif,
        )
    else:
        predict(
            orthophoto_path=args.orthophoto,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            cfg_path=args.config,
            save_prob_tif=args.save_prob_tif,
        )
