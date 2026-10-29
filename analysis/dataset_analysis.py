"""
analysis/dataset_analysis.py
-----------------------------
Inspects the tiled image + mask dataset and produces a structured report
covering tile statistics, class distribution, class imbalance, and a
sample-grid visualisation.

Usage
-----
    python analysis/dataset_analysis.py
    python analysis/dataset_analysis.py --img-dir data/tiles/images --mask-dir data/tiles/masks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from utils.config import load_config, _Config
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_dataset(
    img_dir: str | Path,
    mask_dir: str | Path,
    cfg: Optional[_Config] = None,
    output_dir: str | Path = "outputs",
) -> dict:
    """
    Scan tiled images and masks to compute dataset-level statistics.

    Parameters
    ----------
    img_dir   : Directory containing image tiles (.tif).
    mask_dir  : Directory containing mask tiles (*_mask.tif).
    cfg       : Loaded config (defaults loaded if None).
    output_dir: Where to save the report JSON and sample grid PNG.

    Returns
    -------
    Dict with keys: tile_stats, class_distribution, class_weights,
    imbalance_ratio, split_summary, file_details.
    """
    if cfg is None:
        cfg = load_config()

    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(cfg.dataset.classes)
    num_classes = int(cfg.dataset.num_classes)

    # --- Discover tiles ---
    img_paths = sorted(img_dir.glob("*.tif"))
    if not img_paths:
        img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No tile images found in {img_dir}")

    logger.info(f"Found {len(img_paths)} image tiles in {img_dir}")

    # --- Scan tiles ---
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    file_details = []
    widths, heights, file_sizes = [], [], []

    for ip in img_paths:
        with rasterio.open(ip) as src:
            h, w = src.height, src.width
            bands = src.count
            dtype_str = str(src.dtypes[0])

        size_bytes = ip.stat().st_size
        widths.append(w)
        heights.append(h)
        file_sizes.append(size_bytes)

        # Find paired mask
        mask_candidates = [
            mask_dir / (ip.stem + "_mask.tif"),
            mask_dir / (ip.stem + "_mask.png"),
            mask_dir / ip.name,
        ]
        mask_path = None
        for mc in mask_candidates:
            if mc.exists():
                mask_path = mc
                break

        mask_found = mask_path is not None
        if mask_path and mask_path.exists():
            with rasterio.open(mask_path) as msrc:
                mask_data = msrc.read(1).astype(np.int64)
            for c in range(num_classes):
                pixel_counts[c] += int(np.sum(mask_data == c))
            total_pixels += mask_data.size

        file_details.append({
            "name": ip.name,
            "width": w,
            "height": h,
            "bands": bands,
            "dtype": dtype_str,
            "size_bytes": size_bytes,
            "has_mask": mask_found,
        })

    # --- Class distribution ---
    class_distribution = {}
    for i, name in enumerate(class_names):
        count = int(pixel_counts[i])
        pct = float(count / total_pixels * 100) if total_pixels > 0 else 0.0
        class_distribution[name] = {
            "pixel_count": count,
            "percentage": round(pct, 2),
        }

    # --- Class weights (inverse frequency) ---
    class_weights = []
    if total_pixels > 0 and np.all(pixel_counts > 0):
        freq = pixel_counts / total_pixels
        inv_freq = 1.0 / freq
        norm_weights = inv_freq / inv_freq.sum() * num_classes
        class_weights = [round(float(w), 4) for w in norm_weights]
    else:
        class_weights = [1.0] * num_classes

    # --- Imbalance ratio ---
    non_zero = pixel_counts[pixel_counts > 0]
    imbalance_ratio = float(non_zero.max() / non_zero.min()) if len(non_zero) > 1 else 1.0

    # --- Split summary ---
    n_total = len(img_paths)
    n_test = max(1, int(n_total * cfg.dataset.test_split))
    n_val = max(1, int(n_total * cfg.dataset.val_split))
    n_train = n_total - n_val - n_test
    split_summary = {
        "total": n_total,
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    # --- Tile stats ---
    tile_stats = {
        "count": n_total,
        "width_range": [int(min(widths)), int(max(widths))],
        "height_range": [int(min(heights)), int(max(heights))],
        "avg_file_size_kb": round(float(np.mean(file_sizes)) / 1024, 1),
        "total_size_mb": round(float(sum(file_sizes)) / (1024 * 1024), 2),
        "total_pixels": int(total_pixels),
        "masks_found": sum(1 for d in file_details if d["has_mask"]),
    }

    report = {
        "tile_stats": tile_stats,
        "class_names": class_names,
        "class_distribution": class_distribution,
        "class_weights": class_weights,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "split_summary": split_summary,
        "file_details": file_details,
    }

    # --- Save report ---
    report_path = output_dir / "dataset_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Dataset analysis report saved → {report_path}")

    # --- Generate sample grid ---
    try:
        chart_path = output_dir / "dataset_analysis_report.png"
        generate_sample_grid(img_dir, mask_dir, chart_path, max_samples=6)
        report["chart_path"] = str(chart_path)
    except Exception as exc:
        logger.warning(f"Could not generate sample grid: {exc}")

    return report


# ---------------------------------------------------------------------------
# Sample grid visualisation
# ---------------------------------------------------------------------------

def generate_sample_grid(
    img_dir: str | Path,
    mask_dir: str | Path,
    output_path: str | Path,
    max_samples: int = 6,
) -> None:
    """
    Create a grid image showing sample tiles with overlaid masks.
    Saves as a PNG to output_path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    output_path = Path(output_path)

    img_paths = sorted(img_dir.glob("*.tif"))
    if not img_paths:
        img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        logger.warning("No tiles found for sample grid.")
        return

    n = min(max_samples, len(img_paths))
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    # Class color map: background=gray, building=red, road=purple, vegetation=green
    cmap = ListedColormap([
        [0.3, 0.3, 0.3, 0.0],   # background (transparent)
        [0.8, 0.1, 0.1, 0.5],   # building (red)
        [0.5, 0.2, 0.8, 0.5],   # road (purple)
        [0.1, 0.7, 0.2, 0.5],   # vegetation (green)
    ])

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.patch.set_facecolor("#0f1117")
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ip = img_paths[idx]

        with rasterio.open(ip) as src:
            img = src.read()
        if img.dtype != np.uint8:
            for b in range(img.shape[0]):
                band = img[b].astype(np.float32)
                lo, hi = np.percentile(band, 2), np.percentile(band, 98)
                if hi > lo:
                    band = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
                else:
                    band = np.zeros_like(band)
                img[b] = band.astype(np.uint8)

        rgb = np.transpose(img[:3], (1, 2, 0))
        ax.imshow(rgb)

        # Overlay mask
        mask_candidates = [
            mask_dir / (ip.stem + "_mask.tif"),
            mask_dir / (ip.stem + "_mask.png"),
            mask_dir / ip.name,
        ]
        for mc in mask_candidates:
            if mc.exists():
                with rasterio.open(mc) as msrc:
                    mask_data = msrc.read(1)
                ax.imshow(mask_data, cmap=cmap, vmin=0, vmax=3, alpha=0.45)
                break

        ax.set_title(ip.stem, color="white", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("Dataset Sample Grid — Tiles with Mask Overlay",
                 color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Sample grid saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the tiled dataset.")
    parser.add_argument("--img-dir", default=None, help="Image tiles directory")
    parser.add_argument("--mask-dir", default=None, help="Mask tiles directory")
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_dir = args.img_dir or os.path.join(cfg.paths.tiles_dir, "images")
    mask_dir = args.mask_dir or os.path.join(cfg.paths.tiles_dir, "masks")

    report = analyze_dataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        cfg=cfg,
        output_dir=args.output_dir,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  DATASET ANALYSIS REPORT")
    print(f"{'=' * 60}")
    ts = report["tile_stats"]
    print(f"  Tiles:        {ts['count']}")
    print(f"  Masks found:  {ts['masks_found']}")
    print(f"  Total pixels: {ts['total_pixels']:,}")
    print(f"  Total size:   {ts['total_size_mb']} MB")
    print(f"{'-' * 60}")
    print(f"  {'Class':<15} {'Pixels':>12}  {'%':>7}")
    print(f"{'-' * 60}")
    for name, info in report["class_distribution"].items():
        print(f"  {name:<15} {info['pixel_count']:>12,}  {info['percentage']:>6.2f}%")
    print(f"{'-' * 60}")
    print(f"  Imbalance ratio: {report['imbalance_ratio']}")
    print(f"  Class weights:   {report['class_weights']}")
    ss = report["split_summary"]
    print(f"  Splits -> train: {ss['train']}  val: {ss['val']}  test: {ss['test']}")
    print(f"{'=' * 60}\n")
