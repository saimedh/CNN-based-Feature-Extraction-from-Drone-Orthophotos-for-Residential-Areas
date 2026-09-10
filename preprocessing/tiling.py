"""
preprocessing/tiling.py
------------------------
Splits a large GeoTIFF orthophoto into overlapping square tiles and saves
them together with their corresponding mask tiles.

Key design decisions
--------------------
* Tiles are written as GeoTIFF so that geo-referencing metadata is preserved.
* Non-data (nodata) pixels are detected and tiles with too much nodata are
  discarded (controlled by ``min_valid_ratio`` in config).
* Overlap between adjacent tiles is controlled by ``overlap`` in config.
  A 15 % overlap prevents artefacts at tile boundaries.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window

from utils.config import load_config, _Config
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stride(tile_size: int, overlap: float) -> int:
    """Return the step (in pixels) between tile origins given the overlap."""
    return max(1, int(tile_size * (1.0 - overlap)))


def _pad_array(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Zero-pad the last two axes of ``arr`` to reach (target_h, target_w).

    Works for both (C, H, W) and (H, W) arrays.
    """
    if arr.ndim == 3:
        _, h, w = arr.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    else:
        h, w = arr.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")


# ---------------------------------------------------------------------------
# Core tiling function
# ---------------------------------------------------------------------------

def tile_orthophoto(
    orthophoto_path: str | Path,
    output_img_dir: str | Path,
    mask_path: Optional[str | Path] = None,
    output_mask_dir: Optional[str | Path] = None,
    cfg: Optional[_Config] = None,
) -> list[dict]:
    """
    Tile a GeoTIFF orthophoto (and optionally its paired mask) into overlapping
    square patches and write them to disk.

    Parameters
    ----------
    orthophoto_path  : Path to the source GeoTIFF.
    output_img_dir   : Directory where image tiles are written.
    mask_path        : (Optional) Path to the paired single-band class mask GeoTIFF.
    output_mask_dir  : (Optional) Directory where mask tiles are written.
    cfg              : Loaded config.  Loaded from default location if ``None``.

    Returns
    -------
    List of dicts, one per tile, containing:
        ``tile_id``, ``img_path``, ``mask_path`` (if applicable),
        ``row``, ``col``, ``window`` (rasterio Window), ``transform``.
    """
    if cfg is None:
        cfg = load_config()

    tile_size: int = cfg.tiling.tile_size
    overlap: float = cfg.tiling.overlap
    min_valid: float = cfg.tiling.min_valid_ratio

    os.makedirs(output_img_dir, exist_ok=True)
    if output_mask_dir:
        os.makedirs(output_mask_dir, exist_ok=True)

    stride = _stride(tile_size, overlap)
    tiles_info: list[dict] = []

    with rasterio.open(orthophoto_path) as src:
        img_h, img_w = src.height, src.width
        src_crs = src.crs
        src_transform = src.transform
        nodata_val = src.nodata

        num_rows = math.ceil((img_h - tile_size) / stride) + 1
        num_cols = math.ceil((img_w - tile_size) / stride) + 1

        logger.info(
            f"Tiling '{Path(orthophoto_path).name}' "
            f"({img_h}×{img_w} px) → "
            f"{num_rows}×{num_cols} = {num_rows * num_cols} tiles "
            f"(tile={tile_size}, stride={stride})"
        )

        mask_src = None
        if mask_path:
            mask_src = rasterio.open(mask_path)

        tile_id = 0
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                row_off = row_idx * stride
                col_off = col_idx * stride

                # Clamp to image bounds
                actual_h = min(tile_size, img_h - row_off)
                actual_w = min(tile_size, img_w - col_off)

                win = Window(col_off, row_off, actual_w, actual_h)

                # Read image patch  (C, H, W)
                img_patch = src.read(window=win)  # shape: (bands, actual_h, actual_w)

                # Check nodata validity
                if nodata_val is not None:
                    valid_mask = img_patch != nodata_val
                    valid_ratio = valid_mask.mean()
                else:
                    valid_ratio = 1.0

                if valid_ratio < min_valid:
                    continue  # Skip mostly-empty tiles

                # Pad to full tile_size if near-edge
                img_patch = _pad_array(img_patch, tile_size, tile_size)

                # Compute the affine transform for this tile
                tile_transform = rasterio.windows.transform(win, src_transform)

                # --- Write image tile ---
                stem = Path(orthophoto_path).stem
                img_tile_path = Path(output_img_dir) / f"{stem}_r{row_idx:04d}_c{col_idx:04d}.tif"

                profile = {
                    "driver": "GTiff",
                    "count": img_patch.shape[0],
                    "dtype": img_patch.dtype,
                    "height": tile_size,
                    "width": tile_size,
                    "crs": src_crs,
                    "transform": tile_transform,
                    "compress": "lzw",
                }
                with rasterio.open(img_tile_path, "w", **profile) as dst:
                    dst.write(img_patch)

                tile_record: dict = {
                    "tile_id": tile_id,
                    "img_path": str(img_tile_path),
                    "mask_path": None,
                    "row": row_idx,
                    "col": col_idx,
                    "window": win,
                    "transform": tile_transform,
                }

                # --- Write mask tile (if provided) ---
                if mask_src is not None and output_mask_dir:
                    msk_patch = mask_src.read(1, window=win)  # single band, (H, W)
                    msk_patch = _pad_array(msk_patch, tile_size, tile_size)

                    msk_tile_path = (
                        Path(output_mask_dir)
                        / f"{stem}_r{row_idx:04d}_c{col_idx:04d}_mask.tif"
                    )
                    msk_profile = {
                        "driver": "GTiff",
                        "count": 1,
                        "dtype": msk_patch.dtype,
                        "height": tile_size,
                        "width": tile_size,
                        "crs": src_crs,
                        "transform": tile_transform,
                        "compress": "lzw",
                    }
                    with rasterio.open(msk_tile_path, "w", **msk_profile) as dst:
                        dst.write(msk_patch[np.newaxis, ...])

                    tile_record["mask_path"] = str(msk_tile_path)

                tiles_info.append(tile_record)
                tile_id += 1

        if mask_src is not None:
            mask_src.close()

    logger.info(f"Saved {len(tiles_info)} valid tiles.")
    return tiles_info


def tile_folder(
    folder_path: str | Path,
    output_img_dir: str | Path = "data/tiles",
    output_mask_dir: Optional[str | Path] = "data/masks",
    cfg: Optional[_Config] = None,
) -> list[dict]:
    """
    Tile all GeoTIFF orthophotos in a folder, automatically pairing with
    <stem>_mask.tif if available.
    """
    folder = Path(folder_path)
    orthos = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for f in folder.glob(ext):
            if not f.stem.endswith("_mask"):
                orthos.append(f)
    orthos = sorted(set(orthos))

    all_tiles = []
    logger.info(f"Tiling {len(orthos)} orthophotos in folder: {folder_path}")
    for o in orthos:
        mask_path = None
        for m_ext in (".tif", ".tiff", ".TIF", ".TIFF"):
            candidate = folder / f"{o.stem}_mask{m_ext}"
            if candidate.exists():
                mask_path = candidate
                break

        tiles = tile_orthophoto(
            orthophoto_path=o,
            output_img_dir=output_img_dir,
            mask_path=mask_path,
            output_mask_dir=output_mask_dir if mask_path else None,
            cfg=cfg,
        )
        all_tiles.extend(tiles)
    return all_tiles


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Tile a GeoTIFF orthophoto or a folder of orthophotos.")
    parser.add_argument("orthophoto", help="Path to input GeoTIFF OR folder containing GeoTIFFs")
    parser.add_argument("--mask", default=None, help="Path to class-label mask GeoTIFF (for single-file mode)")
    parser.add_argument("--out-img", default="data/tiles", help="Output image tile dir")
    parser.add_argument("--out-mask", default="data/masks", help="Output mask tile dir")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = Path(args.orthophoto)
    if input_path.is_dir():
        info = tile_folder(input_path, args.out_img, args.out_mask, cfg)
    else:
        info = tile_orthophoto(args.orthophoto, args.out_img, args.mask, args.out_mask, cfg)

    manifest_path = Path(args.out_img) / "tile_manifest.json"
    serialisable = [
        {k: str(v) if not isinstance(v, (int, str, type(None))) else v
         for k, v in t.items()}
        for t in info
    ]
    with open(manifest_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Manifest written → {manifest_path}")
