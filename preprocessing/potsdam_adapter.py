"""
preprocessing/potsdam_adapter.py
---------------------------------
Adapter for the **ISPRS Potsdam** 2D semantic-labelling benchmark dataset.

Potsdam ships as a set of 6000×6000 pixel true-orthophoto (TOP) tiles at
5 cm GSD with corresponding RGB label images.  This module:

1. Discovers Potsdam image/label pairs from the raw download directory.
2. Converts the Potsdam 6-class RGB colour labels into integer class masks
   using either the **native 6-class** scheme or a **remapped 4-class**
   scheme that matches this project's default (background, building, road,
   vegetation).
3. Tiles the large 6000×6000 patches into training-ready 256×256 (or
   configurable) tiles using the same overlap / nodata-filtering logic as
   the main tiling pipeline.
4. Exports tile pairs into ``data/tiles/images/`` and ``data/tiles/masks/``
   so that the standard ``OrthophotoDataset`` / ``build_dataloaders`` can
   consume them without further changes.

Potsdam native colour table (RGB)
─────────────────────────────────
  Impervious surfaces  : (255, 255, 255)
  Building             : (  0,   0, 255)
  Low vegetation       : (  0, 255, 255)
  Tree                 : (  0, 255,   0)
  Car                  : (255, 255,   0)
  Clutter / background : (255,   0,   0)

Default 4-class remapping
─────────────────────────
  Potsdam class          →  Project class (index)
  ──────────────────────────────────────────────
  Clutter / background   →  background (0)
  Building               →  building   (1)
  Impervious surfaces    →  road       (2)
  Low vegetation + Tree  →  vegetation (3)
  Car                    →  background (0)   [too few pixels to help]

Usage
-----
    python preprocessing/potsdam_adapter.py --potsdam-dir data/potsdam/raw \\
                                            --out-img data/tiles/images \\
                                            --out-mask data/tiles/masks
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Potsdam colour table (RGB)
# ──────────────────────────────────────────────────────────────────────

POTSDAM_COLOR_TABLE: dict[str, tuple[int, int, int]] = {
    "impervious_surface": (255, 255, 255),
    "building":           (  0,   0, 255),
    "low_vegetation":     (  0, 255, 255),
    "tree":               (  0, 255,   0),
    "car":                (255, 255,   0),
    "clutter":            (255,   0,   0),
}

# ──────────────────────────────────────────────────────────────────────
# Remapping strategies
# ──────────────────────────────────────────────────────────────────────

# Native 6-class mapping (ISPRS ordering)
NATIVE_6CLASS_MAP: dict[str, int] = {
    "impervious_surface": 0,
    "building":           1,
    "low_vegetation":     2,
    "tree":               3,
    "car":                4,
    "clutter":            5,
}

# Project-compatible 4-class mapping
REMAP_4CLASS_MAP: dict[str, int] = {
    "clutter":            0,  # background
    "building":           1,  # building
    "impervious_surface": 2,  # road
    "low_vegetation":     3,  # vegetation
    "tree":               3,  # vegetation
    "car":                0,  # background (too sparse)
}


def rgb_label_to_classmap(
    label_rgb: np.ndarray,
    remap: str = "4class",
) -> np.ndarray:
    """
    Convert an H×W×3 uint8 RGB Potsdam label image to an H×W int64 class map.

    Parameters
    ----------
    label_rgb : np.ndarray of shape (H, W, 3), dtype uint8.
    remap     : ``"4class"`` (project default) or ``"6class"`` (native ISPRS).

    Returns
    -------
    np.ndarray of shape (H, W), dtype int64, with class indices.
    """
    mapping = REMAP_4CLASS_MAP if remap == "4class" else NATIVE_6CLASS_MAP
    h, w = label_rgb.shape[:2]
    classmap = np.zeros((h, w), dtype=np.int64)

    for cls_name, color in POTSDAM_COLOR_TABLE.items():
        mask = np.all(label_rgb == color, axis=-1)
        classmap[mask] = mapping[cls_name]

    return classmap


# ──────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────

# Typical Potsdam filenames:
#   Image:  top_potsdam_2_10_RGBIR.tif  or  top_potsdam_2_10_RGB.tif
#   Label:  top_potsdam_2_10_label.tif
_TILE_PATTERN = re.compile(
    r"top_potsdam_(\d+_\d+)_(RGB(?:IR)?)\.(tif|png)",
    re.IGNORECASE,
)
_LABEL_PATTERN = re.compile(
    r"top_potsdam_(\d+_\d+)_label\.(tif|png)",
    re.IGNORECASE,
)


def discover_pairs(
    potsdam_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """
    Scan *potsdam_dir* for image + label pairs.

    Returns a sorted list of ``(image_path, label_path)`` tuples.
    """
    potsdam_dir = Path(potsdam_dir)

    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}

    for f in potsdam_dir.rglob("*"):
        if not f.is_file():
            continue
        m = _TILE_PATTERN.match(f.name)
        if m:
            tile_id = m.group(1)
            images[tile_id] = f
            continue
        m = _LABEL_PATTERN.match(f.name)
        if m:
            tile_id = m.group(1)
            labels[tile_id] = f

    common = sorted(set(images) & set(labels))
    pairs = [(images[tid], labels[tid]) for tid in common]

    logger.info(
        f"Potsdam discovery: {len(images)} images, {len(labels)} labels, "
        f"{len(pairs)} matched pairs in '{potsdam_dir}'"
    )
    return pairs


# ──────────────────────────────────────────────────────────────────────
# Tiling
# ──────────────────────────────────────────────────────────────────────

def tile_potsdam(
    potsdam_dir: str | Path,
    out_img_dir: str | Path,
    out_mask_dir: str | Path,
    tile_size: int = 256,
    overlap: float = 0.15,
    min_valid_ratio: float = 0.3,
    remap: str = "4class",
) -> int:
    """
    Tile all Potsdam image/label pairs into 256×256 training patches.

    Parameters
    ----------
    potsdam_dir    : Root of the raw Potsdam download.
    out_img_dir    : Destination for image tiles.
    out_mask_dir   : Destination for mask tiles.
    tile_size      : Size of each square tile in pixels.
    overlap        : Fractional overlap between adjacent tiles (0 – 1).
    min_valid_ratio: Minimum fraction of non-zero pixels to keep a tile.
    remap          : Class remapping strategy (``"4class"`` or ``"6class"``).

    Returns
    -------
    Total number of tile pairs written.
    """
    out_img = Path(out_img_dir)
    out_mask = Path(out_mask_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(potsdam_dir)
    if not pairs:
        logger.warning("No Potsdam image/label pairs found. Nothing to tile.")
        return 0

    stride = int(tile_size * (1 - overlap))
    total_tiles = 0

    for img_path, lbl_path in pairs:
        tile_id = _TILE_PATTERN.match(img_path.name).group(1)  # type: ignore[union-attr]
        logger.info(f"Tiling Potsdam patch {tile_id}…")

        # Load image
        img = np.array(Image.open(img_path).convert("RGB"))
        lbl_rgb = np.array(Image.open(lbl_path).convert("RGB"))

        # Convert label to class map
        classmap = rgb_label_to_classmap(lbl_rgb, remap=remap)

        h, w = img.shape[:2]
        patch_count = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                img_tile = img[y : y + tile_size, x : x + tile_size]
                msk_tile = classmap[y : y + tile_size, x : x + tile_size]

                # Filter mostly-empty tiles
                valid_ratio = np.count_nonzero(img_tile.sum(axis=-1)) / (tile_size ** 2)
                if valid_ratio < min_valid_ratio:
                    continue

                fname = f"potsdam_{tile_id}_{y:05d}_{x:05d}.png"
                Image.fromarray(img_tile).save(out_img / fname)
                Image.fromarray(msk_tile.astype(np.uint8)).save(out_mask / fname)
                patch_count += 1

        logger.info(f"  → {patch_count} tiles from patch {tile_id}")
        total_tiles += patch_count

    logger.info(f"Potsdam tiling complete: {total_tiles} tile pairs written.")
    return total_tiles


# ──────────────────────────────────────────────────────────────────────
# Dataset integrity check
# ──────────────────────────────────────────────────────────────────────

def check_potsdam_dataset(
    img_dir: str | Path,
    mask_dir: str | Path,
    num_classes: int = 4,
) -> dict:
    """
    Validate a tiled Potsdam dataset and report statistics.

    Returns a dict with:
      - total_tiles, paired_tiles, orphan_images, orphan_masks
      - class_pixel_counts, class_pixel_fractions
      - sample_shapes (first 5 image shapes)
      - issues (list of warning strings)
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    img_files = {f.name for f in img_dir.glob("potsdam_*.png")}
    mask_files = {f.name for f in mask_dir.glob("potsdam_*.png")}

    paired = sorted(img_files & mask_files)
    orphan_imgs = sorted(img_files - mask_files)
    orphan_masks = sorted(mask_files - img_files)

    issues: list[str] = []
    if orphan_imgs:
        issues.append(f"{len(orphan_imgs)} image tile(s) missing masks")
    if orphan_masks:
        issues.append(f"{len(orphan_masks)} mask tile(s) missing images")

    # Class distribution
    class_counts = np.zeros(num_classes, dtype=np.int64)
    sample_shapes: list[tuple[int, ...]] = []

    for i, fname in enumerate(paired):
        msk = np.array(Image.open(mask_dir / fname))
        for c in range(num_classes):
            class_counts[c] += int(np.sum(msk == c))

        if i < 5:
            img = np.array(Image.open(img_dir / fname))
            sample_shapes.append(img.shape)

        # Validate mask values
        unique_vals = np.unique(msk)
        bad = [int(v) for v in unique_vals if v >= num_classes]
        if bad:
            issues.append(f"{fname}: unexpected mask values {bad}")

    total_pixels = int(class_counts.sum()) or 1
    fractions = (class_counts / total_pixels).tolist()

    report = {
        "total_images": len(img_files),
        "total_masks": len(mask_files),
        "paired_tiles": len(paired),
        "orphan_images": len(orphan_imgs),
        "orphan_masks": len(orphan_masks),
        "class_pixel_counts": class_counts.tolist(),
        "class_pixel_fractions": [round(f, 6) for f in fractions],
        "sample_shapes": sample_shapes,
        "issues": issues,
    }

    # Pretty-print
    logger.info("─" * 60)
    logger.info("ISPRS Potsdam Dataset Check")
    logger.info("─" * 60)
    logger.info(f"  Paired tiles : {report['paired_tiles']}")
    logger.info(f"  Orphan imgs  : {report['orphan_images']}")
    logger.info(f"  Orphan masks : {report['orphan_masks']}")
    logger.info(f"  Class distribution (fraction):")
    class_labels = ["background", "building", "road", "vegetation"][:num_classes]
    for j, lbl in enumerate(class_labels):
        logger.info(f"    {lbl:16s} : {fractions[j]:.4%}  ({class_counts[j]:,} px)")
    if issues:
        for iss in issues:
            logger.warning(f"  ⚠ {iss}")
    else:
        logger.info("  ✓ No issues found")
    logger.info("─" * 60)

    return report


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ISPRS Potsdam dataset for CNN training pipeline."
    )
    sub = parser.add_subparsers(dest="command")

    # -- tile subcommand --
    tile_p = sub.add_parser("tile", help="Tile raw Potsdam patches into training tiles")
    tile_p.add_argument("--potsdam-dir", required=True, help="Root of raw Potsdam download")
    tile_p.add_argument("--out-img", default="data/tiles/images", help="Image tile output dir")
    tile_p.add_argument("--out-mask", default="data/tiles/masks", help="Mask tile output dir")
    tile_p.add_argument("--tile-size", type=int, default=256)
    tile_p.add_argument("--overlap", type=float, default=0.15)
    tile_p.add_argument("--remap", default="4class", choices=["4class", "6class"])

    # -- check subcommand --
    check_p = sub.add_parser("check", help="Validate a tiled Potsdam dataset")
    check_p.add_argument("--img-dir", default="data/tiles/images")
    check_p.add_argument("--mask-dir", default="data/tiles/masks")
    check_p.add_argument("--num-classes", type=int, default=4)

    args = parser.parse_args()

    if args.command == "tile":
        tile_potsdam(
            potsdam_dir=args.potsdam_dir,
            out_img_dir=args.out_img,
            out_mask_dir=args.out_mask,
            tile_size=args.tile_size,
            overlap=args.overlap,
            remap=args.remap,
        )
    elif args.command == "check":
        check_potsdam_dataset(
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            num_classes=args.num_classes,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
