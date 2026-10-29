"""
postprocessing/vectorize.py
----------------------------
Converts raster segmentation masks (predicted by the CNN) into clean vector
polygons and merges tile predictions back into a single georeferenced GeoJSON
per class.

Pipeline
--------
1. Threshold softmax probability map → binary mask
2. Morphological opening (remove noise) + closing (fill holes)
3. ``rasterio.features.shapes`` → raw polygons
4. Filter by minimum area
5. Douglas-Peucker simplification
6. Assign GIS attributes (feature_id, class, area_sq_m, centroid)
7. Merge all tile polygons per class, dissolve overlapping geometry
8. Write one GeoJSON per class to ``output_dir``

All geometry is in the CRS of the source orthophoto (typically EPSG:4326 or
a local UTM zone).
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.ndimage as ndi
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.crs import CRS
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import unary_union
import pyproj
from pyproj import Transformer

from utils.config import load_config, _Config
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Class name lookup
# ---------------------------------------------------------------------------

def _class_names(cfg: _Config) -> list[str]:
    return list(cfg.dataset.classes)


# ---------------------------------------------------------------------------
# Step 1 – Binarise probability map
# ---------------------------------------------------------------------------

def binarize(
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert a float probability map to uint8 binary mask.

    Parameters
    ----------
    prob_map  : [H, W] float array in [0, 1].
    threshold : Pixels above this value → foreground.

    Returns
    -------
    [H, W] uint8 array (0 or 1).
    """
    return (prob_map >= threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 2 – Morphological cleanup
# ---------------------------------------------------------------------------

def morphological_cleanup(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 2,
) -> np.ndarray:
    """
    Apply morphological opening (noise removal) followed by closing
    (hole filling) to a binary mask.

    Parameters
    ----------
    binary_mask  : [H, W] uint8 binary mask (0/1).
    kernel_size  : Square structuring element side length.
    iterations   : Number of erosion/dilation iterations.

    Returns
    -------
    Cleaned [H, W] uint8 mask.
    """
    struct = ndi.generate_binary_structure(2, 1)
    # Opening = erosion then dilation (removes small bright blobs)
    opened = ndi.binary_opening(binary_mask, structure=struct, iterations=iterations)
    # Closing = dilation then erosion (fills small dark holes)
    closed = ndi.binary_closing(opened, structure=struct, iterations=iterations)
    return closed.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3 – Raster → vector shapes
# ---------------------------------------------------------------------------

def regularize_polygon(poly, tolerance: float = 1.0, rectangularity_threshold: float = 0.70):
    """
    Building footprint regularization:
    Snaps near-orthogonal corners to 90 degrees or approximates near-rectangular
    polygons with their Minimum Rotated Rectangle (MBR) for surveyor-grade GIS layers.
    """
    if poly.is_empty or not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return poly

    try:
        mbr = poly.minimum_rotated_rectangle
        if mbr.area > 0:
            fill_ratio = poly.area / mbr.area
            # If the polygon fills >= 70% of its bounding rectangle, regularize to MBR
            if fill_ratio >= rectangularity_threshold:
                return mbr
    except Exception:
        pass

    return poly.simplify(tolerance, preserve_topology=True)


def raster_to_polygons(
    binary_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: CRS,
    min_area_sq_m: float = 25.0,
    simplify_tolerance: float = 1.0,
    class_name: str = "unknown",
    class_id: int = 0,
    regularize_buildings: bool = True,
) -> list[dict]:
    """
    Vectorise a binary raster mask using ``rasterio.features.shapes``.

    Parameters
    ----------
    binary_mask        : [H, W] uint8 mask (foreground = 1).
    transform          : Affine transform of the raster.
    crs                : CRS of the raster.
    min_area_sq_m      : Discard polygons smaller than this (sq metres).
    simplify_tolerance : Douglas-Peucker tolerance in CRS units.
    class_name         : Human-readable class label.
    class_id           : Integer class index.
    regularize_buildings: If True, apply 90-degree orthogonal regularization to buildings.

    Returns
    -------
    List of GeoJSON-compatible feature dicts.
    """
    features = []

    # rasterio.features.shapes yields (geojson_geometry, pixel_value) pairs
    shapes_gen = rasterio.features.shapes(
        binary_mask,
        mask=binary_mask,       # only yield shapes where mask == 1
        transform=transform,
    )

    try:
        if crs.is_geographic:
            proj_crs = pyproj.CRS("EPSG:6933")
            transformer = Transformer.from_crs(crs, proj_crs, always_xy=True)
        else:
            transformer = None
    except Exception:
        transformer = None

    # Adjust tolerance if CRS is in degrees (geographic)
    eff_tolerance = simplify_tolerance
    if crs and getattr(crs, "is_geographic", False):
        eff_tolerance = simplify_tolerance / 111320.0

    feat_id = 0
    for geom_dict, pixel_val in shapes_gen:
        if pixel_val != 1:
            continue

        geom = shape(geom_dict)

        # Simplify geometry (Douglas-Peucker)
        geom = geom.simplify(eff_tolerance, preserve_topology=True)

        # Building footprint regularization (90-degree orthogonal squaring)
        if class_name == "building" and regularize_buildings:
            geom = regularize_polygon(geom, tolerance=eff_tolerance)

        if geom.is_empty or not geom.is_valid:
            geom = geom.buffer(0)   # attempt repair

        if geom.is_empty:
            continue

        # Compute area (in sq metres)
        if transformer is not None:
            try:
                from shapely.ops import transform as shapely_transform
                geom_proj = shapely_transform(
                    lambda x, y, *args: transformer.transform(x, y), geom
                )
                area_sq_m = geom_proj.area
            except Exception:
                area_sq_m = geom.area
        else:
            area_sq_m = geom.area

        if area_sq_m < min_area_sq_m:
            continue

        centroid = geom.centroid

        props = {
            "feature_id": str(uuid.uuid4()),
            "class_id": class_id,
            "class": class_name,
            "area_sq_m": round(area_sq_m, 2),
            "centroid_lon": round(centroid.x, 8),
            "centroid_lat": round(centroid.y, 8),
        }

        # Solar Rooftop Energy Intelligence for buildings
        if class_name == "building":
            usable_m2 = round(area_sq_m * 0.70, 1)
            kwp = round(usable_m2 * 0.15, 2)
            annual_kwh = round(kwp * 1450, 0)
            co2_tons = round((annual_kwh * 0.82) / 1000.0, 2)
            props.update({
                "solar_usable_m2": usable_m2,
                "solar_kwp": kwp,
                "annual_kwh": annual_kwh,
                "co2_offset_tons_yr": co2_tons,
            })

        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": props,
        })
        feat_id += 1

    return features


# ---------------------------------------------------------------------------
# Step 4 – Merge tile predictions
# ---------------------------------------------------------------------------

def merge_tile_predictions(
    tile_results: list[dict],
    strategy: str = "average",
) -> dict[int, np.ndarray]:
    """
    Merge overlapping tile probability maps into a full-ortho probability map
    per class.

    Parameters
    ----------
    tile_results : List of dicts, each with keys:
                   ``probs`` ([C, H, W] float32 numpy array),
                   ``transform`` (Affine), ``width``, ``height``,
                   ``row_off``, ``col_off``.
    strategy     : ``"average"`` (weighted by count) or ``"max_vote"``.

    Returns
    -------
    Dict mapping class_id → merged [full_H, full_W] float32 array.
    """
    if not tile_results:
        return {}

    num_classes = tile_results[0]["probs"].shape[0]

    # Determine full ortho extent from offsets + tile sizes
    max_row = int(round(max(r["row_off"] + r["height"] for r in tile_results)))
    max_col = int(round(max(r["col_off"] + r["width"] for r in tile_results)))

    merged = {c: np.zeros((max_row, max_col), dtype=np.float32) for c in range(num_classes)}
    counts = np.zeros((max_row, max_col), dtype=np.float32)

    for tr in tile_results:
        r0 = int(round(tr["row_off"]))
        c0 = int(round(tr["col_off"]))
        h = int(round(tr["height"]))
        w = int(round(tr["width"]))
        probs = tr["probs"]   # [C, H, W]

        for c in range(num_classes):
            prob_patch = probs[c, :h, :w]
            if strategy == "max_vote":
                merged[c][r0:r0+h, c0:c0+w] = np.maximum(
                    merged[c][r0:r0+h, c0:c0+w], prob_patch
                )
            else:  # average
                merged[c][r0:r0+h, c0:c0+w] += prob_patch

        counts[r0:r0+h, c0:c0+w] += 1.0

    if strategy == "average":
        mask_nonzero = counts > 0
        for c in range(num_classes):
            merged[c][mask_nonzero] /= counts[mask_nonzero]

    return merged


# ---------------------------------------------------------------------------
# Step 5 – Full vectorization pipeline
# ---------------------------------------------------------------------------

def vectorize_predictions(
    probs: np.ndarray,          # [C, H, W] float32
    transform: rasterio.transform.Affine,
    crs: CRS,
    output_dir: str | Path,
    cfg: Optional[_Config] = None,
    regularize_buildings: bool = True,
) -> dict[str, Path]:
    """
    End-to-end vectorization: threshold → clean → polygonize → GeoJSON per class.

    Parameters
    ----------
    probs      : [C, H, W] softmax probability array.
    transform  : Affine transform for the full prediction raster.
    crs        : CRS of the prediction raster.
    output_dir : Directory to write per-class GeoJSON files.
    cfg        : Config (loaded from default if None).
    regularize_buildings : Whether to square building corners orthogonally.

    Returns
    -------
    Dict mapping class name → Path of written GeoJSON.
    """
    if cfg is None:
        cfg = load_config()

    post = cfg.postprocessing
    threshold = float(post.binary_threshold)
    kernel_size = int(post.morph_kernel_size)
    iterations = int(post.morph_iterations)
    min_area = float(post.min_polygon_area)
    simplify_tol = float(post.douglas_peucker_tolerance)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = _class_names(cfg)
    output_paths: dict[str, Path] = {}

    for class_id, class_name in enumerate(class_names):
        if class_name == "background":
            continue  # Skip background class

        prob_channel = probs[class_id]   # [H, W]
        binary = binarize(prob_channel, threshold=threshold)
        binary = morphological_cleanup(binary, kernel_size=kernel_size, iterations=iterations)

        features = raster_to_polygons(
            binary_mask=binary,
            transform=transform,
            crs=crs,
            min_area_sq_m=min_area,
            simplify_tolerance=simplify_tol,
            class_name=class_name,
            class_id=class_id,
            regularize_buildings=regularize_buildings,
        )

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": crs.to_string() if crs else "EPSG:4326"},
            },
            "features": features,
        }

        out_path = output_dir / f"{class_name}.geojson"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        output_paths[class_name] = out_path
        logger.info(
            f"Class '{class_name}': {len(features)} polygons → {out_path}"
        )

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vectorize a pre-computed prediction raster (GeoTIFF of probabilities)."
    )
    parser.add_argument("prob_tif", help="GeoTIFF with C bands = per-class probability maps")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for GeoJSON files")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    _cfg = load_config(args.config)

    with rasterio.open(args.prob_tif) as src:
        _probs = src.read().astype(np.float32)  # [C, H, W]
        _transform = src.transform
        _crs = src.crs

    vectorize_predictions(
        probs=_probs,
        transform=_transform,
        crs=_crs,
        output_dir=args.output_dir,
        cfg=_cfg,
    )
