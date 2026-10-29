"""
analysis/prediction_analysis.py
---------------------------------
Analyzes the GeoJSON outputs from a completed inference run to produce
spatial insights: per-class feature stats, land-use breakdown, building
compactness, road connectivity, vegetation fragmentation, and charts.

Usage
-----
    python analysis/prediction_analysis.py --run-id odm_aukerman
    python analysis/prediction_analysis.py --geojson-dir outputs/aukerman
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.config import load_config, _Config
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------

def _load_geojson(path: Path) -> list[dict]:
    """Load GeoJSON features from a file, returning an empty list on failure."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("features", [])
    except Exception as exc:
        logger.warning(f"Could not load {path}: {exc}")
        return []


def _polygon_area_from_coords(coords: list) -> float:
    """Compute area of a simple polygon from its coordinate ring using the shoelace formula."""
    if not coords or len(coords) < 3:
        return 0.0
    ring = coords[0] if isinstance(coords[0][0], (list, tuple)) else coords
    n = len(ring)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return abs(area) / 2.0


def _polygon_perimeter_from_coords(coords: list) -> float:
    """Compute perimeter of a polygon ring."""
    if not coords:
        return 0.0
    ring = coords[0] if isinstance(coords[0][0], (list, tuple)) else coords
    n = len(ring)
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = ring[j][0] - ring[i][0]
        dy = ring[j][1] - ring[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)
    return perimeter


def _extract_feature_stats(features: list[dict]) -> dict:
    """Extract area and perimeter statistics from GeoJSON features."""
    areas, perimeters = [], []

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        # Try properties first (pre-computed by vectorize.py)
        area = props.get("area_sq_m", None)
        if area is None and geom.get("type") in ("Polygon", "MultiPolygon"):
            coords = geom.get("coordinates", [])
            if geom["type"] == "Polygon":
                area = _polygon_area_from_coords(coords)
            elif geom["type"] == "MultiPolygon":
                area = sum(_polygon_area_from_coords(c) for c in coords)
        if area is not None:
            areas.append(float(area))

        perim = props.get("perimeter_m", None)
        if perim is None and geom.get("type") == "Polygon":
            coords = geom.get("coordinates", [])
            perim = _polygon_perimeter_from_coords(coords)
        if perim is not None:
            perimeters.append(float(perim))

    if not areas:
        return {
            "count": len(features),
            "total_area_m2": 0,
            "mean_area_m2": 0,
            "median_area_m2": 0,
            "std_area_m2": 0,
            "min_area_m2": 0,
            "max_area_m2": 0,
        }

    arr = np.array(areas)
    result = {
        "count": len(features),
        "total_area_m2": round(float(arr.sum()), 2),
        "mean_area_m2": round(float(arr.mean()), 2),
        "median_area_m2": round(float(np.median(arr)), 2),
        "std_area_m2": round(float(arr.std()), 2),
        "min_area_m2": round(float(arr.min()), 2),
        "max_area_m2": round(float(arr.max()), 2),
    }

    if perimeters:
        parr = np.array(perimeters)
        result["mean_perimeter_m"] = round(float(parr.mean()), 2)
        result["total_perimeter_m"] = round(float(parr.sum()), 2)

    return result


# ---------------------------------------------------------------------------
# Class-specific analysis
# ---------------------------------------------------------------------------

def _analyze_buildings(features: list[dict]) -> dict:
    """Building-specific metrics: compactness, estimated stories, solar potential."""
    stats = _extract_feature_stats(features)

    compactness_scores = []
    solar_estimates = []

    for feat in features:
        props = feat.get("properties", {})
        area = props.get("area_sq_m", 0)
        perim = props.get("perimeter_m", 0)

        # Compactness (isoperimetric quotient): 4π × area / perimeter²
        # Perfect circle = 1.0, more irregular = lower
        if perim > 0 and area > 0:
            compactness = (4 * math.pi * area) / (perim * perim)
            compactness_scores.append(min(1.0, compactness))

        # Solar estimate: ~150 W/m² average rooftop solar potential
        if area > 0:
            solar_kwp = area * 0.15  # kWp estimate
            solar_estimates.append(solar_kwp)

    if compactness_scores:
        stats["mean_compactness"] = round(float(np.mean(compactness_scores)), 4)
        stats["regularity"] = "High" if stats["mean_compactness"] > 0.7 else (
            "Medium" if stats["mean_compactness"] > 0.4 else "Low"
        )
    else:
        stats["mean_compactness"] = 0
        stats["regularity"] = "N/A"

    if solar_estimates:
        stats["total_solar_kwp"] = round(float(sum(solar_estimates)), 2)
        stats["mean_solar_kwp"] = round(float(np.mean(solar_estimates)), 2)
    else:
        stats["total_solar_kwp"] = 0
        stats["mean_solar_kwp"] = 0

    return stats


def _analyze_roads(features: list[dict]) -> dict:
    """Road-specific metrics: total length, average width estimate."""
    stats = _extract_feature_stats(features)

    lengths = []
    for feat in features:
        props = feat.get("properties", {})
        area = props.get("area_sq_m", 0)
        perim = props.get("perimeter_m", 0)

        # Estimate road length from elongated polygon:
        # For a rectangle with length L and width W:  area = L*W, perim = 2*(L+W)
        # If L >> W, then L ≈ perim/2, W ≈ 2*area/perim
        if perim > 0 and area > 0:
            estimated_length = perim / 2.0
            estimated_width = 2.0 * area / perim
            lengths.append(estimated_length)

    if lengths:
        stats["total_road_length_m"] = round(float(sum(lengths)), 2)
        stats["mean_segment_length_m"] = round(float(np.mean(lengths)), 2)
        if stats["count"] > 0 and stats["total_area_m2"] > 0 and stats.get("total_perimeter_m", 0) > 0:
            stats["avg_width_m"] = round(
                2.0 * stats["total_area_m2"] / stats["total_perimeter_m"], 2
            )
        else:
            stats["avg_width_m"] = 0
    else:
        stats["total_road_length_m"] = 0
        stats["mean_segment_length_m"] = 0
        stats["avg_width_m"] = 0

    return stats


def _analyze_vegetation(features: list[dict]) -> dict:
    """Vegetation-specific metrics: fragmentation, canopy stats."""
    stats = _extract_feature_stats(features)

    if stats["count"] > 0 and stats["total_area_m2"] > 0:
        # Fragmentation index: more patches of smaller size = higher fragmentation
        # Simple: mean_area / max_area (closer to 1 = uniform, closer to 0 = fragmented)
        if stats["max_area_m2"] > 0:
            stats["fragmentation_index"] = round(
                1.0 - (stats["mean_area_m2"] / stats["max_area_m2"]), 4
            )
        else:
            stats["fragmentation_index"] = 0

        stats["canopy_area_m2"] = stats["total_area_m2"]
        stats["patch_density"] = round(
            stats["count"] / (stats["total_area_m2"] / 10000.0)  # patches per hectare
            if stats["total_area_m2"] > 0 else 0, 2
        )
    else:
        stats["fragmentation_index"] = 0
        stats["canopy_area_m2"] = 0
        stats["patch_density"] = 0

    return stats


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_predictions(
    run_dir: str | Path,
    cfg: Optional[_Config] = None,
) -> dict:
    """
    Analyze GeoJSON prediction outputs from a completed inference run.

    Parameters
    ----------
    run_dir : Directory containing building.geojson, road.geojson, vegetation.geojson.
    cfg     : Loaded config (defaults loaded if None).

    Returns
    -------
    Dict with per-class analysis, land-use breakdown, and overall summary.
    """
    if cfg is None:
        cfg = load_config()

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load features per class
    building_feats = _load_geojson(run_dir / "building.geojson")
    road_feats = _load_geojson(run_dir / "road.geojson")
    vegetation_feats = _load_geojson(run_dir / "vegetation.geojson")

    # Class-specific analysis
    building_analysis = _analyze_buildings(building_feats)
    road_analysis = _analyze_roads(road_feats)
    vegetation_analysis = _analyze_vegetation(vegetation_feats)

    # Total feature count
    total_features = (
        building_analysis["count"]
        + road_analysis["count"]
        + vegetation_analysis["count"]
    )

    # Land use breakdown
    total_area = (
        building_analysis["total_area_m2"]
        + road_analysis["total_area_m2"]
        + vegetation_analysis["total_area_m2"]
    )

    land_use = {}
    if total_area > 0:
        land_use = {
            "building": {
                "area_m2": building_analysis["total_area_m2"],
                "percentage": round(building_analysis["total_area_m2"] / total_area * 100, 2),
            },
            "road": {
                "area_m2": road_analysis["total_area_m2"],
                "percentage": round(road_analysis["total_area_m2"] / total_area * 100, 2),
            },
            "vegetation": {
                "area_m2": vegetation_analysis["total_area_m2"],
                "percentage": round(vegetation_analysis["total_area_m2"] / total_area * 100, 2),
            },
        }
    else:
        for cls in ("building", "road", "vegetation"):
            land_use[cls] = {"area_m2": 0, "percentage": 0}

    # Confidence distribution (from feature properties if available)
    all_confidences = []
    for feats in (building_feats, road_feats, vegetation_feats):
        for feat in feats:
            conf = feat.get("properties", {}).get("confidence", None)
            if conf is not None:
                all_confidences.append(float(conf))

    confidence_stats = {}
    if all_confidences:
        carr = np.array(all_confidences)
        confidence_stats = {
            "mean": round(float(carr.mean()), 4),
            "median": round(float(np.median(carr)), 4),
            "std": round(float(carr.std()), 4),
            "min": round(float(carr.min()), 4),
            "max": round(float(carr.max()), 4),
            "histogram": _compute_histogram(carr, bins=10),
        }

    report = {
        "run_id": run_dir.name,
        "total_features": total_features,
        "total_mapped_area_m2": round(total_area, 2),
        "total_mapped_area_hectares": round(total_area / 10000.0, 4),
        "per_class": {
            "building": building_analysis,
            "road": road_analysis,
            "vegetation": vegetation_analysis,
        },
        "land_use": land_use,
        "confidence": confidence_stats,
        "features_per_hectare": round(
            total_features / (total_area / 10000.0) if total_area > 0 else 0, 2
        ),
    }

    # Save report
    report_path = run_dir / "prediction_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Prediction analysis saved → {report_path}")

    # Generate charts
    try:
        chart_path = run_dir / "analysis_charts.png"
        generate_analysis_charts(report, chart_path)
        report["chart_path"] = str(chart_path)
    except Exception as exc:
        logger.warning(f"Could not generate charts: {exc}")

    return report


def _compute_histogram(arr: np.ndarray, bins: int = 10) -> dict:
    """Return a serialisable histogram dict."""
    counts, edges = np.histogram(arr, bins=bins, range=(0, 1))
    return {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 3) for e in edges],
    }


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_analysis_charts(report: dict, output_path: str | Path) -> None:
    """
    Generate a multi-panel analysis chart:
      1. Land-use pie chart
      2. Feature count bar chart
      3. Area comparison bar chart
      4. Confidence histogram (if available)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    has_confidence = bool(report.get("confidence", {}).get("histogram"))
    n_panels = 4 if has_confidence else 3

    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5, 5))
    fig.patch.set_facecolor("#0f1117")

    classes = ["building", "road", "vegetation"]
    colors = ["#ef4444", "#a855f7", "#22c55e"]

    # --- Panel 1: Land-use pie chart ---
    ax1 = axes[0]
    land_use = report.get("land_use", {})
    sizes = [land_use.get(c, {}).get("percentage", 0) for c in classes]
    if sum(sizes) > 0:
        ax1.pie(
            sizes, labels=classes, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"color": "white", "fontsize": 9},
            wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
        )
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                 color="white", fontsize=12, transform=ax1.transAxes)
    ax1.set_title("Land Use Breakdown", color="white", fontsize=12, fontweight="bold", pad=12)
    ax1.set_facecolor("#0f1117")

    # --- Panel 2: Feature count bar chart ---
    ax2 = axes[1]
    per_class = report.get("per_class", {})
    counts = [per_class.get(c, {}).get("count", 0) for c in classes]
    bars = ax2.bar(classes, counts, color=colors, edgecolor="#1a1a2e", linewidth=1.5,
                   width=0.6, zorder=3)
    for bar, val in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                 str(val), ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
    ax2.set_title("Feature Count", color="white", fontsize=12, fontweight="bold", pad=12)
    ax2.set_facecolor("#14141f")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#333")
    ax2.spines["left"].set_color("#333")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("Count", color="white", fontsize=10)
    ax2.grid(axis="y", color="#222", linestyle="--", alpha=0.5)

    # --- Panel 3: Area comparison ---
    ax3 = axes[2]
    areas = [per_class.get(c, {}).get("total_area_m2", 0) for c in classes]
    bars3 = ax3.bar(classes, areas, color=colors, edgecolor="#1a1a2e", linewidth=1.5,
                    width=0.6, zorder=3)
    for bar, val in zip(bars3, areas):
        label = f"{val:.0f}" if val < 10000 else f"{val / 1000:.1f}k"
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(areas) * 0.02,
                 label, ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
    ax3.set_title("Total Area (m²)", color="white", fontsize=12, fontweight="bold", pad=12)
    ax3.set_facecolor("#14141f")
    ax3.tick_params(colors="white")
    ax3.spines["bottom"].set_color("#333")
    ax3.spines["left"].set_color("#333")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_ylabel("Area (m²)", color="white", fontsize=10)
    ax3.grid(axis="y", color="#222", linestyle="--", alpha=0.5)

    # --- Panel 4: Confidence histogram (optional) ---
    if has_confidence:
        ax4 = axes[3]
        hist = report["confidence"]["histogram"]
        bin_edges = hist["bin_edges"]
        counts_h = hist["counts"]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts_h))]
        bin_widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(counts_h))]
        ax4.bar(bin_centers, counts_h, width=bin_widths, color="#38bdf8",
                edgecolor="#1a1a2e", linewidth=1.5, zorder=3)
        ax4.set_title("Confidence Distribution", color="white", fontsize=12, fontweight="bold", pad=12)
        ax4.set_facecolor("#14141f")
        ax4.tick_params(colors="white")
        ax4.spines["bottom"].set_color("#333")
        ax4.spines["left"].set_color("#333")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.set_xlabel("Confidence", color="white", fontsize=10)
        ax4.set_ylabel("Count", color="white", fontsize=10)
        ax4.grid(axis="y", color="#222", linestyle="--", alpha=0.5)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Analysis charts saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prediction results.")
    parser.add_argument("--run-id", default=None, help="Run ID (subfolder name under outputs/)")
    parser.add_argument("--geojson-dir", default=None, help="Direct path to GeoJSON output directory")
    parser.add_argument("--config", default=None, help="Config YAML path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.geojson_dir:
        run_dir = Path(args.geojson_dir)
    elif args.run_id:
        run_dir = Path(cfg.paths.output_dir) / args.run_id
    else:
        # Default to first available run
        out_base = Path(cfg.paths.output_dir)
        runs = sorted([d for d in out_base.iterdir() if d.is_dir()]) if out_base.is_dir() else []
        if not runs:
            print("No runs found. Run inference first.")
            sys.exit(1)
        run_dir = runs[0]
        print(f"Auto-selected run: {run_dir.name}")

    report = analyze_predictions(run_dir=run_dir, cfg=cfg)

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  PREDICTION ANALYSIS -- {report['run_id']}")
    print(f"{'=' * 65}")
    print(f"  Total features:        {report['total_features']}")
    print(f"  Mapped area:           {report['total_mapped_area_m2']:,.2f} m2")
    print(f"                         ({report['total_mapped_area_hectares']:.4f} ha)")
    print(f"  Features / hectare:    {report['features_per_hectare']}")
    print(f"{'-' * 65}")
    print(f"  {'Class':<15} {'Count':>8}  {'Area (m2)':>12}  {'Mean (m2)':>10}")
    print(f"{'-' * 65}")
    for cls in ("building", "road", "vegetation"):
        pc = report["per_class"].get(cls, {})
        print(
            f"  {cls:<15} {pc.get('count', 0):>8}  "
            f"{pc.get('total_area_m2', 0):>12,.2f}  "
            f"{pc.get('mean_area_m2', 0):>10,.2f}"
        )
    print(f"{'-' * 65}")
    lu = report.get("land_use", {})
    print(f"\n  Land Use:  "
          f"Building {lu.get('building', {}).get('percentage', 0):.1f}% | "
          f"Road {lu.get('road', {}).get('percentage', 0):.1f}% | "
          f"Vegetation {lu.get('vegetation', {}).get('percentage', 0):.1f}%")

    # Building specifics
    bp = report["per_class"].get("building", {})
    if bp.get("count", 0) > 0:
        print(f"\n  Building Insights:")
        print(f"    Shape regularity:   {bp.get('regularity', 'N/A')}")
        print(f"    Mean compactness:   {bp.get('mean_compactness', 0):.4f}")
        print(f"    Solar potential:    {bp.get('total_solar_kwp', 0):.2f} kWp")

    # Road specifics
    rp = report["per_class"].get("road", {})
    if rp.get("count", 0) > 0:
        print(f"\n  Road Insights:")
        print(f"    Total road length:  {rp.get('total_road_length_m', 0):.2f} m")
        print(f"    Avg width:          {rp.get('avg_width_m', 0):.2f} m")

    # Vegetation specifics
    vp = report["per_class"].get("vegetation", {})
    if vp.get("count", 0) > 0:
        print(f"\n  Vegetation Insights:")
        print(f"    Canopy area:        {vp.get('canopy_area_m2', 0):,.2f} m2")
        print(f"    Fragmentation:      {vp.get('fragmentation_index', 0):.4f}")
        print(f"    Patch density:      {vp.get('patch_density', 0):.2f} /ha")

    print(f"\n{'=' * 65}\n")
