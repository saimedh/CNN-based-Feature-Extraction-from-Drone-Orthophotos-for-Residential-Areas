"""
tests/test_geocoding.py
-----------------------
Unit tests for:
1. NominatimCache (storing, retrieving, 5-decimal rounding, avoiding duplicate requests).
2. NominatimRateLimiter (rate-limiting queueing with >= 1.0s interval).
3. 100m Grid-cell assignment and enrichment logic.
"""

import asyncio
import json
import math
import tempfile
import time
from pathlib import Path

import pytest
from shapely.geometry import Polygon, mapping

from services.geocoding import (
    NominatimCache,
    NominatimRateLimiter,
    NominatimService,
    round_coord,
)


# ============================================================================
# 1. Cache Tests
# ============================================================================

def test_cache_round_coord():
    """Verify coordinate rounding to 5 decimal places (~1.1m precision)."""
    assert round_coord(41.3038123456) == 41.30381
    assert round_coord(-81.7505199999) == -81.75052
    assert round_coord(12.345674) == 12.34567
    assert round_coord(12.345676) == 12.34568


def test_cache_reverse_storage_and_retrieval():
    """Verify caching stores and retrieves by rounded coordinates without re-querying."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_cache.db"
        cache = NominatimCache(db_path=str(db_file))

        # Initial lookup must be None
        assert cache.get_reverse(41.303814, -81.750516) is None

        test_data = {
            "street": "Aukerman Way",
            "locality": "North Royalton",
            "suburb": "Cuyahoga",
            "city": "Cleveland",
            "postcode": "44133",
            "house_number": "123",
            "building_name": "Park Pavilion",
            "display_name": "123 Aukerman Way, Cleveland, OH",
            "source": "OpenStreetMap",
        }

        # Store with unrounded coordinates
        cache.set_reverse(41.303814, -81.750516, test_data)

        # Lookup with slightly different coordinate within 5-decimal precision (~1m)
        cached = cache.get_reverse(41.303811, -81.750519)
        assert cached is not None
        assert cached["street"] == "Aukerman Way"
        assert cached["locality"] == "North Royalton"
        assert cached["city"] == "Cleveland"
        assert cached["house_number"] == "123"
        assert cached["source"] == "OpenStreetMap"
        assert cached["cached"] is True


def test_cache_search_storage_and_retrieval():
    """Verify search queries are cached with case-insensitivity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_cache.db"
        cache = NominatimCache(db_path=str(db_file))

        results = [
            {"lat": 41.3038, "lon": -81.7505, "display_name": "Aukerman Park, Ohio"}
        ]
        cache.set_search("Aukerman Park", results)

        # Retrieve with different casing and whitespace
        retrieved = cache.get_search("  aukerman park  ")
        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0]["display_name"] == "Aukerman Park, Ohio"


# ============================================================================
# 2. Rate Limiter Tests
# ============================================================================

def test_rate_limiter_min_interval():
    """Verify rate limiter enforces at least the specified interval between requests."""
    async def _test():
        limiter = NominatimRateLimiter(min_interval_seconds=0.15)
        await limiter.acquire()
        t1 = time.monotonic()
        await limiter.acquire()
        t2 = time.monotonic()
        assert t2 - t1 >= 0.13, f"Expected >= 0.13s between acquires, got {t2 - t1:.3f}s"

    asyncio.run(_test())


def test_rate_limiter_concurrent_queue():
    """Verify concurrent requests are serialized through the rate limiter."""
    async def _test():
        limiter = NominatimRateLimiter(min_interval_seconds=0.10)
        timestamps = []

        async def worker():
            await limiter.acquire()
            timestamps.append(time.monotonic())

        await asyncio.gather(worker(), worker(), worker())

        assert len(timestamps) == 3
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i - 1]
            assert diff >= 0.08, f"Expected serialization diff >= 0.08s, got {diff:.3f}s"

    asyncio.run(_test())


# ============================================================================
# 3. Grid-Cell Assignment & Enrichment Tests
# ============================================================================

def test_grid_cell_clustering_and_enrichment():
    """
    Verify 100m grid cell assignment:
    - Multiple buildings close to each other (<100m) are assigned to the same cell.
    - Reverse geocode is executed only once per cell (via cache or mocked call).
    - All buildings receive the cell's locality and street with source='area name'.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_enrich.db"
            service = NominatimService(db_path=str(db_file))

            # Pre-seed cache for expected coordinates and mock reverse_geocode
            cached_data = {
                "street": "Park Road",
                "locality": "Aukerman Residential Area",
                "suburb": "Cuyahoga",
                "city": "North Royalton",
                "postcode": "44133",
                "source": "OpenStreetMap",
            }
            service.cache.set_reverse(41.3038, -81.7505, cached_data)
            service.cache.set_reverse(41.303825, -81.750425, cached_data)
            async def _mock_rev(lat, lon):
                return cached_data
            service.reverse_geocode = _mock_rev

            # Create two buildings inside the same 100m cell (~10m apart)
            bldg1 = Polygon([
                (-81.75050, 41.30380),
                (-81.75040, 41.30380),
                (-81.75040, 41.30390),
                (-81.75050, 41.30390),
                (-81.75050, 41.30380),
            ])
            bldg2 = Polygon([
                (-81.75045, 41.30375),
                (-81.75035, 41.30375),
                (-81.75035, 41.30385),
                (-81.75045, 41.30385),
                (-81.75045, 41.30375),
            ])

            test_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(bldg1),
                        "properties": {
                            "feature_id": "bldg_001",
                            "area_sq_m": 85.0,
                            "confidence": 0.88,
                        },
                    },
                    {
                        "type": "Feature",
                        "geometry": mapping(bldg2),
                        "properties": {
                            "feature_id": "bldg_002",
                            "area_sq_m": 72.0,
                            "confidence": 0.65,
                        },
                    },
                ],
            }

            input_path = Path(tmp_dir) / "building.geojson"
            output_path = Path(tmp_dir) / "building_enriched.geojson"

            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(test_geojson, f)

            # Run enrichment without external Overpass network call
            result = await service.enrich_buildings(
                geojson_path=input_path,
                output_path=output_path,
                check_osm_overpass=False,
            )

            assert result["status"] == "completed"
            assert result["total_buildings"] == 2
            # Because both buildings are within ~15m of each other, they cluster into 1 cell
            assert result["grid_cells_geocoded"] == 1

            with open(output_path, "r", encoding="utf-8") as f:
                enriched = json.load(f)

            features = enriched["features"]
            assert len(features) == 2

            for feat in features:
                props = feat["properties"]
                assert props["street"] == "Park Road"
                assert props["locality"] == "Aukerman Residential Area"
                assert props["source_of_name"] == "area name"
                assert props["city"] == "North Royalton"
                # House number must be None for area name
                assert props["house_number"] is None

    asyncio.run(_test())


def test_iou_calculation():
    """Verify polygon IoU calculation for Overpass OSM matching."""
    poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])  # Area = 4
    poly2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2), (1, 0)])  # Area = 4, Intersect = 2, Union = 6
    iou = NominatimService.calculate_polygon_iou(poly1, poly2)
    assert abs(iou - (2.0 / 6.0)) < 1e-4

    # Identical polygons -> IoU = 1.0
    assert abs(NominatimService.calculate_polygon_iou(poly1, poly1) - 1.0) < 1e-4

    # Disjoint polygons -> IoU = 0.0
    poly3 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)])
    assert NominatimService.calculate_polygon_iou(poly1, poly3) == 0.0
