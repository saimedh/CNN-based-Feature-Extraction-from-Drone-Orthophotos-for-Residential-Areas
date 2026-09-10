"""
tests/test_postprocessing.py
------------------------------
Unit tests for mask binarization, morphological cleanup, and polygon
filtering in postprocessing/vectorize.py.
"""

import numpy as np
import pytest
from postprocessing.vectorize import binarize, morphological_cleanup


class TestBinarize:
    def test_above_threshold_is_one(self):
        prob = np.array([[0.1, 0.5, 0.8], [0.49, 0.51, 0.99]])
        result = binarize(prob, threshold=0.5)
        expected = np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_all_zeros_when_threshold_above_max(self):
        prob = np.full((10, 10), 0.3, dtype=np.float32)
        result = binarize(prob, threshold=0.9)
        assert result.sum() == 0

    def test_all_ones_when_threshold_below_min(self):
        prob = np.full((10, 10), 0.9, dtype=np.float32)
        result = binarize(prob, threshold=0.1)
        assert result.sum() == 100

    def test_output_dtype_is_uint8(self):
        prob = np.random.rand(8, 8).astype(np.float32)
        result = binarize(prob)
        assert result.dtype == np.uint8


class TestMorphologicalCleanup:
    def test_output_shape_unchanged(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        result = morphological_cleanup(mask, kernel_size=3, iterations=1)
        assert result.shape == mask.shape

    def test_isolated_single_pixel_removed(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[32, 32] = 1  # isolated single pixel
        result = morphological_cleanup(mask, kernel_size=3, iterations=2)
        assert result.sum() == 0, "Single pixel should be removed by opening"

    def test_large_blob_preserved(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 1  # 40×40 foreground square
        result = morphological_cleanup(mask, kernel_size=3, iterations=1)
        assert result.sum() > 0, "Large foreground region should survive cleanup"

    def test_output_is_binary(self):
        mask = np.random.randint(0, 2, (32, 32), dtype=np.uint8)
        result = morphological_cleanup(mask)
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 1}), f"Expected binary output, got {unique_vals}"


class TestRegularizePolygon:
    def test_regularize_squaring(self):
        from shapely.geometry import Polygon
        from postprocessing.vectorize import regularize_polygon

        # Create a slightly wobbly near-rectangular polygon (e.g. rotated building with jitter)
        coords = [(0, 0), (10.2, 0.1), (9.9, 5.2), (-0.1, 4.9), (0, 0)]
        poly = Polygon(coords)
        squared = regularize_polygon(poly, tolerance=0.5)

        assert squared.is_valid
        assert not squared.is_empty
        # Minimum bounding rectangle should have 5 coordinates (4 vertices + closing)
        assert len(squared.exterior.coords) == 5

    def test_regularize_non_rectangular_keeps_general_shape(self):
        from shapely.geometry import Polygon
        from postprocessing.vectorize import regularize_polygon

        # L-shaped building (fill ratio is lower than 0.70 of its MBR)
        coords = [(0, 0), (10, 0), (10, 3), (3, 3), (3, 10), (0, 10), (0, 0)]
        poly = Polygon(coords)
        simplified = regularize_polygon(poly, tolerance=0.5)

        assert simplified.is_valid
        assert not simplified.is_empty
        # Should preserve L-shape rather than forcing full bounding box
        assert simplified.area < poly.minimum_rotated_rectangle.area


class TestSolarRooftopIntelligence:
    def test_solar_attributes_generated_for_building(self):
        from rasterio.transform import Affine
        from rasterio.crs import CRS
        from postprocessing.vectorize import raster_to_polygons

        # 100x100 mask with 50x50 building block
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:70, 20:70] = 1
        transform = Affine.translation(0, 0) @ Affine.scale(1.0, -1.0)
        crs = CRS.from_epsg(32643)

        features = raster_to_polygons(
            binary_mask=mask,
            transform=transform,
            crs=crs,
            min_area_sq_m=10.0,
            class_name="building",
            class_id=1,
            regularize_buildings=True,
        )

        assert len(features) >= 1
        props = features[0]["properties"]
        assert "solar_usable_m2" in props
        assert "solar_kwp" in props
        assert "annual_kwh" in props
        assert "co2_offset_tons_yr" in props

        assert props["solar_usable_m2"] > 0
        assert props["solar_kwp"] > 0
        assert props["annual_kwh"] > 0
        assert props["co2_offset_tons_yr"] > 0

    def test_solar_attributes_not_added_for_roads(self):
        from rasterio.transform import Affine
        from rasterio.crs import CRS
        from postprocessing.vectorize import raster_to_polygons

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 45:55] = 1
        transform = Affine.translation(0, 0) @ Affine.scale(1.0, -1.0)
        crs = CRS.from_epsg(32643)

        features = raster_to_polygons(
            binary_mask=mask,
            transform=transform,
            crs=crs,
            min_area_sq_m=10.0,
            class_name="road",
            class_id=2,
            regularize_buildings=False,
        )

        assert len(features) >= 1
        props = features[0]["properties"]
        assert "solar_usable_m2" not in props
        assert "solar_kwp" not in props


class TestMergeTilePredictions:
    def test_merge_with_edge_tiles(self):
        from postprocessing.vectorize import merge_tile_predictions

        # Tile size 256x256, but edge tile has height=100, width=80
        tile_results = [
            {
                "probs": np.ones((4, 256, 256), dtype=np.float32) * 0.8,
                "row_off": 0,
                "col_off": 0,
                "height": 256,
                "width": 256,
            },
            {
                "probs": np.ones((4, 256, 256), dtype=np.float32) * 0.6,
                "row_off": 0,
                "col_off": 200,
                "height": 256,
                "width": 80,
            },
        ]
        merged = merge_tile_predictions(tile_results, strategy="average")
        assert 0 in merged
        assert merged[0].shape == (256, 280)
        # In non-overlapping first tile
        assert np.isclose(merged[0][0, 50], 0.8)
        # In overlap region (col 200 to 255)
        assert np.isclose(merged[0][0, 220], (0.8 + 0.6) / 2)
        # In edge-only region (col 256 to 279)
        assert np.isclose(merged[0][0, 270], 0.6)

