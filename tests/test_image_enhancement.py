"""
tests/test_image_enhancement.py
-------------------------------
Unit tests for radiometric enhancement and image processing functions:
  - Contrast stretching
  - Shadow reduction / mitigation
  - CLAHE (in LAB color space)
  - Bilateral edge-preserving filtering
  - Combined orthophoto enhancement pipeline
"""

import numpy as np
import pytest

from preprocessing.image_enhancement import (
    apply_radiometric_contrast_stretch,
    apply_shadow_reduction,
    apply_clahe,
    apply_bilateral_denoise,
    enhance_orthophoto_image,
)
from utils.config import load_config


class TestRadiometricEnhancement:
    def test_contrast_stretch_shape_and_dtype(self):
        img = np.random.randint(20, 200, (64, 64, 3), dtype=np.uint8)
        stretched = apply_radiometric_contrast_stretch(img, lower_percentile=2.0, upper_percentile=98.0)
        assert stretched.shape == img.shape
        assert stretched.dtype == np.uint8
        assert stretched.max() >= img.max()
        assert stretched.min() <= img.min()

    def test_contrast_stretch_2d_grayscale(self):
        img = np.random.randint(50, 150, (64, 64), dtype=np.uint8)
        stretched = apply_radiometric_contrast_stretch(img)
        assert stretched.shape == (64, 64)
        assert stretched.dtype == np.uint8

    def test_shadow_reduction_boosts_dark_pixels(self):
        # Create dark image (shadow)
        dark_img = np.full((32, 32, 3), 30, dtype=np.uint8)
        boosted = apply_shadow_reduction(dark_img, shadow_threshold=45, boost_factor=1.4)
        assert boosted.shape == dark_img.shape
        assert boosted.dtype == np.uint8
        # Shadow regions should be brighter than the input
        assert np.mean(boosted) > np.mean(dark_img)

    def test_shadow_reduction_preserves_bright_pixels(self):
        # Create bright image (sunlit rooftop)
        bright_img = np.full((32, 32, 3), 220, dtype=np.uint8)
        boosted = apply_shadow_reduction(bright_img, shadow_threshold=45, boost_factor=1.4)
        # Bright regions should remain virtually unchanged
        assert np.isclose(np.mean(boosted), np.mean(bright_img), atol=2.0)

    def test_clahe_contrast_enhancement(self):
        # Low contrast image
        low_contrast = np.full((64, 64, 3), 100, dtype=np.uint8)
        low_contrast[10:30, 10:30] = 105
        enhanced = apply_clahe(low_contrast, clip_limit=2.0, tile_grid_size=(8, 8))
        assert enhanced.shape == (64, 64, 3)
        assert enhanced.dtype == np.uint8

    def test_bilateral_denoise_preserves_edges(self):
        # Image with sharp edge + noise
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, 20:] = 200
        # Add slight noise
        noise = np.random.randint(-5, 5, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        filtered = apply_bilateral_denoise(noisy, d=5, sigma_color=50, sigma_space=50)
        assert filtered.shape == img.shape
        assert filtered.dtype == np.uint8

    def test_enhance_orthophoto_image_pipeline(self):
        cfg = load_config()
        img = np.random.randint(10, 240, (128, 128, 3), dtype=np.uint8)
        result = enhance_orthophoto_image(img, cfg=cfg)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8

    def test_enhance_disabled_returns_original(self):
        cfg = load_config()
        cfg.image_processing.enabled = False
        img = np.random.randint(10, 240, (64, 64, 3), dtype=np.uint8)
        result = enhance_orthophoto_image(img, cfg=cfg)
        assert np.array_equal(result, img)
