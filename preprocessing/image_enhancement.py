"""
preprocessing/image_enhancement.py
----------------------------------
Image processing and radiometric enhancement pipeline for drone orthophotos.

Specialized for aerial and UAV imagery:
  1. Radiometric Percentile Contrast Stretching (2%–98% percentile)
  2. Shadow Mitigation / Reduction (boosts deep building & tree shadow detail)
  3. CLAHE (Contrast Limited Adaptive Histogram Equalization in LAB space)
  4. Bilateral Edge-Preserving Denoising (smooths sensor noise while preserving rooftop curbs)
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from utils.config import load_config, _Config
from utils.logger import get_logger

logger = get_logger(__name__)


def apply_radiometric_contrast_stretch(
    img: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """
    Apply percentile-based radiometric contrast stretching per channel.

    Parameters
    ----------
    img : np.ndarray (H, W, C) uint8 or float/uint16
    lower_percentile : float (default 2.0)
    upper_percentile : float (default 98.0)

    Returns
    -------
    np.ndarray uint8 (H, W, C)
    """
    was_2d = (img.ndim == 2)
    if was_2d:
        img = np.expand_dims(img, axis=-1)

    stretched = np.zeros_like(img, dtype=np.uint8)
    channels = img.shape[-1]

    for c in range(channels):
        band = img[:, :, c].astype(np.float32)
        lo = np.percentile(band, lower_percentile)
        hi = np.percentile(band, upper_percentile)
        if hi > lo:
            scaled = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
            stretched[:, :, c] = scaled.astype(np.uint8)
        else:
            stretched[:, :, c] = np.clip(band, 0, 255).astype(np.uint8)

    if was_2d:
        return stretched.squeeze(axis=-1)
    return stretched


def apply_shadow_reduction(
    img_rgb: np.ndarray,
    shadow_threshold: int = 45,
    boost_factor: float = 1.35,
) -> np.ndarray:
    """
    Selectively boost luminance in deep shadow areas (e.g. narrow alleys,
    north-facing building eaves, tree shadows) without overexposing rooftops.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3) uint8 RGB
    shadow_threshold : int (Luminance below which is considered shadow)
    boost_factor : float (Multiplicative boost applied smoothly to shadow regions)

    Returns
    -------
    np.ndarray (H, W, 3) uint8 RGB
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return img_rgb

    # Work in LAB color space so chromaticity is preserved
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Smooth shadow mask using Gaussian blur to prevent harsh contour boundaries
    l_float = l_channel.astype(np.float32)
    # Mask is 1.0 in dark regions, decaying towards 0 near shadow_threshold * 1.5
    thresh_hi = shadow_threshold * 1.5
    shadow_weight = np.clip((thresh_hi - l_float) / (thresh_hi - shadow_threshold + 1e-5), 0.0, 1.0)
    shadow_weight = cv2.GaussianBlur(shadow_weight, (9, 9), 0)

    # Apply adaptive gamma/boost curve
    boosted_l = np.clip(l_float * (1.0 + (boost_factor - 1.0) * shadow_weight), 0, 255).astype(np.uint8)

    merged = cv2.merge([boosted_l, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_clahe(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the
    Luminance channel of an RGB orthophoto.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3) uint8 RGB
    clip_limit : float threshold for contrast limiting
    tile_grid_size : Tuple[int, int] grid for histogram equalization

    Returns
    -------
    np.ndarray (H, W, 3) uint8 RGB
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        # Grayscale fallback
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img_rgb)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    merged = cv2.merge([cl, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_bilateral_denoise(
    img_rgb: np.ndarray,
    d: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Apply edge-preserving bilateral filter to eliminate UAV camera sensor noise
    while preserving building boundaries and road markings.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3) or (H, W) uint8
    d : int diameter of each pixel neighborhood
    sigma_color : float filter sigma in color space
    sigma_space : float filter sigma in coordinate space

    Returns
    -------
    np.ndarray uint8 of same shape
    """
    return cv2.bilateralFilter(img_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def enhance_orthophoto_image(
    img: np.ndarray,
    cfg: Optional[_Config] = None,
    enable_stretch: bool = True,
    enable_shadow_reduction: bool = True,
    enable_clahe: bool = True,
    enable_denoise: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: Tuple[int, int] = (8, 8),
    bilateral_d: int = 5,
    shadow_threshold: int = 45,
    shadow_boost: float = 1.35,
) -> np.ndarray:
    """
    Execute full image processing & enhancement pipeline on an orthophoto patch or tile.

    Parameters
    ----------
    img : np.ndarray (H, W, 3) uint8 (RGB)
    cfg : Optional loaded config to override parameters
    """
    if cfg is not None and hasattr(cfg, "image_processing"):
        ip = cfg.image_processing
        if hasattr(ip, "enabled") and not ip.enabled:
            return img

        if hasattr(ip, "contrast_stretch"):
            enable_stretch = getattr(ip.contrast_stretch, "enabled", enable_stretch)
        if hasattr(ip, "shadow_reduction"):
            enable_shadow_reduction = getattr(ip.shadow_reduction, "enabled", enable_shadow_reduction)
            shadow_threshold = getattr(ip.shadow_reduction, "shadow_threshold", shadow_threshold)
            shadow_boost = getattr(ip.shadow_reduction, "boost_factor", shadow_boost)
        if hasattr(ip, "clahe"):
            enable_clahe = getattr(ip.clahe, "enabled", enable_clahe)
            clahe_clip_limit = getattr(ip.clahe, "clip_limit", clahe_clip_limit)
            grid = getattr(ip.clahe, "tile_grid_size", None)
            if grid and len(grid) == 2:
                clahe_grid_size = (int(grid[0]), int(grid[1]))
        if hasattr(ip, "bilateral_filter"):
            enable_denoise = getattr(ip.bilateral_filter, "enabled", enable_denoise)
            bilateral_d = getattr(ip.bilateral_filter, "d", bilateral_d)

    # 1. Radiometric percentile contrast stretch
    out = img
    if enable_stretch:
        out = apply_radiometric_contrast_stretch(out)

    # 2. Shadow recovery in deep urban / tree shadows
    if enable_shadow_reduction and out.ndim == 3 and out.shape[2] == 3:
        out = apply_shadow_reduction(out, shadow_threshold=shadow_threshold, boost_factor=shadow_boost)

    # 3. CLAHE local contrast equalization
    if enable_clahe:
        out = apply_clahe(out, clip_limit=clahe_clip_limit, tile_grid_size=clahe_grid_size)

    # 4. Bilateral edge-preserving noise suppression
    if enable_denoise:
        out = apply_bilateral_denoise(out, d=bilateral_d)

    return out
