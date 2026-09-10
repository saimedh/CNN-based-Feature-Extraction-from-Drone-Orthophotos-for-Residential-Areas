"""
tests/test_tiling.py
--------------------
Tests for single-file and folder-wide tiling and inference helper routines.
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from preprocessing.tiling import tile_folder
from inference.predict import predict_folder
from utils.config import load_config


def _make_dummy_geotiff(path: Path, width: int = 300, height: int = 300, count: int = 3, dtype=np.uint8):
    transform = from_origin(100.0, 50.0, 0.1, 0.1)
    data = np.ones((count, height, width), dtype=dtype) * 128
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_tile_folder(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    tiles_dir = tmp_path / "tiles"
    masks_dir = tmp_path / "masks"

    # Create two orthophotos, one with mask
    ortho1 = raw_dir / "ortho1.tif"
    _make_dummy_geotiff(ortho1, 300, 300, count=3)
    mask1 = raw_dir / "ortho1_mask.tif"
    _make_dummy_geotiff(mask1, 300, 300, count=1)

    ortho2 = raw_dir / "ortho2.tif"
    _make_dummy_geotiff(ortho2, 300, 300, count=3)

    cfg = load_config()
    tiles = tile_folder(raw_dir, output_img_dir=tiles_dir, output_mask_dir=masks_dir, cfg=cfg)

    assert len(tiles) > 0
    assert tiles_dir.exists()
    produced_tiles = list(tiles_dir.glob("*.tif"))
    assert len(produced_tiles) == len(tiles)


def test_predict_folder_non_directory(tmp_path):
    non_dir = tmp_path / "not_a_dir.txt"
    non_dir.write_text("hello")
    with pytest.raises(NotADirectoryError):
        predict_folder(folder_path=non_dir, checkpoint_path="checkpoints/best.pt")
