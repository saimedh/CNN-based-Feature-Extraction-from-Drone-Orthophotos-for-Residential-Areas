"""
tests/test_potsdam_adapter.py
------------------------------
Unit tests for the ISPRS Potsdam dataset adapter.

Covers:
  - RGB label → class-index conversion (4-class and 6-class)
  - File discovery from directory structure
  - Tiling + nodata filtering
  - Dataset integrity checker
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.potsdam_adapter import (
    POTSDAM_COLOR_TABLE,
    REMAP_4CLASS_MAP,
    NATIVE_6CLASS_MAP,
    rgb_label_to_classmap,
    discover_pairs,
    tile_potsdam,
    check_potsdam_dataset,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def potsdam_raw_dir(tmp_path: Path) -> Path:
    """Create a mini Potsdam-style raw directory with 2 tile pairs."""
    raw = tmp_path / "potsdam" / "raw"
    raw.mkdir(parents=True)

    size = (128, 128)
    for tid in ["2_10", "2_11"]:
        # Image
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        Image.fromarray(img).save(raw / f"top_potsdam_{tid}_RGB.png")

        # Label (building block in centre, low-veg elsewhere)
        lbl = np.zeros((*size, 3), dtype=np.uint8)
        lbl[:, :] = POTSDAM_COLOR_TABLE["low_vegetation"]
        lbl[30:80, 30:80] = POTSDAM_COLOR_TABLE["building"]
        lbl[90:110, :] = POTSDAM_COLOR_TABLE["impervious_surface"]
        Image.fromarray(lbl).save(raw / f"top_potsdam_{tid}_label.png")

    return raw


# ── rgb_label_to_classmap ──────────────────────────────────────────

class TestRgbLabelToClassmap:
    def test_4class_building_mapped_to_1(self):
        """Building pixels → class 1 in 4-class mode."""
        lbl = np.zeros((10, 10, 3), dtype=np.uint8)
        lbl[:, :] = POTSDAM_COLOR_TABLE["building"]
        classmap = rgb_label_to_classmap(lbl, remap="4class")
        assert np.all(classmap == 1)

    def test_4class_impervious_mapped_to_road(self):
        """Impervious surfaces → class 2 (road) in 4-class mode."""
        lbl = np.zeros((10, 10, 3), dtype=np.uint8)
        lbl[:, :] = POTSDAM_COLOR_TABLE["impervious_surface"]
        classmap = rgb_label_to_classmap(lbl, remap="4class")
        assert np.all(classmap == 2)

    def test_4class_tree_and_lowveg_merged(self):
        """Tree and low-veg both → class 3 (vegetation)."""
        lbl = np.zeros((10, 20, 3), dtype=np.uint8)
        lbl[:, :10] = POTSDAM_COLOR_TABLE["tree"]
        lbl[:, 10:] = POTSDAM_COLOR_TABLE["low_vegetation"]
        classmap = rgb_label_to_classmap(lbl, remap="4class")
        assert np.all(classmap == 3)

    def test_4class_car_mapped_to_background(self):
        """Car → class 0 (background) in 4-class mode."""
        lbl = np.zeros((10, 10, 3), dtype=np.uint8)
        lbl[:, :] = POTSDAM_COLOR_TABLE["car"]
        classmap = rgb_label_to_classmap(lbl, remap="4class")
        assert np.all(classmap == 0)

    def test_6class_preserves_all_classes(self):
        """In 6-class mode, each Potsdam class gets a distinct index."""
        seen = set()
        for cls_name, color in POTSDAM_COLOR_TABLE.items():
            lbl = np.zeros((4, 4, 3), dtype=np.uint8)
            lbl[:, :] = color
            classmap = rgb_label_to_classmap(lbl, remap="6class")
            idx = classmap[0, 0]
            seen.add(int(idx))
        assert len(seen) == 6, f"Expected 6 distinct classes, got {seen}"

    def test_output_dtype_is_int64(self):
        lbl = np.zeros((8, 8, 3), dtype=np.uint8)
        classmap = rgb_label_to_classmap(lbl, remap="4class")
        assert classmap.dtype == np.int64

    def test_output_shape_matches_input(self):
        lbl = np.random.randint(0, 256, (50, 70, 3), dtype=np.uint8)
        classmap = rgb_label_to_classmap(lbl, remap="4class")
        assert classmap.shape == (50, 70)


# ── discover_pairs ─────────────────────────────────────────────────

class TestDiscoverPairs:
    def test_finds_all_pairs(self, potsdam_raw_dir: Path):
        pairs = discover_pairs(potsdam_raw_dir)
        assert len(pairs) == 2

    def test_returns_path_tuples(self, potsdam_raw_dir: Path):
        pairs = discover_pairs(potsdam_raw_dir)
        for img_path, lbl_path in pairs:
            assert isinstance(img_path, Path)
            assert isinstance(lbl_path, Path)
            assert img_path.exists()
            assert lbl_path.exists()

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        pairs = discover_pairs(empty)
        assert pairs == []

    def test_unmatched_files_not_paired(self, potsdam_raw_dir: Path):
        """If a label exists without a matching image, it should be skipped."""
        # Add an orphan label
        orphan = potsdam_raw_dir / "top_potsdam_9_99_label.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(orphan)

        pairs = discover_pairs(potsdam_raw_dir)
        assert len(pairs) == 2  # orphan not paired


# ── tile_potsdam ───────────────────────────────────────────────────

class TestTilePotsdam:
    def test_produces_tile_files(self, potsdam_raw_dir: Path, tmp_path: Path):
        out_img = tmp_path / "tiles" / "images"
        out_mask = tmp_path / "tiles" / "masks"
        count = tile_potsdam(
            potsdam_dir=potsdam_raw_dir,
            out_img_dir=out_img,
            out_mask_dir=out_mask,
            tile_size=64,
            overlap=0.0,
            min_valid_ratio=0.0,
        )
        assert count > 0
        assert len(list(out_img.glob("potsdam_*.png"))) == count
        assert len(list(out_mask.glob("potsdam_*.png"))) == count

    def test_tile_dimensions_correct(self, potsdam_raw_dir: Path, tmp_path: Path):
        out_img = tmp_path / "tiles" / "images"
        out_mask = tmp_path / "tiles" / "masks"
        tile_sz = 32
        tile_potsdam(
            potsdam_dir=potsdam_raw_dir,
            out_img_dir=out_img,
            out_mask_dir=out_mask,
            tile_size=tile_sz,
            overlap=0.0,
            min_valid_ratio=0.0,
        )
        for fp in out_img.glob("*.png"):
            img = np.array(Image.open(fp))
            assert img.shape[:2] == (tile_sz, tile_sz)

    def test_mask_values_within_range(self, potsdam_raw_dir: Path, tmp_path: Path):
        out_img = tmp_path / "tiles" / "images"
        out_mask = tmp_path / "tiles" / "masks"
        tile_potsdam(
            potsdam_dir=potsdam_raw_dir,
            out_img_dir=out_img,
            out_mask_dir=out_mask,
            tile_size=64,
            overlap=0.0,
            remap="4class",
        )
        for fp in out_mask.glob("*.png"):
            msk = np.array(Image.open(fp))
            assert msk.max() < 4, f"Mask value {msk.max()} exceeds 4-class range"

    def test_empty_dir_returns_zero(self, tmp_path: Path):
        empty = tmp_path / "empty_raw"
        empty.mkdir()
        count = tile_potsdam(
            potsdam_dir=empty,
            out_img_dir=tmp_path / "o1",
            out_mask_dir=tmp_path / "o2",
        )
        assert count == 0


# ── check_potsdam_dataset ──────────────────────────────────────────

class TestCheckPotsdamDataset:
    def test_reports_correct_counts(self, potsdam_raw_dir: Path, tmp_path: Path):
        out_img = tmp_path / "tiles" / "images"
        out_mask = tmp_path / "tiles" / "masks"
        count = tile_potsdam(
            potsdam_dir=potsdam_raw_dir,
            out_img_dir=out_img,
            out_mask_dir=out_mask,
            tile_size=64,
            overlap=0.0,
            min_valid_ratio=0.0,
        )
        report = check_potsdam_dataset(out_img, out_mask, num_classes=4)
        assert report["paired_tiles"] == count
        assert report["orphan_images"] == 0
        assert report["orphan_masks"] == 0

    def test_detects_orphan_images(self, tmp_path: Path):
        img_d = tmp_path / "img"
        msk_d = tmp_path / "msk"
        img_d.mkdir()
        msk_d.mkdir()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            img_d / "potsdam_2_10_00000_00000.png"
        )
        report = check_potsdam_dataset(img_d, msk_d, num_classes=4)
        assert report["orphan_images"] == 1
        assert report["paired_tiles"] == 0

    def test_class_distribution_sums_to_one(self, potsdam_raw_dir: Path, tmp_path: Path):
        out_img = tmp_path / "tiles" / "images"
        out_mask = tmp_path / "tiles" / "masks"
        tile_potsdam(
            potsdam_dir=potsdam_raw_dir,
            out_img_dir=out_img,
            out_mask_dir=out_mask,
            tile_size=64,
            overlap=0.0,
            min_valid_ratio=0.0,
        )
        report = check_potsdam_dataset(out_img, out_mask, num_classes=4)
        total = sum(report["class_pixel_fractions"])
        assert abs(total - 1.0) < 0.01, f"Fractions sum to {total}, expected ~1.0"
