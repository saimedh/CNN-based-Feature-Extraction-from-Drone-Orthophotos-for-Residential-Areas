"""
scripts/create_potsdam_demo.py
-------------------------------
Generate a small synthetic ISPRS-Potsdam-style dataset for testing and
demonstration purposes.  Mirrors the real Potsdam directory layout:

    data/potsdam/raw/
        top_potsdam_2_10_RGB.png
        top_potsdam_2_10_label.png
        top_potsdam_2_11_RGB.png
        top_potsdam_2_11_label.png
        ...

Each "image" is resized from the Aukerman orthophoto (or generated if absent).
Each "label" is a synthetic RGB mask using the official ISPRS Potsdam colour
table:
    Impervious surfaces → (255, 255, 255)
    Building            → (  0,   0, 255)
    Low vegetation      → (  0, 255, 255)
    Tree                → (  0, 255,   0)
    Car                 → (255, 255,   0)
    Clutter             → (255,   0,   0)

Usage
-----
    python scripts/create_potsdam_demo.py
"""

import os
import sys
import numpy as np
from PIL import Image

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ISPRS Potsdam colour table (RGB)
POTSDAM_COLORS = {
    "impervious_surface": (255, 255, 255),
    "building":           (  0,   0, 255),
    "low_vegetation":     (  0, 255, 255),
    "tree":               (  0, 255,   0),
    "car":                (255, 255,   0),
    "clutter":            (255,   0,   0),
}


def create_synthetic_potsdam_label(height: int, width: int, seed: int = 0) -> np.ndarray:
    """
    Create a synthetic ISPRS-Potsdam-style RGB label mask.

    The mask is divided into regions of the 6 Potsdam classes, with
    random rectangular buildings and road strips overlaid for realism.
    """
    rng = np.random.RandomState(seed)
    label = np.zeros((height, width, 3), dtype=np.uint8)

    # Background: fill with low-vegetation
    label[:, :] = POTSDAM_COLORS["low_vegetation"]

    # Top third: impervious surface (roads / pavements)
    label[: height // 3, :] = POTSDAM_COLORS["impervious_surface"]

    # Scattered tree regions
    for _ in range(8):
        x = rng.randint(0, max(1, width - 120))
        y = rng.randint(height // 3, max(height // 3 + 1, height - 120))
        w = rng.randint(60, 160)
        h = rng.randint(60, 160)
        label[y : y + h, x : x + w] = POTSDAM_COLORS["tree"]

    # Rectangular buildings
    for _ in range(10):
        x = rng.randint(0, max(1, width - 100))
        y = rng.randint(0, max(1, height - 100))
        w = rng.randint(40, 140)
        h = rng.randint(40, 140)
        label[y : y + h, x : x + w] = POTSDAM_COLORS["building"]

    # Road strips (horizontal + vertical)
    road_y = height // 2
    label[road_y - 15 : road_y + 15, :] = POTSDAM_COLORS["impervious_surface"]
    road_x = width // 2
    label[:, road_x - 15 : road_x + 15] = POTSDAM_COLORS["impervious_surface"]

    # Cars on road
    for _ in range(5):
        cx = rng.randint(0, max(1, width - 30))
        cy = rng.choice([road_y - 10, road_y + 5])
        label[cy : cy + 8, cx : cx + 20] = POTSDAM_COLORS["car"]

    # Clutter patches
    for _ in range(3):
        x = rng.randint(0, max(1, width - 40))
        y = rng.randint(0, max(1, height - 40))
        label[y : y + 30, x : x + 30] = POTSDAM_COLORS["clutter"]

    return label


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "potsdam", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Try to use the Aukerman orthophoto as source imagery
    src_candidates = [
        os.path.join(base_dir, "data", "raw", "aukerman.png"),
        os.path.join(base_dir, "data", "raw", "odm_aukerman_orthophoto.tif"),
    ]
    src_img = None
    for cand in src_candidates:
        if os.path.exists(cand):
            print(f"Loading source image: {cand}")
            src_img = Image.open(cand).convert("RGB")
            break

    # Potsdam patches are 6000×6000 — we use 512×512 for demo
    target_size = (512, 512)

    # Generate 4 demo Potsdam "patches" with typical naming
    tile_ids = ["2_10", "2_11", "3_10", "3_11"]

    for i, tid in enumerate(tile_ids):
        img_name = f"top_potsdam_{tid}_RGB.png"
        lbl_name = f"top_potsdam_{tid}_label.png"

        # Image
        if src_img is not None:
            # Crop slightly different regions for variety
            rng = np.random.RandomState(42 + i)
            ox = rng.randint(0, max(1, src_img.width - target_size[0]))
            oy = rng.randint(0, max(1, src_img.height - target_size[1]))
            patch = src_img.crop((ox, oy, ox + target_size[0], oy + target_size[1]))
            patch = patch.resize(target_size, Image.LANCZOS)
        else:
            # Generate random colour noise as fallback
            arr = np.random.RandomState(i).randint(0, 256, (*target_size, 3), dtype=np.uint8)
            patch = Image.fromarray(arr)

        patch.save(os.path.join(raw_dir, img_name))
        print(f"  Created: {img_name}  ({target_size[0]}×{target_size[1]})")

        # Label
        lbl = create_synthetic_potsdam_label(*target_size, seed=100 + i)
        Image.fromarray(lbl).save(os.path.join(raw_dir, lbl_name))
        print(f"  Created: {lbl_name}")

    print(f"\n✓ Demo Potsdam dataset created at: {raw_dir}")
    print(f"  Patches: {len(tile_ids)}")
    print(f"  Size   : {target_size[0]}×{target_size[1]} pixels")
    print()
    print("Next steps:")
    print("  1. Tile:   python preprocessing/potsdam_adapter.py tile --potsdam-dir data/potsdam/raw")
    print("  2. Check:  python preprocessing/potsdam_adapter.py check")
    print("  3. Train:  python training/train.py --config config/config_potsdam.yaml")


if __name__ == "__main__":
    main()
