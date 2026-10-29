"""
Tile UAVid data (Images + Labels) into training-ready 256x256 patches.
Maps UAVid 8-class color labels to our 4-class scheme:
  Clutter/Human/Cars → background (0)
  Building           → building   (1)
  Road               → road       (2)
  Tree/Vegetation    → vegetation (3)
"""
import os, sys, numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

# UAVid color table (RGB) → 4-class
UAVID_REMAP = {
    (0, 0, 0):       0,   # Clutter → background
    (128, 0, 0):     1,   # Building → building
    (128, 64, 128):  2,   # Road → road
    (192, 0, 192):   0,   # Static Car → background
    (0, 128, 0):     3,   # Tree → vegetation
    (128, 128, 0):   3,   # Vegetation → vegetation
    (64, 64, 0):     0,   # Human → background
    (64, 0, 128):    0,   # Moving Car → background
}

def rgb_to_classmap(label_rgb):
    h, w = label_rgb.shape[:2]
    classmap = np.zeros((h, w), dtype=np.uint8)
    for color, cls_id in UAVID_REMAP.items():
        mask = np.all(label_rgb == color, axis=-1)
        classmap[mask] = cls_id
    return classmap

def tile_uavid(uavid_dir, out_img, out_mask, tile_size=256, overlap=0.15):
    uavid = Path(uavid_dir)
    out_img = Path(out_img)
    out_mask = Path(out_mask)
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    stride = int(tile_size * (1 - overlap))
    total = 0

    for split in ["train", "valid"]:
        split_dir = uavid / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            img_dir = seq_dir / "Images"
            lbl_dir = seq_dir / "Labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue

            for img_file in sorted(img_dir.glob("*.png")):
                lbl_file = lbl_dir / img_file.name
                if not lbl_file.exists():
                    continue

                img = np.array(Image.open(img_file).convert("RGB"))
                lbl = np.array(Image.open(lbl_file).convert("RGB"))
                classmap = rgb_to_classmap(lbl)

                h, w = img.shape[:2]
                prefix = f"uavid_{split}_{seq_dir.name}_{img_file.stem}"
                count = 0

                for y in range(0, h - tile_size + 1, stride):
                    for x in range(0, w - tile_size + 1, stride):
                        img_tile = img[y:y+tile_size, x:x+tile_size]
                        msk_tile = classmap[y:y+tile_size, x:x+tile_size]

                        valid = np.count_nonzero(img_tile.sum(axis=-1)) / (tile_size**2)
                        if valid < 0.3:
                            continue

                        fname = f"{prefix}_{y:05d}_{x:05d}.png"
                        Image.fromarray(img_tile).save(out_img / fname)
                        Image.fromarray(msk_tile).save(out_mask / fname)
                        count += 1

                logger.info(f"  {prefix}: {count} tiles")
                total += count

    logger.info(f"UAVid tiling complete: {total} tile pairs written.")
    return total

if __name__ == "__main__":
    tile_uavid("data/uavid", "data/tiles/images", "data/tiles/masks")
