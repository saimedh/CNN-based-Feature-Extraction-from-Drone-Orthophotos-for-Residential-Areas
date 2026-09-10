"""
Create a demo UAVid-style dataset from the Aukerman orthophoto.
This generates the folder structure expected by UAVidToolKit:

    data/uavid/
        train/
            seq1/
                Images/
                    000000.png
                Labels/
                    000000.png
        valid/
            seq1/
                Images/
                    000000.png
                Labels/
                    000000.png

The Labels are synthetic color masks using UAVid's 8-class color table.
"""

import os
import numpy as np
from PIL import Image

# UAVid color table (RGB)
UAVID_COLORS = {
    'Clutter':     [0, 0, 0],
    'Building':    [128, 0, 0],
    'Road':        [128, 64, 128],
    'Static_Car':  [192, 0, 192],
    'Tree':        [0, 128, 0],
    'Vegetation':  [128, 128, 0],
    'Human':       [64, 64, 0],
    'Moving_Car':  [64, 0, 128],
}

def create_synthetic_label(height, width):
    """Create a synthetic UAVid-style RGB label mask."""
    label = np.zeros((height, width, 3), dtype=np.uint8)
    colors = list(UAVID_COLORS.values())

    # Divide image into horizontal bands for different classes
    n_classes = len(colors)
    band_h = height // n_classes

    for i, clr in enumerate(colors):
        y_start = i * band_h
        y_end = (i + 1) * band_h if i < n_classes - 1 else height
        label[y_start:y_end, :] = clr

    # Add some rectangular "buildings"
    for _ in range(5):
        x = np.random.randint(0, width - 100)
        y = np.random.randint(0, height - 100)
        w = np.random.randint(50, 150)
        h = np.random.randint(50, 150)
        label[y:y+h, x:x+w] = UAVID_COLORS['Building']

    # Add some "roads"
    road_y = height // 2
    label[road_y-20:road_y+20, :] = UAVID_COLORS['Road']

    return label


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uavid_dir = os.path.join(base_dir, 'data', 'uavid')

    # Load source image (Aukerman orthophoto)
    src_path = os.path.join(base_dir, 'data', 'raw', 'aukerman.png')
    if not os.path.exists(src_path):
        # Try the TIF version
        src_path = os.path.join(base_dir, 'data', 'raw', 'odm_aukerman_orthophoto.tif')

    print(f"Loading source image: {src_path}")
    src_img = Image.open(src_path).convert('RGB')

    # Resize to a reasonable UAVid-like resolution (e.g., 1024x768)
    target_w, target_h = 1024, 768
    src_resized = src_img.resize((target_w, target_h), Image.LANCZOS)

    splits = ['train', 'valid']
    for split in splits:
        for seq_idx in range(1, 3):  # 2 sequences per split
            seq_name = f'seq{seq_idx}'
            img_dir = os.path.join(uavid_dir, split, seq_name, 'Images')
            lbl_dir = os.path.join(uavid_dir, split, seq_name, 'Labels')

            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            # Create 2 frames per sequence
            for frame_idx in range(2):
                frame_name = f'{frame_idx:06d}.png'

                # Save image (with slight random crop variation)
                np.random.seed(seq_idx * 100 + frame_idx)
                offset_x = np.random.randint(0, 50)
                offset_y = np.random.randint(0, 50)
                arr = np.array(src_resized)
                # Slight shift for variety
                shifted = np.roll(np.roll(arr, offset_x, axis=1), offset_y, axis=0)
                img = Image.fromarray(shifted)
                img.save(os.path.join(img_dir, frame_name))
                print(f"  Created: {split}/{seq_name}/Images/{frame_name}")

                # Create synthetic label
                label = create_synthetic_label(target_h, target_w)
                label_img = Image.fromarray(label)
                label_img.save(os.path.join(lbl_dir, frame_name))
                print(f"  Created: {split}/{seq_name}/Labels/{frame_name}")

    print(f"\n✓ Demo UAVid dataset created at: {uavid_dir}")
    print(f"  Structure:")
    for split in splits:
        print(f"    {split}/")
        for seq_idx in range(1, 3):
            print(f"      seq{seq_idx}/")
            print(f"        Images/ (2 frames)")
            print(f"        Labels/ (2 frames)")


if __name__ == '__main__':
    main()
