"""
scripts/download_uavid.py
--------------------------
Download the full UAVid 2020 semantic segmentation dataset and prepare it
for use with this project's training pipeline.

UAVid 2020 contains:
  - 42 video sequences (20 train, 7 valid, 15 test)
  - 4K resolution frames with pixel-level semantic labels
  - 8 classes: Clutter, Building, Road, Static Car, Tree, Low Vegetation,
               Human, Moving Car

This script:
  1.  Downloads the "images only" archive from EOStore (~3 GB)
  2.  Extracts the archive into  data/uavid/
  3.  Renames subdirectories to the expected  train/ valid/ test/ layout
  4.  Converts UAVid's 8-class RGB colour labels  ->  4-class single-band
      TrainId masks that match the project config (0=background, 1=building,
      2=road, 3=vegetation)
  5.  Tiles the images and masks using preprocessing/tiling.py (optional)

Usage
-----
    python scripts/download_uavid.py                   # download + extract + convert
    python scripts/download_uavid.py --skip-download    # if already downloaded manually
    python scripts/download_uavid.py --tile             # also run tiling after conversion

Manual download (if automated download fails)
----------------------------------------------
Go to: https://eostore.itc.utwente.nl:5001/sharing/2Fu3x8Grb
Password: uavid2020
Download the ZIP, place it in  data/  as  data/uavid_images.zip
Then run:  python scripts/download_uavid.py --skip-download

License: CC BY-NC-SA 4.0
Citation:
    @article{LYU2020108,
        author  = {Ye Lyu and George Vosselman and Gui-Song Xia and Alper Yilmaz and Michael Ying Yang},
        title   = {UAVid: A semantic segmentation dataset for UAV imagery},
        journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
        volume  = {165},
        pages   = {108--119},
        year    = {2020},
    }
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UAVID_DIR = DATA_DIR / "uavid"
ZIP_PATH = DATA_DIR / "uavid_images.zip"

# Official EOStore download link (images-only version)
EOSTORE_URL = "https://eostore.itc.utwente.nl:5001/sharing/2Fu3x8Grb"
EOSTORE_PASSWORD = "uavid2020"

# UAVid 8-class colour table  ->  RGB tuples
UAVID_COLOR_MAP = {
    (0,   0,   0):   "Clutter",        # 0
    (128, 0,   0):   "Building",       # 1
    (128, 64,  128): "Road",           # 2
    (192, 0,   192): "Static_Car",     # 3
    (0,   128, 0):   "Tree",           # 4
    (128, 128, 0):   "Low_Vegetation", # 5
    (64,  64,  0):   "Human",          # 6
    (64,  0,   128): "Moving_Car",     # 7
}

# Mapping: UAVid 8-class  ->  Project 4-class
# Project classes: 0=background, 1=building, 2=road, 3=vegetation
UAVID_TO_PROJECT = {
    "Clutter":        0,  # -> background
    "Building":       1,  # -> building
    "Road":           2,  # -> road
    "Static_Car":     0,  # -> background
    "Tree":           3,  # -> vegetation
    "Low_Vegetation": 3,  # -> vegetation
    "Human":          0,  # -> background
    "Moving_Car":     0,  # -> background
}

# Build a direct RGB -> class_id lookup (for vectorised conversion)
_RGB_TO_CLASS = {}
for rgb, name in UAVID_COLOR_MAP.items():
    _RGB_TO_CLASS[rgb] = UAVID_TO_PROJECT[name]


# --------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------

def download_from_eostore(dest_path: Path) -> bool:
    """
    Attempt to download the UAVid images-only archive from EOStore.
    Returns True on success, False if manual download is required.
    """
    print("=" * 65)
    print(" UAVid Dataset Download")
    print("=" * 65)

    # Try using requests (handles redirects, but EOStore may need cookies)
    try:
        import requests
        print(f"\nAttempting download from EOStore ...")
        print(f"  URL:  {EOSTORE_URL}")
        print(f"  Dest: {dest_path}")

        session = requests.Session()

        # EOStore Synology sharing page — try direct download
        # The sharing link usually resolves to a page where you enter the password
        # and then get a download link.  Automated download may not work for all
        # Synology configs, so we provide a fallback.
        # EOStore often uses a self-signed certificate
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = session.get(EOSTORE_URL, allow_redirects=True, timeout=30, verify=False)

        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
            print("\n[!] EOStore returned an HTML page (password-protected share).")
            print("    Automated download is not supported for this share type.")
            _print_manual_instructions(dest_path)
            return False

        # If we somehow got a direct ZIP
        if resp.status_code == 200:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"\n[OK] Download complete: {dest_path}  ({dest_path.stat().st_size / 1e9:.1f} GB)")
            return True

        print(f"\n[!] Unexpected status code: {resp.status_code}")
        _print_manual_instructions(dest_path)
        return False

    except ImportError:
        print("\n[!] 'requests' library not installed.")
        _print_manual_instructions(dest_path)
        return False
    except Exception as e:
        print(f"\n[!] Download error: {e}")
        _print_manual_instructions(dest_path)
        return False


def _print_manual_instructions(dest_path: Path) -> None:
    """Print step-by-step manual download instructions."""
    print()
    print("-" * 65)
    print(" MANUAL DOWNLOAD INSTRUCTIONS")
    print("-" * 65)
    print()
    print("  The UAVid dataset is hosted on EOStore (password-protected).")
    print("  Please download it manually:")
    print()
    print("  1. Open this URL in your browser:")
    print(f"     {EOSTORE_URL}")
    print()
    print(f"  2. Enter the password:  {EOSTORE_PASSWORD}")
    print()
    print("  3. Download the ZIP file(s)")
    print()
    print(f"  4. Save/move the ZIP to:")
    print(f"     {dest_path}")
    print()
    print("  5. Re-run this script with --skip-download:")
    print(f"     python scripts/download_uavid.py --skip-download")
    print()
    print("  Alternative: Download from Baidu Pan")
    print("     https://pan.baidu.com/s/18QzaqN-GGKhZxxGrxy-ESA")
    print("     Password: rart")
    print()
    print("  Alternative: Download from Dataset Ninja (Supervisely format)")
    print("     pip install dataset-tools")
    print("     python -c \"import dataset_tools as dt; dt.download('UAVid', '~/uavid/')\"")
    print("-" * 65)


# --------------------------------------------------------------
# Extract & reorganise
# --------------------------------------------------------------

def extract_and_organise(zip_path: Path, dest_dir: Path) -> None:
    """Extract the UAVid ZIP and normalise the folder structure."""
    print(f"\nExtracting {zip_path.name} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    print(f"  Extracted to: {dest_dir}")

    # The ZIP may create a nested folder like uavid_v2/ or UAVid2020/
    # Find the actual root that contains seq* folders or train/valid/test
    _normalise_structure(dest_dir)

    print("[OK] Extraction and folder normalisation complete.")


def _normalise_structure(root: Path) -> None:
    """
    Ensure the final layout is:
        root/
          train/  seq1/ seq2/ ...
          valid/  seq1/ seq2/ ...
          test/   seq1/ seq2/ ...
    """
    # Check if the desired structure already exists
    if (root / "train").is_dir() or (root / "valid").is_dir():
        print("  Structure looks correct (train/valid already present).")
        return

    # Look for a single nested directory that contains the actual data
    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(subdirs) == 1:
        nested = subdirs[0]
        print(f"  Found nested dir: {nested.name}, moving contents up ...")
        for item in nested.iterdir():
            target = root / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        # Remove the now-empty nested dir
        if nested.exists() and not any(nested.iterdir()):
            nested.rmdir()

    # Rename common variants
    rename_map = {
        "training":   "train",
        "Training":   "train",
        "validation": "valid",
        "Validation": "valid",
        "val":        "valid",
        "Val":        "valid",
        "testing":    "test",
        "Testing":    "test",
    }
    for old_name, new_name in rename_map.items():
        old_path = root / old_name
        new_path = root / new_name
        if old_path.is_dir() and not new_path.exists():
            print(f"  Renaming {old_name}/ -> {new_name}/")
            old_path.rename(new_path)


# --------------------------------------------------------------
# Label conversion: RGB colour mask -> single-band TrainId mask
# --------------------------------------------------------------

def convert_rgb_label_to_trainid(label_path: Path) -> np.ndarray:
    """
    Convert a UAVid RGB colour label PNG to a single-channel class index mask
    using the project's 4-class mapping.

    Returns
    -------
    np.ndarray of shape (H, W), dtype uint8, values in {0, 1, 2, 3}
    """
    label_img = np.array(Image.open(label_path).convert("RGB"))
    h, w = label_img.shape[:2]
    trainid = np.zeros((h, w), dtype=np.uint8)

    for rgb, class_id in _RGB_TO_CLASS.items():
        mask = np.all(label_img == np.array(rgb, dtype=np.uint8), axis=-1)
        trainid[mask] = class_id

    return trainid


def convert_all_labels(uavid_root: Path) -> int:
    """
    Walk the UAVid directory tree and create TrainId/ folders alongside
    existing Labels/ folders.

    Returns the number of label files converted.
    """
    print("\nConverting UAVid 8-class RGB labels -> 4-class TrainId masks ...")
    count = 0

    for split in ["train", "valid", "test"]:
        split_dir = uavid_root / split
        if not split_dir.is_dir():
            continue

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            labels_dir = seq_dir / "Labels"
            trainid_dir = seq_dir / "TrainId"
            trainid_dir.mkdir(exist_ok=True)

            if not labels_dir.is_dir():
                continue

            for lbl_file in sorted(labels_dir.glob("*.png")):
                trainid = convert_rgb_label_to_trainid(lbl_file)
                out_path = trainid_dir / lbl_file.name
                Image.fromarray(trainid).save(out_path)
                count += 1

            if count > 0:
                print(f"  {split}/{seq_dir.name}: {len(list(labels_dir.glob('*.png')))} labels converted")

    print(f"[OK] Converted {count} label files total.")
    return count


# --------------------------------------------------------------
# Dataset statistics
# --------------------------------------------------------------

def print_dataset_stats(uavid_root: Path) -> None:
    """Print a summary of the downloaded dataset."""
    print()
    print("=" * 65)
    print(" UAVid Dataset Summary")
    print("=" * 65)

    class_names = ["background", "building", "road", "vegetation"]
    total_images = 0
    total_labels = 0

    for split in ["train", "valid", "test"]:
        split_dir = uavid_root / split
        if not split_dir.is_dir():
            continue

        seqs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        n_imgs = 0
        n_lbls = 0
        for seq in seqs:
            imgs = list((seq / "Images").glob("*.png")) if (seq / "Images").is_dir() else []
            lbls = list((seq / "Labels").glob("*.png")) if (seq / "Labels").is_dir() else []
            n_imgs += len(imgs)
            n_lbls += len(lbls)

        total_images += n_imgs
        total_labels += n_lbls
        print(f"  {split:8s}:  {len(seqs):2d} sequences  |  {n_imgs:4d} images  |  {n_lbls:4d} labels")

    print(f"  {'TOTAL':8s}:                    |  {total_images:4d} images  |  {total_labels:4d} labels")
    print()
    print(f"  Project class mapping (UAVid 8-class -> 4-class):")
    print(f"    0 background  <- Clutter, Static Car, Human, Moving Car")
    print(f"    1 building    <- Building")
    print(f"    2 road        <- Road")
    print(f"    3 vegetation  <- Tree, Low Vegetation")
    print()
    print(f"  Data location:  {uavid_root}")
    print(f"  TrainId masks:  {uavid_root}/{{split}}/{{seq}}/TrainId/")
    print("=" * 65)


# --------------------------------------------------------------
# Optional: Tile the dataset
# --------------------------------------------------------------

def tile_dataset(uavid_root: Path) -> None:
    """
    Run tiling on UAVid images and converted TrainId masks using
    the project's preprocessing/tiling.py.
    """
    print("\nTiling UAVid images and masks ...")
    tiling_script = PROJECT_ROOT / "preprocessing" / "tiling.py"
    if not tiling_script.exists():
        print(f"  [!] Tiling script not found at {tiling_script}")
        print(f"    Skipping tiling — you can tile manually later.")
        return

    tiles_out = UAVID_DIR / "tiles"
    tiles_out.mkdir(parents=True, exist_ok=True)

    import subprocess

    for split in ["train", "valid"]:
        split_dir = uavid_root / split
        if not split_dir.is_dir():
            continue

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            images_dir = seq_dir / "Images"
            trainid_dir = seq_dir / "TrainId"

            if not images_dir.is_dir():
                continue

            for img_file in sorted(images_dir.glob("*.png")):
                mask_file = trainid_dir / img_file.name if trainid_dir.is_dir() else None
                out_prefix = f"{split}_{seq_dir.name}_{img_file.stem}"

                cmd = [
                    sys.executable,
                    str(tiling_script),
                    str(img_file),
                    "--out-img", str(tiles_out / "images"),
                    "--out-mask", str(tiles_out / "masks"),
                ]
                if mask_file and mask_file.exists():
                    cmd.extend(["--mask", str(mask_file)])

                print(f"  Tiling {split}/{seq_dir.name}/{img_file.name} ...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    [!] Tiling failed: {result.stderr[:200]}")

    print("[OK] Tiling complete.")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the full UAVid 2020 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_uavid.py                    # download + extract + convert
  python scripts/download_uavid.py --skip-download     # already downloaded manually
  python scripts/download_uavid.py --tile              # also tile after conversion
  python scripts/download_uavid.py --zip data/my.zip   # use custom ZIP path
        """,
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (use if you already have the ZIP file).",
    )
    parser.add_argument(
        "--zip", type=str, default=None,
        help=f"Path to the UAVid ZIP file. Default: {ZIP_PATH}",
    )
    parser.add_argument(
        "--dest", type=str, default=None,
        help=f"Destination directory for extracted data. Default: {UAVID_DIR}",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip label conversion step.",
    )
    parser.add_argument(
        "--tile", action="store_true",
        help="Also tile images and masks after conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    zip_path = Path(args.zip) if args.zip else ZIP_PATH
    dest_dir = Path(args.dest) if args.dest else UAVID_DIR

    # -- Step 1: Download --------------------------------------
    if not args.skip_download:
        if zip_path.exists():
            size_gb = zip_path.stat().st_size / 1e9
            print(f"ZIP file already exists: {zip_path}  ({size_gb:.1f} GB)")
            print("Skipping download.  Use --zip to specify a different file.\n")
        else:
            success = download_from_eostore(zip_path)
            if not success:
                print("\n>>> Please download the file manually and re-run with --skip-download")
                sys.exit(1)

    # -- Step 2: Extract & organise ----------------------------
    if zip_path.exists():
        # Check if already extracted
        if (dest_dir / "train").is_dir() and len(list((dest_dir / "train").iterdir())) > 2:
            print(f"\nDataset appears already extracted at {dest_dir}")
            print("Skipping extraction.\n")
        else:
            extract_and_organise(zip_path, dest_dir)
    else:
        if not (dest_dir / "train").is_dir():
            print(f"\n[X] No ZIP file found at {zip_path}")
            print(f"    and no extracted data found at {dest_dir}")
            _print_manual_instructions(zip_path)
            sys.exit(1)

    # -- Step 3: Convert labels --------------------------------
    if not args.skip_convert:
        convert_all_labels(dest_dir)

    # -- Step 4: Print stats -----------------------------------
    print_dataset_stats(dest_dir)

    # -- Step 5: Optional tiling -------------------------------
    if args.tile:
        tile_dataset(dest_dir)

    print("\n[OK] UAVid dataset preparation complete!")
    print(f"  Next steps:")
    print(f"    1. Review the data in  {dest_dir}")
    print(f"    2. Tile the images:    python scripts/download_uavid.py --skip-download --tile")
    print(f"    3. Start training:     python training/train.py")


if __name__ == "__main__":
    main()
