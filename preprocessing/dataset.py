"""
preprocessing/dataset.py
------------------------
PyTorch Dataset that loads paired image tiles and mask tiles from disk,
applies albumentations augmentations, and returns normalised tensors ready
for model training.

Class index mapping (default, configurable via YAML):
  0 → background
  1 → building
  2 → road
  3 → vegetation
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Callable

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from preprocessing.image_enhancement import enhance_orthophoto_image
from utils.config import load_config, _Config
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Standard ImageNet normalisation (applied after albumentations)
# ---------------------------------------------------------------------------
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def build_train_transforms(cfg: _Config) -> A.Compose:
    """Return the albumentations pipeline for training tiles."""
    aug = cfg.augmentation
    crop_sz = int(aug.random_crop_size)

    transforms = [
        A.RandomCrop(height=crop_sz, width=crop_sz, p=1.0),
        A.HorizontalFlip(p=0.5 if aug.horizontal_flip else 0.0),
        A.VerticalFlip(p=0.5 if aug.vertical_flip else 0.0),
        A.Rotate(limit=int(aug.rotate_limit), p=0.5),
        A.ColorJitter(
            brightness=float(aug.color_jitter.brightness),
            contrast=float(aug.color_jitter.contrast),
            saturation=float(aug.color_jitter.saturation),
            hue=float(aug.color_jitter.hue),
            p=0.5,
        ),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def build_val_transforms(cfg: _Config) -> A.Compose:
    """Return the albumentations pipeline for validation / test tiles."""
    tile_sz = int(cfg.tiling.tile_size)
    return A.Compose([
        A.Resize(height=tile_sz, width=tile_sz),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class OrthophotoDataset(Dataset):
    """
    Loads GeoTIFF image tiles and their corresponding mask tiles.

    Parameters
    ----------
    img_dir    : Directory containing image tiles (.tif).
    mask_dir   : Directory containing mask tiles (_mask.tif).
    transform  : albumentations Compose pipeline.
    tile_ext   : File extension to glob for image tiles.
    """

    def __init__(
        self,
        img_dir: str | Path,
        mask_dir: str | Path,
        transform: Optional[Callable] = None,
        tile_ext: str = ".tif",
        cfg: Optional[_Config] = None,
        enable_enhancement: bool = True,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.cfg = cfg or load_config()
        self.enable_enhancement = enable_enhancement

        # Collect all image tile paths (sorted for reproducibility)
        self.img_paths = sorted(
            list(self.img_dir.glob(f"*{tile_ext}")) +
            (list(self.img_dir.glob("*.png")) if tile_ext != ".png" else [])
        )
        if not self.img_paths:
            raise FileNotFoundError(
                f"No tile files found in '{img_dir}'. "
                "Run preprocessing/tiling.py or potsdam_adapter.py first."
            )

        # Build paired mask paths and validate
        self.mask_paths: list[Optional[Path]] = []
        for ip in self.img_paths:
            # Try: same name in mask dir (Potsdam/UAVid style)
            candidate_same = self.mask_dir / ip.name
            # Try: stem + _mask + ext (GeoTIFF style)
            candidate_mask = self.mask_dir / (ip.stem + "_mask" + ip.suffix)
            if candidate_same.exists():
                self.mask_paths.append(candidate_same)
            elif candidate_mask.exists():
                self.mask_paths.append(candidate_mask)
            else:
                self.mask_paths.append(None)

        logger.info(
            f"OrthophotoDataset: {len(self.img_paths)} tiles | "
            f"masks found: {sum(1 for m in self.mask_paths if m)} | "
            f"enhancement: {'enabled' if self.enable_enhancement else 'disabled'}"
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        dict with keys:
            ``image``  : FloatTensor [3, H, W]  (normalised)
            ``mask``   : LongTensor  [H, W]      (class indices 0..C-1)
            ``img_path``: str  (for debugging / postprocessing)
        """
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # --- Load image (support both GeoTIFF and PNG) ---
        if img_path.suffix.lower() in ('.tif', '.tiff'):
            with rasterio.open(img_path) as src:
                img = src.read()  # (C, H, W), uint8 or uint16
            if img.dtype != np.uint8:
                img = _rescale_to_uint8(img)
            img = np.transpose(img[:3], (1, 2, 0))  # (H, W, 3)
        else:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(img_path).convert("RGB"))  # (H, W, 3)

        # Apply radiometric & image enhancement (CLAHE, bilateral filter, shadow reduction)
        if self.enable_enhancement:
            img = enhance_orthophoto_image(img, cfg=self.cfg)

        # --- Load mask ---
        if mask_path and mask_path.exists():
            if mask_path.suffix.lower() in ('.tif', '.tiff'):
                with rasterio.open(mask_path) as msrc:
                    mask = msrc.read(1).astype(np.int64)  # (H, W)
            else:
                from PIL import Image as PILImage
                mask = np.array(PILImage.open(mask_path)).astype(np.int64)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]  # take first channel
        else:
            # Dummy all-background mask when no label exists
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)

        # --- Apply transforms ---
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            image_tensor = augmented["image"]         # FloatTensor [3, H, W]
            mask_arr = augmented["mask"]
            if isinstance(mask_arr, torch.Tensor):
                mask_tensor = mask_arr.long()
            else:
                mask_tensor = torch.from_numpy(mask_arr).long()
        else:
            image_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "img_path": str(img_path),
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale a multi-band array to uint8 per band using 2–98th percentile."""
    result = np.zeros_like(arr, dtype=np.uint8)
    for b in range(arr.shape[0]):
        band = arr[b].astype(np.float32)
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi > lo:
            band = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
        else:
            band = np.zeros_like(band)
        result[b] = band.astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    img_dir: Optional[str | Path] = None,
    mask_dir: Optional[str | Path] = None,
    cfg: Optional[_Config] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a flat tile directory.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    if cfg is None:
        cfg = load_config()

    if img_dir is None:
        img_dir = cfg.paths.tiles_dir
    if mask_dir is None:
        mask_dir = cfg.paths.masks_dir

    full_dataset = OrthophotoDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=None,  # transforms applied after splitting
        cfg=cfg,
    )

    n_total = len(full_dataset)
    n_test = max(1, int(n_total * cfg.dataset.test_split))
    n_val = max(1, int(n_total * cfg.dataset.val_split))
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(int(cfg.dataset.random_seed))
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Assign transforms
    train_tfm = build_train_transforms(cfg)
    val_tfm = build_val_transforms(cfg)

    train_ds.dataset.transform = train_tfm  # type: ignore[union-attr]
    # Subsets share the dataset reference — use separate wrappers for val/test
    val_ds = _TransformSubset(val_ds, val_tfm, cfg=cfg)
    test_ds = _TransformSubset(test_ds, val_tfm, cfg=cfg)

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        f"Splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader


class _TransformSubset(Dataset):
    """Wraps a random_split Subset to apply a *different* transform."""

    def __init__(
        self,
        subset: torch.utils.data.Subset,
        transform: Callable,
        cfg: Optional[_Config] = None,
        enable_enhancement: bool = True,
    ) -> None:
        self.subset = subset
        self.transform = transform
        self.cfg = cfg or load_config()
        self.enable_enhancement = enable_enhancement

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.subset.indices[idx]
        img_path = self.subset.dataset.img_paths[real_idx]
        with rasterio.open(img_path) as src:
            img = src.read()
        if img.dtype != np.uint8:
            img = _rescale_to_uint8(img)
        img = np.transpose(img[:3], (1, 2, 0))

        # Apply radiometric & image enhancement (CLAHE, bilateral filter, shadow reduction)
        if self.enable_enhancement:
            img = enhance_orthophoto_image(img, cfg=self.cfg)

        mask_path_raw = self.subset.dataset.mask_paths[real_idx]
        if mask_path_raw and Path(mask_path_raw).exists():
            with rasterio.open(mask_path_raw) as msrc:
                mask = msrc.read(1).astype(np.int64)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)

        augmented = self.transform(image=img, mask=mask)
        mask_arr = augmented["mask"]
        if isinstance(mask_arr, torch.Tensor):
            mask_tensor = mask_arr.long()
        else:
            mask_tensor = torch.from_numpy(mask_arr).long()

        return {
            "image": augmented["image"],
            "mask": mask_tensor,
            "img_path": str(img_path),
        }


if __name__ == "__main__":
    print("Testing OrthoDataset and DataLoader creation...")
    cfg = load_config()
    print(f"Loaded config: classes={cfg.dataset.classes}")
    try:
        train_loader, val_loader, test_loader = build_dataloaders(cfg=cfg)
        print(f"Successfully built dataloaders:")
        print(f"  Train batches: {len(train_loader)} (total samples: {len(train_loader.dataset)})")
        print(f"  Val batches:   {len(val_loader)} (total samples: {len(val_loader.dataset)})")
        if test_loader:
            print(f"  Test batches:  {len(test_loader)} (total samples: {len(test_loader.dataset)})")
        
        # Test 1 batch if available
        if len(train_loader) > 0:
            sample_batch = next(iter(train_loader))
            print(f"Sample batch image shape: {sample_batch['image'].shape}")
            print(f"Sample batch mask shape:  {sample_batch['mask'].shape}")
            print("Dataset check completed successfully!")
        else:
            print("No tiles found in data directory yet. Use preprocessing/tiling.py to generate tiles.")
    except Exception as e:
        print(f"Note: Could not load full batch (e.g. tiles directory may be empty or unpopulated): {e}")
