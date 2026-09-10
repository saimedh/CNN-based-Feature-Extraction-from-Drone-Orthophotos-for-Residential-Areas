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
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    ) -> None:
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Collect all image tile paths (sorted for reproducibility)
        self.img_paths = sorted(self.img_dir.glob(f"*{tile_ext}"))
        if not self.img_paths:
            raise FileNotFoundError(
                f"No '{tile_ext}' files found in '{img_dir}'. "
                "Run preprocessing/tiling.py first."
            )

        # Build paired mask paths and validate
        self.mask_paths: list[Optional[Path]] = []
        for ip in self.img_paths:
            candidate = self.mask_dir / (ip.stem + "_mask" + tile_ext)
            if candidate.exists():
                self.mask_paths.append(candidate)
            else:
                # Fallback: mask file with same stem (no suffix)
                candidate2 = self.mask_dir / (ip.name)
                self.mask_paths.append(candidate2 if candidate2.exists() else None)

        logger.info(
            f"OrthophotoDataset: {len(self.img_paths)} tiles | "
            f"masks found: {sum(1 for m in self.mask_paths if m)}"
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

        # --- Load image (bands × H × W → H × W × 3 for albumentations) ---
        with rasterio.open(img_path) as src:
            img = src.read()  # (C, H, W), uint8 or uint16

        # Normalise to [0, 255] uint8 for albumentations compatibility
        if img.dtype != np.uint8:
            img = _rescale_to_uint8(img)

        img = np.transpose(img[:3], (1, 2, 0))  # Keep only first 3 bands → (H, W, 3)

        # --- Load mask ---
        if mask_path and mask_path.exists():
            with rasterio.open(mask_path) as msrc:
                mask = msrc.read(1).astype(np.int64)  # (H, W)
        else:
            # Dummy all-background mask when no label exists
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)

        # --- Apply transforms ---
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            image_tensor = augmented["image"]         # FloatTensor [3, H, W]
            mask_tensor = torch.from_numpy(augmented["mask"]).long()
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
    img_dir: str | Path,
    mask_dir: str | Path,
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

    full_dataset = OrthophotoDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=None,  # transforms applied after splitting
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
    val_ds = _TransformSubset(val_ds, val_tfm)
    test_ds = _TransformSubset(test_ds, val_tfm)

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

    def __init__(self, subset: torch.utils.data.Subset, transform: Callable) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.subset.dataset[self.subset.indices[idx]]
        # Re-apply transform over the raw arrays
        # (OrthophotoDataset already returns tensors; we need raw here)
        img_path = sample["img_path"]
        with rasterio.open(img_path) as src:
            img = src.read()
        if img.dtype != np.uint8:
            img = _rescale_to_uint8(img)
        img = np.transpose(img[:3], (1, 2, 0))

        mask_path_raw = self.subset.dataset.mask_paths[self.subset.indices[idx]]
        if mask_path_raw and Path(mask_path_raw).exists():
            with rasterio.open(mask_path_raw) as msrc:
                mask = msrc.read(1).astype(np.int64)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)

        augmented = self.transform(image=img, mask=mask)
        return {
            "image": augmented["image"],
            "mask": torch.from_numpy(augmented["mask"]).long(),
            "img_path": img_path,
        }
