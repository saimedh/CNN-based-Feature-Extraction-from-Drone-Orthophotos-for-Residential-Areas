"""
models/losses.py
-----------------
Combined Dice + Binary Cross-Entropy loss for multi-class segmentation.

The combination is:
    L = w_dice * DiceLoss + w_bce * CrossEntropyLoss

BCE here is actually ``nn.CrossEntropyLoss`` (works on raw logits with
multi-class targets), not ``BCEWithLogitsLoss`` which is binary-only.

References
----------
* Milletari et al. 2016 – "V-Net: Fully Convolutional Neural Networks for
  Volumetric Medical Image Segmentation" (Dice loss derivation)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss computed over all classes and averaged.

    Parameters
    ----------
    smooth    : Small constant to avoid division by zero.
    ignore_index : Class index to ignore (e.g. 255 for unlabelled pixels).
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, C, H, W]  raw (pre-softmax) model output.
        targets : [B, H, W]     integer class indices.

        Returns
        -------
        Scalar Dice loss.
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        # One-hot encode targets → [B, C, H, W]
        target_one_hot = F.one_hot(targets.long(), num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Flatten spatial dims
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        tgt_flat = target_one_hot.view(target_one_hot.shape[0], num_classes, -1)

        intersection = (probs_flat * tgt_flat).sum(dim=[0, 2])
        cardinality = probs_flat.sum(dim=[0, 2]) + tgt_flat.sum(dim=[0, 2])

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class DiceBCELoss(nn.Module):
    """
    Weighted combination of Dice loss and Cross-Entropy loss.

    Parameters
    ----------
    dice_weight  : Weight applied to the Dice loss term.
    bce_weight   : Weight applied to the CrossEntropy loss term.
    smooth       : Smoothing factor for Dice.
    ignore_index : Class index to exclude from CE loss.
    class_weights: Optional [C] tensor of per-class weights for CE.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, C, H, W]
        targets : [B, H, W]

        Returns
        -------
        Combined scalar loss.
        """
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets.long())
        return self.dice_weight * dice_loss + self.bce_weight * ce_loss


def build_loss(cfg=None) -> DiceBCELoss:
    """
    Factory that constructs the loss from YAML config.

    Parameters
    ----------
    cfg : Loaded _Config.

    Returns
    -------
    DiceBCELoss instance.
    """
    from utils.config import load_config

    if cfg is None:
        cfg = load_config()

    loss_cfg = cfg.training.loss
    return DiceBCELoss(
        dice_weight=float(loss_cfg.dice_weight),
        bce_weight=float(loss_cfg.bce_weight),
    )
