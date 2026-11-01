"""
utils/metrics.py
----------------
Pixel-wise segmentation metrics: IoU (Jaccard), Dice, F1 per class.
Works on PyTorch tensors — batch dimension expected.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _one_hot(
    targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Convert class-index mask [B, H, W] → one-hot [B, C, H, W]."""
    clamped = torch.clamp(targets.long(), 0, num_classes - 1)
    return F.one_hot(clamped, num_classes).permute(0, 3, 1, 2).float()


class SegmentationMetrics:
    """
    Accumulates predictions across batches and computes final metrics.

    Usage
    -----
    >>> metrics = SegmentationMetrics(num_classes=4, smooth=1e-6)
    >>> for logits, targets in loader:
    ...     metrics.update(logits, targets)
    >>> results = metrics.compute()
    """

    def __init__(self, num_classes: int, smooth: float = 1e-6) -> None:
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self) -> None:
        """Clear accumulators."""
        self._intersection = torch.zeros(self.num_classes)
        self._union = torch.zeros(self.num_classes)
        self._pred_sum = torch.zeros(self.num_classes)
        self._target_sum = torch.zeros(self.num_classes)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        logits  : [B, C, H, W] raw model output or [B, H, W] class indices.
        targets : [B, H, W] integer class indices.
        """
        if logits.dim() == 4:
            preds = logits.argmax(dim=1)  # [B, H, W]
        else:
            preds = logits
        pred_oh = _one_hot(preds, self.num_classes)
        tgt_oh = _one_hot(targets, self.num_classes)

        # Flatten spatial dims for dot products
        pred_flat = pred_oh.view(pred_oh.shape[0], self.num_classes, -1)
        tgt_flat = tgt_oh.view(tgt_oh.shape[0], self.num_classes, -1)

        intersection = (pred_flat * tgt_flat).sum(dim=[0, 2])   # [C]
        union = pred_flat.sum(dim=[0, 2]) + tgt_flat.sum(dim=[0, 2]) - intersection

        self._intersection += intersection.cpu()
        self._union += union.cpu()
        self._pred_sum += pred_flat.sum(dim=[0, 2]).cpu()
        self._target_sum += tgt_flat.sum(dim=[0, 2]).cpu()

    def compute(self) -> dict[str, list[float]]:
        """
        Returns a dict with per-class and mean values for IoU, Dice, F1.

        Returns
        -------
        dict with keys: ``iou``, ``dice``, ``f1``, ``mean_iou``, ``mean_dice``, ``mean_f1``
        """
        eps = self.smooth

        iou = ((self._intersection + eps) / (self._union + eps)).tolist()
        dice = (
            (2 * self._intersection + eps)
            / (self._pred_sum + self._target_sum + eps)
        ).tolist()
        # F1 == Dice for binary-equivalent per-class evaluation
        f1 = dice

        total_px = float(self._target_sum.sum().item())
        correct = float(self._intersection.sum().item())
        pixel_accuracy = (correct / total_px) if total_px > 0 else 0.0

        return {
            "iou": iou,
            "dice": dice,
            "f1": f1,
            "mean_iou": float(sum(iou) / len(iou)),
            "mean_dice": float(sum(dice) / len(dice)),
            "mean_f1": float(sum(f1) / len(f1)),
            "pixel_accuracy": float(pixel_accuracy),
        }


def batch_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Convenience function for per-batch IoU (used inside training loop).

    Returns
    -------
    mean IoU scalar tensor.
    """
    preds = logits.argmax(dim=1)
    pred_oh = _one_hot(preds, num_classes).view(preds.shape[0], num_classes, -1)
    tgt_oh = _one_hot(targets, num_classes).view(targets.shape[0], num_classes, -1)
    intersection = (pred_oh * tgt_oh).sum(dim=[0, 2])
    union = pred_oh.sum(dim=[0, 2]) + tgt_oh.sum(dim=[0, 2]) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


def numpy_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    class_names: Optional[list[str]] = None,
    smooth: float = 1e-6,
) -> dict:
    """
    Pixel-wise IoU / Dice / F1 from integer class maps [H, W].
    """
    names = class_names or [str(i) for i in range(num_classes)]
    per_class: dict = {}
    ious, dices, f1s = [], [], []

    for c in range(num_classes):
        pred_c = pred == c
        tgt_c = target == c
        inter = float(np.logical_and(pred_c, tgt_c).sum())
        pred_sum = float(pred_c.sum())
        tgt_sum = float(tgt_c.sum())
        union = pred_sum + tgt_sum - inter
        iou = (inter + smooth) / (union + smooth)
        dice = (2 * inter + smooth) / (pred_sum + tgt_sum + smooth)
        name = names[c] if c < len(names) else str(c)
        per_class[name] = {
            "iou": round(float(iou), 4),
            "dice": round(float(dice), 4),
            "f1": round(float(dice), 4),
        }
        ious.append(float(iou))
        dices.append(float(dice))
        f1s.append(float(dice))

    return {
        "per_class": per_class,
        "mean_iou": round(float(np.mean(ious)), 4),
        "mean_dice": round(float(np.mean(dices)), 4),
        "mean_f1": round(float(np.mean(f1s)), 4),
    }


def softmax_confidence_by_class(
    probs: np.ndarray,
    class_names: list[str],
) -> dict[str, dict]:
    """
    Mean softmax confidence for pixels assigned to each class.

    Parameters
    ----------
    probs : [C, H, W] softmax probabilities
    """
    pred = np.argmax(probs, axis=0)
    conf_map = np.max(probs, axis=0)
    out: dict[str, dict] = {}
    for c, name in enumerate(class_names):
        mask = pred == c
        n_px = int(mask.sum())
        mean_conf = float(conf_map[mask].mean()) if n_px else 0.0
        out[name] = {
            "confidence": round(mean_conf, 4),
            "pixel_count": n_px,
        }
    return out
