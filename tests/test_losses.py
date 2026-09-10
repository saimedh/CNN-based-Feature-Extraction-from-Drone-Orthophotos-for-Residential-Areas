"""
tests/test_losses.py
---------------------
Unit tests for DiceLoss and DiceBCELoss.
"""

import pytest
import torch
from models.losses import DiceLoss, DiceBCELoss


class TestDiceLoss:
    C, B, H, W = 4, 2, 32, 32

    def _perfect_logits(self, targets: torch.Tensor) -> torch.Tensor:
        """Build logits that give near-perfect softmax probs for each target class."""
        logits = torch.full((self.B, self.C, self.H, self.W), -10.0)
        for b in range(self.B):
            for c in range(self.C):
                mask = targets[b] == c
                if mask.any():
                    logits[b, c][mask] = 10.0
        return logits

    def test_perfect_prediction_loss_near_zero(self):
        # Ensure ALL 4 classes are present so Dice is well-defined for every class.
        targets = torch.zeros(self.B, self.H, self.W, dtype=torch.long)
        targets[:, 10:20, 10:20] = 1   # building patch
        targets[:, 20:28, 20:28] = 2   # road patch
        targets[:,  0:8,  24:32] = 3   # vegetation patch
        logits = self._perfect_logits(targets)
        loss = DiceLoss()(logits, targets)
        assert loss.item() < 0.01, f"Expected loss≈0, got {loss.item()}"

    def test_loss_is_scalar(self):
        targets = torch.randint(0, self.C, (self.B, self.H, self.W))
        logits = torch.randn(self.B, self.C, self.H, self.W)
        loss = DiceLoss()(logits, targets)
        assert loss.shape == torch.Size([])

    def test_loss_range_zero_to_one(self):
        targets = torch.randint(0, self.C, (self.B, self.H, self.W))
        logits = torch.randn(self.B, self.C, self.H, self.W)
        loss = DiceLoss()(logits, targets)
        assert 0.0 <= loss.item() <= 1.05  # slight overshoot possible with noisy logits


class TestDiceBCELoss:
    C, B, H, W = 4, 2, 32, 32

    def test_combined_loss_is_positive(self):
        targets = torch.randint(0, self.C, (self.B, self.H, self.W))
        logits = torch.randn(self.B, self.C, self.H, self.W)
        loss_fn = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.0

    def test_weights_sum_to_one_or_not_required(self):
        """Weights don't have to sum to 1 — just check the weighted combination runs."""
        targets = torch.randint(0, self.C, (self.B, self.H, self.W))
        logits = torch.randn(self.B, self.C, self.H, self.W)
        loss_fn = DiceBCELoss(dice_weight=0.7, bce_weight=0.3)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), "Loss should be finite"
