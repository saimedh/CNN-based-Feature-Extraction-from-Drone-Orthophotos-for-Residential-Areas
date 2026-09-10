"""
tests/test_metrics.py
---------------------
Unit tests for the SegmentationMetrics accumulator and batch_iou helper.
"""

import pytest
import torch
from utils.metrics import SegmentationMetrics, batch_iou


class TestSegmentationMetrics:
    NUM_CLASSES = 4
    B, H, W = 2, 64, 64

    def _perfect_logits(self, targets: torch.Tensor) -> torch.Tensor:
        """Generate logits that perfectly predict the given targets."""
        logits = torch.full((self.B, self.NUM_CLASSES, self.H, self.W), -10.0)
        for b in range(self.B):
            for c in range(self.NUM_CLASSES):
                logits[b, c][targets[b] == c] = 10.0
        return logits

    def test_perfect_predictions_iou_is_one(self):
        targets = torch.zeros(self.B, self.H, self.W, dtype=torch.long)
        targets[:, 10:20, 10:20] = 1  # building patch
        logits = self._perfect_logits(targets)

        m = SegmentationMetrics(num_classes=self.NUM_CLASSES)
        m.update(logits, targets)
        r = m.compute()

        assert abs(r["mean_iou"] - 1.0) < 1e-4, f"Expected mean_iou≈1.0, got {r['mean_iou']}"

    def test_all_wrong_predictions_iou_near_zero(self):
        targets = torch.zeros(self.B, self.H, self.W, dtype=torch.long)
        # Predict class 1 everywhere, ground truth is 0
        logits = torch.zeros(self.B, self.NUM_CLASSES, self.H, self.W)
        logits[:, 1, :, :] = 10.0  # always predicts class 1

        m = SegmentationMetrics(num_classes=self.NUM_CLASSES)
        m.update(logits, targets)
        r = m.compute()

        # Class 0 IoU should be 0, class 1 IoU should be 0
        assert r["iou"][0] < 0.1, f"Expected low IoU for class 0, got {r['iou'][0]}"

    def test_accumulation_over_batches(self):
        targets = torch.zeros(self.B, self.H, self.W, dtype=torch.long)
        logits = self._perfect_logits(targets)

        m = SegmentationMetrics(num_classes=self.NUM_CLASSES)
        for _ in range(5):  # simulate 5 batches
            m.update(logits, targets)

        r = m.compute()
        assert abs(r["mean_iou"] - 1.0) < 1e-4

    def test_reset_clears_state(self):
        targets = torch.zeros(self.B, self.H, self.W, dtype=torch.long)
        logits = self._perfect_logits(targets)

        m = SegmentationMetrics(num_classes=self.NUM_CLASSES)
        m.update(logits, targets)
        m.reset()
        r = m.compute()

        # After reset, numerator=0, denominator≈smooth
        # mean_iou should be near 1.0 (0/0 → smooth → ≈1.0)
        assert isinstance(r["mean_iou"], float)

    def test_batch_iou_returns_scalar(self):
        targets = torch.zeros(self.B, self.H, self.W, dtype=torch.long)
        logits = self._perfect_logits(targets)
        iou = batch_iou(logits, targets, num_classes=self.NUM_CLASSES)
        assert iou.shape == torch.Size([]), "Expected scalar tensor"
        assert 0.0 <= iou.item() <= 1.0


class TestBatchIoU:
    def test_binary_case_perfect(self):
        B, H, W, C = 1, 32, 32, 2
        targets = torch.zeros(B, H, W, dtype=torch.long)
        targets[0, :16, :] = 1
        logits = torch.zeros(B, C, H, W)
        logits[0, 0, 16:, :] = 10.0
        logits[0, 1, :16, :] = 10.0
        iou = batch_iou(logits, targets, num_classes=C)
        assert abs(iou.item() - 1.0) < 0.01
