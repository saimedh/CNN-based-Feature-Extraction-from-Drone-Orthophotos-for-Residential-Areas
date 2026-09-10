"""
evaluation/evaluate.py
-----------------------
Runs inference on a test split (or any labelled tile set) and computes
per-class IoU, Dice, and F1.

Outputs
-------
* Console table of per-class and mean metrics
* ``outputs/evaluation_report.json``   – machine-readable
* ``outputs/evaluation_report.txt``    – human-readable

Usage
-----
    python evaluation/evaluate.py --checkpoint checkpoints/best.pt
    python evaluation/evaluate.py --checkpoint checkpoints/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.cuda.amp import autocast

from models import SegmentationModel
from preprocessing.dataset import build_dataloaders
from utils.config import load_config, _Config
from utils.logger import get_logger
from utils.metrics import SegmentationMetrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    cfg_path: Optional[str] = None,
    split: str = "test",
) -> dict:
    """
    Load a checkpoint and evaluate on the requested data split.

    Parameters
    ----------
    checkpoint_path : Path to .pt checkpoint file.
    cfg_path        : Path to YAML config (None → default location).
    split           : One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    Dict of metrics including per-class and mean values.
    """
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Evaluating split: {split}")

    # ---- Model ----
    model = SegmentationModel(cfg).to(device)
    model.load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # ---- Data ----
    train_loader, val_loader, test_loader = build_dataloaders(
        img_dir=cfg.paths.tiles_dir,
        mask_dir=cfg.paths.masks_dir,
        cfg=cfg,
    )
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map.get(split, test_loader)

    # ---- Metrics accumulator ----
    metrics = SegmentationMetrics(num_classes=int(cfg.model.num_classes))
    class_names = list(cfg.dataset.classes)

    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=cfg.training.mixed_precision):
            logits = model(images)

        metrics.update(logits.float(), masks)

        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"  Processed {batch_idx + 1}/{len(loader)} batches…")

    elapsed = time.time() - t0
    results = metrics.compute()
    results["elapsed_seconds"] = round(elapsed, 2)
    results["split"] = split
    results["checkpoint"] = checkpoint_path

    # ---- Pretty-print table ----
    _print_results(results, class_names)

    # ---- Save report ----
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    txt_path = output_dir / "evaluation_report.txt"
    _save_txt_report(results, class_names, txt_path)

    logger.info(f"Evaluation report saved to {output_dir}")
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_results(results: dict, class_names: list[str]) -> None:
    """Print a formatted table to the console."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"{'Class':<20} {'IoU':>8}  {'Dice':>8}  {'F1':>8}")
    print(sep)
    for i, name in enumerate(class_names):
        iou = results["iou"][i]
        dice = results["dice"][i]
        f1 = results["f1"][i]
        print(f"{name:<20} {iou:>8.4f}  {dice:>8.4f}  {f1:>8.4f}")
    print(sep)
    print(
        f"{'Mean':<20} "
        f"{results['mean_iou']:>8.4f}  "
        f"{results['mean_dice']:>8.4f}  "
        f"{results['mean_f1']:>8.4f}"
    )
    print(f"{sep}\n")


def _save_txt_report(results: dict, class_names: list[str], path: Path) -> None:
    lines = [
        "CNN-based Feature Extraction — Evaluation Report",
        "=" * 60,
        f"Split:      {results.get('split', 'N/A')}",
        f"Checkpoint: {results.get('checkpoint', 'N/A')}",
        f"Elapsed:    {results.get('elapsed_seconds', 0):.1f}s",
        "",
        f"{'Class':<20} {'IoU':>8}  {'Dice':>8}  {'F1':>8}",
        "─" * 60,
    ]
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<20} "
            f"{results['iou'][i]:>8.4f}  "
            f"{results['dice'][i]:>8.4f}  "
            f"{results['f1'][i]:>8.4f}"
        )
    lines += [
        "─" * 60,
        f"{'Mean':<20} "
        f"{results['mean_iou']:>8.4f}  "
        f"{results['mean_dice']:>8.4f}  "
        f"{results['mean_f1']:>8.4f}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        cfg_path=args.config,
        split=args.split,
    )
