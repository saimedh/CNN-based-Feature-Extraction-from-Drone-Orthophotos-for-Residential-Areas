"""
training/train.py
-----------------
Full training loop with:
* Mixed-precision (fp16) via torch.cuda.amp
* AdamW optimiser + cosine-annealing LR scheduler
* Best-model checkpointing (by validation IoU)
* Early stopping
* TensorBoard logging (optional)
* Graceful KeyboardInterrupt handling

Usage
-----
    python training/train.py
    python training/train.py --config config/config.yaml
    python training/train.py --resume checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import SegmentationModel, build_loss
from preprocessing.dataset import build_dataloaders
from utils.config import load_config, ensure_dirs, _Config
from utils.logger import get_logger
from utils.metrics import SegmentationMetrics, batch_iou

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Keeps only the top-k checkpoints by val IoU."""

    def __init__(self, ckpt_dir: str, top_k: int = 3) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self._heap: list[tuple[float, str]] = []  # (neg_iou, path)

    def save(self, model: SegmentationModel, optimizer: AdamW, epoch: int, val_iou: float) -> str:
        path = str(self.ckpt_dir / f"epoch_{epoch:04d}_iou_{val_iou:.4f}.pt")
        model.save_checkpoint(
            path,
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            val_iou=val_iou,
        )
        heapq.heappush(self._heap, (val_iou, path))  # min-heap by iou

        # Evict worst if > top_k
        if len(self._heap) > self.top_k:
            _, worst_path = heapq.heappop(self._heap)
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass

        return path

    @property
    def best_iou(self) -> float:
        return max(iou for iou, _ in self._heap) if self._heap else 0.0


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: SegmentationModel,
    loader,
    optimizer: AdamW,
    loss_fn,
    scaler: GradScaler,
    device: torch.device,
    cfg: _Config,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_classes = int(cfg.model.num_classes)

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.training.mixed_precision):
            logits = model(images)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        iou = batch_iou(logits.detach().float(), masks, num_classes)
        total_loss += loss.item()
        total_iou += iou.item()

        if (batch_idx + 1) % 20 == 0:
            logger.debug(
                f"  [E{epoch}] step {batch_idx+1}/{len(loader)} "
                f"loss={loss.item():.4f}  batch_iou={iou.item():.4f}"
            )

    n = len(loader)
    return {"loss": total_loss / n, "iou": total_iou / n}


@torch.no_grad()
def _val_epoch(
    model: SegmentationModel,
    loader,
    loss_fn,
    device: torch.device,
    cfg: _Config,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=int(cfg.model.num_classes))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=cfg.training.mixed_precision):
            logits = model(images)
            loss = loss_fn(logits, masks)

        total_loss += loss.item()
        metrics.update(logits.float(), masks)

    results = metrics.compute()
    results["loss"] = total_loss / len(loader)
    return results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    cfg_path: Optional[str] = None,
    resume: Optional[str] = None,
) -> None:
    """
    Run the full training pipeline.

    Parameters
    ----------
    cfg_path : Path to YAML config (None → default location).
    resume   : Path to checkpoint to resume from.
    """
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Data ----
    train_loader, val_loader, _ = build_dataloaders(
        img_dir=cfg.paths.tiles_dir,
        mask_dir=cfg.paths.masks_dir,
        cfg=cfg,
    )

    # ---- Model ----
    model = SegmentationModel(cfg).to(device)

    # ---- Loss / Optimiser / Scheduler ----
    loss_fn = build_loss(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    total_epochs = int(cfg.training.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)
    scaler = GradScaler(enabled=cfg.training.mixed_precision)

    ckpt_manager = CheckpointManager(
        ckpt_dir=cfg.paths.checkpoints_dir,
        top_k=int(cfg.training.save_top_k),
    )

    # ---- Resume ----
    start_epoch = 1
    if resume:
        ckpt = model.load_checkpoint(resume, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Resuming from epoch {start_epoch}")

    # ---- TensorBoard (optional) ----
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=str(Path(cfg.paths.logs_dir) / "tensorboard"))
        logger.info("TensorBoard logging enabled.")
    except Exception:
        tb_writer = None

    # ---- Training loop ----
    patience = int(cfg.training.early_stopping_patience)
    no_improve = 0
    best_iou = ckpt_manager.best_iou

    logger.info(
        f"Starting training for {total_epochs} epochs  "
        f"(patience={patience}, mixed_precision={cfg.training.mixed_precision})"
    )

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            t0 = time.time()

            train_stats = _train_epoch(
                model, train_loader, optimizer, loss_fn, scaler, device, cfg, epoch
            )
            val_stats = _val_epoch(model, val_loader, loss_fn, device, cfg)
            scheduler.step()

            val_iou = val_stats["mean_iou"]
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"train_loss={train_stats['loss']:.4f}  "
                f"train_iou={train_stats['iou']:.4f} | "
                f"val_loss={val_stats['loss']:.4f}  "
                f"val_iou={val_iou:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"time={elapsed:.1f}s"
            )

            if tb_writer:
                tb_writer.add_scalars("Loss", {"train": train_stats["loss"], "val": val_stats["loss"]}, epoch)
                tb_writer.add_scalars("IoU", {"train": train_stats["iou"], "val": val_iou}, epoch)
                tb_writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

            # ---- Checkpoint ----
            ckpt_manager.save(model, optimizer, epoch, val_iou)

            # ---- Best model symlink / copy ----
            if val_iou > best_iou:
                best_iou = val_iou
                no_improve = 0
                best_path = Path(cfg.paths.checkpoints_dir) / "best.pt"
                model.save_checkpoint(
                    str(best_path), epoch=epoch,
                    optimizer_state=optimizer.state_dict(), val_iou=val_iou
                )
                logger.info(f"  ↑ New best model  val_IoU={best_iou:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state…")
        model.save_checkpoint(
            str(Path(cfg.paths.checkpoints_dir) / "interrupted.pt"),
            epoch=epoch, optimizer_state=optimizer.state_dict(), val_iou=val_iou
        )

    if tb_writer:
        tb_writer.close()

    logger.info(f"Training complete. Best val_IoU: {best_iou:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on orthophoto tiles.")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(cfg_path=args.config, resume=args.resume)
