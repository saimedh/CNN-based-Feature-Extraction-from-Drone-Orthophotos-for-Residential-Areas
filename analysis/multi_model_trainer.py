"""
analysis/multi_model_trainer.py
--------------------------------
Multi-model training, benchmarking, and accuracy comparison engine.
Trains and compares multiple CNN semantic segmentation architectures
(U-Net, DeepLabV3+, FPN, MAnet, PSPNet, Linknet) on drone orthophoto tiles.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.segmentation_model import SegmentationModel, _ARCH_MAP
from models.losses import build_loss
from preprocessing.dataset import build_dataloaders
from utils.config import load_config, _Config
from utils.logger import get_logger
from utils.metrics import SegmentationMetrics, batch_iou

logger = get_logger(__name__)

# Predefined architecture configurations
MODEL_SPECS = {
    "unet": {
        "name": "U-Net (ResNet34)",
        "arch": "unet",
        "encoder": "resnet34",
        "description": "Standard U-Net with skip connections; great overall feature delineation.",
    },
    "deeplabv3plus": {
        "name": "DeepLabV3+ (ResNet50)",
        "arch": "deeplabv3plus",
        "encoder": "resnet50",
        "description": "Atrous Spatial Pyramid Pooling; superior road and boundary capture.",
    },
    "fpn": {
        "name": "FPN (ResNet34)",
        "arch": "fpn",
        "encoder": "resnet34",
        "description": "Feature Pyramid Network; efficient multi-scale semantic representation.",
    },
    "manet": {
        "name": "MAnet (ResNet34)",
        "arch": "manet",
        "encoder": "resnet34",
        "description": "Multi-scale Attention Network; fine-grained boundary sharpness.",
    },
    "pspnet": {
        "name": "PSPNet (ResNet34)",
        "arch": "pspnet",
        "encoder": "resnet34",
        "description": "Pyramid Scene Parsing Network; strong global context aggregation.",
    },
    "linknet": {
        "name": "Linknet (ResNet34)",
        "arch": "linknet",
        "encoder": "resnet34",
        "description": "Lightweight encoder-decoder linking; fastest inference throughput.",
    },
}


def build_custom_model(arch: str, encoder: str, num_classes: int = 4, in_channels: int = 3) -> nn.Module:
    """Build a segmentation model given architecture and encoder name."""
    import segmentation_models_pytorch as smp
    arch_lower = arch.lower()
    if arch_lower not in _ARCH_MAP:
        raise ValueError(f"Unknown architecture '{arch}'. Supported: {list(_ARCH_MAP.keys())}")
    
    builder = _ARCH_MAP[arch_lower]
    return builder(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
    )


def train_single_model(
    model_key: str,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    cfg: Optional[_Config] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a single architecture on the dataset.
    Returns full accuracy metrics and performance breakdown.
    """
    if cfg is None:
        cfg = load_config()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = MODEL_SPECS.get(model_key, {
        "name": model_key.upper(),
        "arch": model_key,
        "encoder": "resnet34",
        "description": "Custom model architecture",
    })

    logger.info(f"--- Training {spec['name']} ({epochs} epochs) on {device} ---")

    # Build model
    num_classes = int(cfg.dataset.num_classes)
    model = build_custom_model(
        arch=spec["arch"],
        encoder=spec["encoder"],
        num_classes=num_classes,
        in_channels=int(cfg.model.in_channels),
    ).to(device)

    param_count_m = sum(p.numel() for p in model.parameters()) / 1e6

    # Dataloaders
    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / "data" / "tiles" / "images"
    mask_dir = base_dir / "data" / "tiles" / "masks"
    if not img_dir.is_dir() or not any(img_dir.iterdir()):
        img_dir = base_dir / "data" / "tiles" / "tiles" / "images"
        mask_dir = base_dir / "data" / "tiles" / "tiles" / "masks"

    all_tiles = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.png"))
    if not img_dir.is_dir() or not mask_dir.is_dir() or not all_tiles:
        raise FileNotFoundError(
            f"No tiled labelled dataset at '{img_dir}' / '{mask_dir}'. "
            "Tile a GeoTIFF with a paired mask before training."
        )

    try:
        cfg.training.batch_size = batch_size
        cfg.training.num_workers = 0
        train_loader, val_loader, test_loader = build_dataloaders(
            img_dir=img_dir,
            mask_dir=mask_dir,
            cfg=cfg,
        )
    except Exception as e:
        raise RuntimeError(f"Could not build dataloaders from tiled dataset: {e}") from e

    if not train_loader or len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Add labelled tiles before benchmarking.")

    criterion = build_loss(cfg)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    start_time = time.time()
    train_losses = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0
        for batch in train_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        train_losses.append(round(avg_loss, 4))
        logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    training_time_sec = round(time.time() - start_time, 2)

    model.eval()
    eval_metrics = SegmentationMetrics(num_classes=num_classes)
    eval_loader = test_loader or val_loader
    if not eval_loader or len(eval_loader) == 0:
        raise RuntimeError("No validation/test tiles available for real IoU/F1 accuracy.")

    inference_times = []
    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"]
            t0 = time.time()
            logits = model(images)
            inference_times.append((time.time() - t0) / max(1, images.shape[0]))
            eval_metrics.update(logits.cpu(), masks.cpu())

    scores = eval_metrics.compute()
    iou_per_class = scores["iou"]
    f1_per_class = scores["f1"]
    mean_iou = scores["mean_iou"]
    mean_f1 = scores["mean_f1"]
    pixel_acc = float(scores.get("pixel_accuracy") or 0.0)

    avg_latency_ms = round(float(np.mean(inference_times)) * 1000, 1) if inference_times else None
    fps = round(1000.0 / avg_latency_ms, 1) if avg_latency_ms else None

    class_names = list(cfg.dataset.classes)
    class_breakdown = {}
    for i, cname in enumerate(class_names):
        ciou = float(iou_per_class[i]) if i < len(iou_per_class) else 0.0
        cf1 = float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
        class_breakdown[cname] = {
            "iou": round(ciou * 100, 2),
            "f1": round(cf1 * 100, 2),
            "dice": round(cf1 * 100, 2),
        }

    return {
        "model_key": model_key,
        "name": spec["name"],
        "architecture": spec["arch"],
        "encoder": spec["encoder"],
        "description": spec["description"],
        "parameters_m": round(param_count_m, 2),
        "training_time_sec": training_time_sec,
        "epochs": epochs,
        "final_train_loss": train_losses[-1] if train_losses else 0.0,
        "mean_iou_pct": round(mean_iou * 100, 2),
        "mean_f1_pct": round(mean_f1 * 100, 2),
        "pixel_accuracy_pct": round(pixel_acc * 100, 2),
        "latency_ms": avg_latency_ms,
        "inference_fps": fps,
        "class_breakdown": class_breakdown,
        "train_losses": train_losses,
        "metrics_source": "tiled_dataset_eval",
    }


def train_and_benchmark_multi_models(
    model_keys: Optional[List[str]] = None,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    cfg: Optional[_Config] = None,
) -> Dict[str, Any]:
    """
    Train and compare multiple models side-by-side.
    Returns comprehensive benchmark report and generates comparison chart.
    """
    if cfg is None:
        cfg = load_config()

    if not model_keys:
        model_keys = ["unet", "deeplabv3plus", "fpn", "manet", "pspnet", "linknet"]

    logger.info(f"Starting Multi-Model Benchmark across {len(model_keys)} architectures...")
    results = []

    for key in model_keys:
        try:
            res = train_single_model(
                model_key=key,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                cfg=cfg,
            )
            results.append(res)
        except Exception as e:
            logger.exception(f"Training failed for {key}: {e}")

    # Rank models by mean IoU
    results = sorted(results, key=lambda x: x["mean_iou_pct"], reverse=True)
    for rank, res in enumerate(results, 1):
        res["rank"] = rank

    # Compile report
    report = {
        "timestamp": time.time(),
        "total_models": len(results),
        "epochs": epochs,
        "learning_rate": lr,
        "best_model": results[0]["name"] if results else None,
        "best_miou": results[0]["mean_iou_pct"] if results else 0,
        "metrics_source": "tiled_dataset_eval",
        "models": results,
    }

    # Save report to outputs directory
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / cfg.paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "multi_model_accuracy_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Multi-model accuracy report saved → {report_file}")

    # Generate comparative chart PNG
    chart_file = out_dir / "multi_model_accuracy_chart.png"
    try:
        generate_multi_model_chart(report, chart_file)
    except Exception as e:
        logger.warning(f"Could not generate chart PNG: {e}")

    return report


def generate_multi_model_chart(report: Dict[str, Any], output_path: str | Path) -> None:
    """Generate a multi-model comparison bar chart and save to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = report.get("models", [])
    if not models:
        return

    names = [m["name"].split("(")[0].strip() for m in models]
    mious = [m["mean_iou_pct"] for m in models]
    bldgs = [m["class_breakdown"].get("building", {}).get("iou", 0) for m in models]
    roads = [m["class_breakdown"].get("road", {}).get("iou", 0) for m in models]
    vegs = [m["class_breakdown"].get("vegetation", {}).get("iou", 0) for m in models]
    fps_list = [m["inference_fps"] or 0 for m in models]

    x = np.arange(len(names))
    width = 0.18

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax1.set_facecolor("#0f172a")

    # Bar series
    r1 = ax1.bar(x - 1.5 * width, mious, width, label="Mean IoU (%)", color="#6366f1", alpha=0.9)
    r2 = ax1.bar(x - 0.5 * width, bldgs, width, label="Building IoU (%)", color="#f59e0b", alpha=0.9)
    r3 = ax1.bar(x + 0.5 * width, roads, width, label="Road IoU (%)", color="#38bdf8", alpha=0.9)
    r4 = ax1.bar(x + 1.5 * width, vegs, width, label="Vegetation IoU (%)", color="#10b981", alpha=0.9)

    ax1.set_ylabel("IoU Accuracy (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_title("Multi-Model Segmentation Accuracy & Speed Benchmark", color="#f8fafc", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, color="#e2e8f0", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add speed on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, fps_list, color="#ec4899", marker="o", linewidth=2.5, label="Throughput (FPS)")
    ax2.set_ylabel("Inference Speed (FPS)", color="#ec4899", fontsize=11, fontweight="bold")
    ax2.tick_params(colors="#ec4899")
    ax2.set_ylim(0, max(fps_list) * 1.3)

    # Combined legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(str(output_path), facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
