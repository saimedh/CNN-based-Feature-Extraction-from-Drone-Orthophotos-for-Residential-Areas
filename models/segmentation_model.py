"""
models/segmentation_model.py
-----------------------------
Thin wrapper around ``segmentation-models-pytorch`` (smp) that builds the
segmentation network from YAML config.

Supported architectures
-----------------------
* ``unet``         – U-Net with configurable encoder
* ``deeplabv3plus``– DeepLabV3+ with configurable encoder

The wrapper also exposes ``predict_proba`` for inference so callers never
have to remember to apply softmax manually.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "segmentation-models-pytorch is required. "
        "Install via: pip install segmentation-models-pytorch"
    ) from exc

from utils.config import load_config, _Config
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_ARCH_MAP = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "linknet": smp.Linknet,
    "manet": smp.MAnet,
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Optional[_Config] = None) -> nn.Module:
    """
    Instantiate and return the segmentation model defined in config.

    Parameters
    ----------
    cfg : Loaded config.  Loaded from default location if ``None``.

    Returns
    -------
    torch.nn.Module
        The segmentation model (not moved to device yet).
    """
    if cfg is None:
        cfg = load_config()

    arch = str(cfg.model.architecture).lower()
    encoder = str(cfg.model.encoder)
    raw_weights = getattr(cfg.model, "encoder_weights", None)
    if raw_weights is None or str(raw_weights).lower() in ("none", "null", "false", ""):
        encoder_weights = None
    else:
        encoder_weights = str(raw_weights)
    in_channels = int(cfg.model.in_channels)
    num_classes = int(cfg.model.num_classes)

    if arch not in _ARCH_MAP:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from: {list(_ARCH_MAP)}"
        )

    ModelClass = _ARCH_MAP[arch]

    model = ModelClass(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits — loss handles activation internally
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Built {arch.upper()} | encoder={encoder} | "
        f"classes={num_classes} | trainable params={n_params:,}"
    )
    return model


# ---------------------------------------------------------------------------
# Wrapped model with convenience methods
# ---------------------------------------------------------------------------

class SegmentationModel(nn.Module):
    """
    Convenience wrapper that adds:
    * ``predict_proba`` – forward pass + softmax
    * ``predict``        – forward pass + argmax (class map)
    * ``load_checkpoint``
    * ``save_checkpoint``
    """

    def __init__(self, cfg: Optional[_Config] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.net = build_model(cfg)
        self.num_classes = int(cfg.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits [B, C, H, W]."""
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities [B, C, H, W]."""
        self.eval()
        return torch.softmax(self.net(x), dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class map [B, H, W]."""
        return self.predict_proba(x).argmax(dim=1)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        val_iou: float = 0.0,
        extra: Optional[dict] = None,
    ) -> None:
        payload = {
            "epoch": epoch,
            "val_iou": val_iou,
            "model_state_dict": self.state_dict(),
        }
        if optimizer_state is not None:
            payload["optimizer_state_dict"] = optimizer_state
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        logger.info(f"Checkpoint saved → {path}  (epoch={epoch}, val_IoU={val_iou:.4f})")

    def load_checkpoint(
        self,
        path: str,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict:
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(
            f"Loaded checkpoint '{path}'  "
            f"(epoch={checkpoint.get('epoch', '?')}, "
            f"val_IoU={checkpoint.get('val_iou', '?')})"
        )
        return checkpoint
