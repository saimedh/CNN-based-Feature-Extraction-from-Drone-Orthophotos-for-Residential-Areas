"""
utils/config.py
---------------
Loads and exposes the YAML configuration as a nested namespace so every
module can do:  from utils.config import load_config; cfg = load_config()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class _Config(dict):
    """Dot-access wrapper around a plain dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict):
                return _Config(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str | Path | None = None) -> _Config:
    """
    Load YAML config.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to YAML file.  Defaults to ``config/config.yaml`` relative to
        the repository root.

    Returns
    -------
    _Config
        A dot-accessible dict holding all configuration values.
    """
    if config_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return _Config(raw)


def ensure_dirs(cfg: _Config) -> None:
    """Create all directories defined in cfg.paths if they don't exist."""
    for name, path_str in cfg.paths.items():
        os.makedirs(path_str, exist_ok=True)
