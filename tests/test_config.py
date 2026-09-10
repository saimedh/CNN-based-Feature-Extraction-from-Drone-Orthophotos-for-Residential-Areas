"""
tests/test_config.py
---------------------
Unit tests for the YAML config loader.
"""

import pytest
from pathlib import Path
from utils.config import load_config, _Config


class TestLoadConfig:
    def test_loads_default_config(self):
        """Ensure the default config.yaml loads without errors."""
        cfg = load_config()
        assert isinstance(cfg, _Config)

    def test_dot_access(self):
        cfg = load_config()
        # Dot access should work one level deep
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "training")
        assert hasattr(cfg, "tiling")

    def test_nested_dot_access(self):
        cfg = load_config()
        assert cfg.model.architecture in ("unet", "deeplabv3plus", "fpn", "pspnet", "linknet", "manet")
        assert isinstance(cfg.tiling.tile_size, int)

    def test_missing_config_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/non/existent/path/config.yaml")

    def test_config_classes_is_list(self):
        cfg = load_config()
        assert isinstance(list(cfg.dataset.classes), list)
        assert "background" in cfg.dataset.classes
