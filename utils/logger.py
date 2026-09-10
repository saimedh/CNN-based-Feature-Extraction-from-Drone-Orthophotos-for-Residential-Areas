"""
utils/logger.py
---------------
Centralised logging setup:
- console handler (coloured via ``colorlog`` if available)
- rotating file handler
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import colorlog  # type: ignore

    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


def get_logger(
    name: str = "orthophoto",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a configured Logger.

    Parameters
    ----------
    name    : Logger name (typically the calling module's ``__name__``).
    log_dir : Directory where the rotating log file is stored.
    level   : Logging level (default INFO).
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # --- Console handler ---
    if _HAS_COLORLOG:
        console_fmt = colorlog.ColoredFormatter(
            "%(log_color)s" + fmt,
            datefmt=datefmt,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        console_fmt = logging.Formatter(fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # --- File handler (10 MB, keep 5 backups) ---
    log_file = Path(log_dir) / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger
