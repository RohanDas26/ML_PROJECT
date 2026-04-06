"""
src.utils.logger — Structured Logging
======================================
Provides a consistent, color-coded logger for the entire pipeline.
"""

import logging
import sys
from pathlib import Path


_CONFIGURED_LOGGERS = set()


def get_logger(name: str = "energy_forecast",
               level: str = "INFO",
               log_file: str | None = None) -> logging.Logger:
    """Return a configured logger (idempotent per logger name)."""
    global _CONFIGURED_LOGGERS

    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler (use UTF-8 wrapper to avoid cp1252 errors on Windows)
    import io
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = logging.StreamHandler(stream)
    console.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED_LOGGERS.add(name)
    return logger

