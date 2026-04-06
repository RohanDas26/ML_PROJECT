"""
src.utils.io — I/O Helpers
===========================
Model serialization, CSV/JSON export, and path utilities.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    """Serialize a model (or pipeline) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Model saved -> %s (%.1f KB)", path, path.stat().st_size / 1024)


def load_model(path: str | Path) -> Any:
    """Deserialize a model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    log.info("Model loaded <- %s", path)
    return model


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save a DataFrame to CSV with sensible defaults."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    log.info("DataFrame saved -> %s (%d rows x %d cols)", path, len(df), len(df.columns))


def save_json(data: dict, path: str | Path) -> None:
    """Save a dictionary as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("JSON saved -> %s", path)


def load_json(path: str | Path) -> dict:
    """Load a JSON file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory tree if it doesn't exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
