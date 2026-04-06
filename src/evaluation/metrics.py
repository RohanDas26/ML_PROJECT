"""
src.evaluation.metrics — Custom Metrics
========================================
MAPE, adjusted-R², and any domain-specific metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Undefined when y_true contains zeros; uses ε floor.
    """
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Adjusted R² that penalises unnecessary features.

    adj_R² = 1 - (1-R²) × (n-1) / (n-p-1)
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n - n_features - 1 <= 0:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int | None = None,
) -> dict:
    """
    Compute a full suite of regression metrics.

    Returns dict with keys: MSE, RMSE, MAE, R2, MAPE, Adj_R2 (if n_features given).
    """
    metrics = {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }
    if n_features is not None:
        metrics["Adj_R2"] = adjusted_r2(y_true, y_pred, n_features)
    return metrics
