"""
src.visualization.plots — Publication-Quality Plots
=====================================================
All matplotlib/seaborn plots used for results, diagnostics,
and presentation figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers

from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

# Global style
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
})


def _save(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    """Save figure and log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved -> %s", path)


# ------------------------------------------------------------------
# 1. Actual vs Predicted
# ------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot with perfect-prediction diagonal."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="k", linewidth=0.3)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual (Trillion BTU)")
    ax.set_ylabel("Predicted (Trillion BTU)")
    ax.set_title(title)
    ax.legend()
    if save_path:
        _save(fig, save_path)
    return fig


# ------------------------------------------------------------------
# 2. Residuals
# ------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Residuals vs predicted + histogram side-by-side."""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors="k", linewidth=0.3)
    ax1.axhline(0, color="r", linestyle="--", lw=1.5)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residual")
    ax1.set_title("Residuals vs Predicted")

    # Histogram
    ax2.hist(residuals, bins=30, edgecolor="k", alpha=0.7, density=True)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution")

    fig.suptitle(title, fontsize=14, y=1.02)
    if save_path:
        _save(fig, save_path)
    return fig


# ------------------------------------------------------------------
# 3. Model Comparison Bar Chart
# ------------------------------------------------------------------

def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart ranking models by a given metric."""
    sorted_df = results_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_df) * 0.5)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_df)))
    ax.barh(sorted_df["Model"], sorted_df[metric], color=colors, edgecolor="k")
    ax.set_xlabel(metric)
    ax.set_title(title)

    # Annotate values
    for i, (val, name) in enumerate(zip(sorted_df[metric], sorted_df["Model"])):
        ax.text(val + 0.5, i, f"{val:.2f}", va="center", fontsize=9)

    if save_path:
        _save(fig, save_path)
    return fig


# ------------------------------------------------------------------
# 4. Feature Importance
# ------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Top-N feature importance horizontal bar chart."""
    importances = np.asarray(importances).ravel()
    feature_names = np.asarray(feature_names)
    top_n = min(top_n, len(importances))  # guard against small arrays
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(
        feature_names[idx],
        importances[idx],
        color=plt.cm.plasma(np.linspace(0.2, 0.8, top_n)),
        edgecolor="k",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    if save_path:
        _save(fig, save_path)
    return fig


# ------------------------------------------------------------------
# 5. Time Series Actual vs Forecast
# ------------------------------------------------------------------

def plot_time_series_forecast(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Forecast vs Actual",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay actual and predicted as time series."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, label="Actual", lw=1.5)
    ax.plot(dates, y_pred, label="Predicted", lw=1.5, linestyle="--")
    ax.fill_between(dates, y_true, y_pred, alpha=0.15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (Trillion BTU)")
    ax.set_title(title)
    ax.legend()
    if save_path:
        _save(fig, save_path)
    return fig


# ------------------------------------------------------------------
# 6. Cross-Validation Fold Scores
# ------------------------------------------------------------------

def plot_cv_scores(
    fold_scores: list[float],
    metric_name: str = "RMSE",
    title: str = "Cross-Validation Scores",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of per-fold CV scores with mean line."""
    fig, ax = plt.subplots(figsize=(8, 4))
    folds = range(1, len(fold_scores) + 1)
    ax.bar(folds, fold_scores, color="steelblue", edgecolor="k")
    ax.axhline(np.mean(fold_scores), color="red", linestyle="--",
               label=f"Mean = {np.mean(fold_scores):.2f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    if save_path:
        _save(fig, save_path)
    return fig
