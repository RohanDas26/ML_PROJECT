"""
src.models.trainer — Unified Model Training & Selection
========================================================
GridSearchCV with TimeSeriesSplit across all model families.
Reports metrics in ORIGINAL units (Trillion BTU) via inverse transform.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.linear_models import get_linear_model_configs
from src.models.tree_models import get_tree_model_configs
from src.utils.logger import get_logger

log = get_logger(__name__)


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train_orig: np.ndarray,
    y_test_orig: np.ndarray,
    scaler_y,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Train every model via GridSearchCV with TimeSeriesSplit,
    evaluate on the test set, and return a results table.

    Returns
    -------
    results_df : pd.DataFrame with columns Model, Best_Params, MSE, RMSE,
                 MAE, R2, Train_MSE, Overfit_Ratio, CV_Score, Time, Errors
    best_models : dict mapping model name -> fitted best estimator
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Merge all model families
    configs = {}
    configs.update(get_linear_model_configs())
    configs.update(get_tree_model_configs(random_state=random_state))

    results = []
    best_models = {}

    log.info("=" * 70)
    log.info("HYPERPARAMETER TUNING & MODEL SELECTION  (%d models, %d-fold TSCV)",
             len(configs), n_splits)
    log.info("=" * 70)

    for name, cfg in configs.items():
        log.info("Tuning: %s …", name)
        t0 = time.time()

        grid = GridSearchCV(
            cfg["model"],
            cfg["params"],
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[name] = best_model

        # Predictions (scaled) -> inverse-transform to original units
        y_pred_scaled = best_model.predict(X_test)
        y_train_pred_scaled = best_model.predict(X_train)

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        # Metrics in original Trillion BTU units
        mse = mean_squared_error(y_test_orig, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        train_mse = mean_squared_error(y_train_orig, y_train_pred)
        overfit = mse / train_mse if train_mse > 0 else float("inf")
        elapsed = time.time() - t0

        results.append({
            "Model": name,
            "Best_Params": str(grid.best_params_),
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Train_MSE": train_mse,
            "Overfit_Ratio": overfit,
            "CV_Score": -grid.best_score_,
            "Time": elapsed,
            "Errors": y_test_orig - y_pred,
        })

        log.info("  %s -> RMSE=%.2f  R2=%.4f  (%.1fs)  params=%s",
                 name, rmse, r2, elapsed, grid.best_params_)

    results_df = pd.DataFrame(results).sort_values("MSE").reset_index(drop=True)
    log.info("Best model: %s  (RMSE=%.2f, R²=%.4f)",
             results_df.iloc[0]["Model"],
             results_df.iloc[0]["RMSE"],
             results_df.iloc[0]["R2"])

    return results_df, best_models
