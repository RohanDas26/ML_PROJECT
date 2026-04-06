"""
src.models.trainer — Unified Model Training & Selection
========================================================
GridSearchCV & Optuna with TimeSeriesSplit across all model families.
Reports metrics in ORIGINAL units (Trillion BTU) via inverse transform.
"""

import time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor

from src.models.linear_models import get_linear_model_configs
from src.models.tree_models import get_tree_model_configs
from src.models.deep_models import SklearnLSTM
from src.utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = get_logger(__name__)


def tune_with_optuna(model_name: str, X_train: np.ndarray, y_train: np.ndarray, cv) -> tuple:
    """Run Optuna optimization for tree models."""
    def objective(trial):
        if model_name == "XGBoost_Optuna":
            from xgboost import XGBRegressor
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            model = XGBRegressor(random_state=42, **params)
        elif model_name == "LightGBM_Optuna":
            from lightgbm import LGBMRegressor
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            model = LGBMRegressor(random_state=42, **params, verbose=-1)
        else:
            raise ValueError()

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, timeout=120)

    if model_name == "XGBoost_Optuna":
        from xgboost import XGBRegressor
        best_model = XGBRegressor(random_state=42, **study.best_params)
    else:
        from lightgbm import LGBMRegressor
        best_model = LGBMRegressor(random_state=42, **study.best_params, verbose=-1)

    best_model.fit(X_train, y_train)
    return best_model, study.best_params, -study.best_value


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
    quick_mode: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Train every model via GridSearchCV/Optuna with TimeSeriesSplit,
    evaluate on the test set, and return a results table.

    Returns
    -------
    results_df : pd.DataFrame with columns Model, Best_Params, MSE, RMSE,
                 MAE, R2, Train_MSE, Overfit_Ratio, CV_Score, Time, Errors
    best_models : dict mapping model name -> fitted best estimator
    """
    # In quick_mode (for dashboards/demos), use 3 folds and skip expensive models
    effective_splits = 3 if quick_mode else n_splits
    tscv = TimeSeriesSplit(n_splits=effective_splits)

    # Merge all model families
    configs = {}
    configs.update(get_linear_model_configs())
    configs.update(get_tree_model_configs(random_state=random_state))
    
    # Add Deep Learning — skip in quick_mode (8+ param combos × 5 PyTorch folds)
    if not quick_mode:
        configs["LSTM_PyTorch"] = {
            "model": SklearnLSTM(random_state=random_state),
            "params": {
                "hidden_size": [32, 64],
                "num_layers": [1, 2],
                "dropout": [0.2, 0.4]
            }
        }

    results = []
    best_models = {}

    log.info("=" * 70)
    log.info("HYPERPARAMETER TUNING & MODEL SELECTION (GridSearchCV + Optuna + Stacking)")
    log.info("=" * 70)

    # 1. Standard Grid Search
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

        y_pred_scaled = best_model.predict(X_test)
        y_train_pred_scaled = best_model.predict(X_train)

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

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
            "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
            "Train_MSE": train_mse, "Overfit_Ratio": overfit,
            "CV_Score": -grid.best_score_, "Time": elapsed, "Errors": y_test_orig - y_pred,
        })
        log.info("  %s -> RMSE=%.2f R2=%.4f (%.1fs)", name, rmse, r2, elapsed)

    # 2. Optuna Search — skipped in quick_mode (adds ~4 min each)
    if not quick_mode:
        optuna_models = ["XGBoost_Optuna", "LightGBM_Optuna"]
        
        log.info("Pruning feature space for Optuna (Top 20 features via RandomForest Selection)...")
        from sklearn.ensemble import RandomForestRegressor
        rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_selector.fit(X_train, y_train)
        
        importances = rf_selector.feature_importances_
        top_k_idx = np.argsort(importances)[-20:]
        
        X_train_pruned = X_train[:, top_k_idx]
        X_test_pruned  = X_test[:, top_k_idx]
        log.info(f"Reduced feature space from {X_train.shape[1]} to 20 dimensions.")
        
        for name in optuna_models:
            log.info("Tuning: %s …", name)
            t0 = time.time()
            best_model, best_params, cv_score = tune_with_optuna(name, X_train_pruned, y_train, tscv)
            best_models[name] = best_model

            y_pred_scaled       = best_model.predict(X_test_pruned)
            y_train_pred_scaled = best_model.predict(X_train_pruned)
            y_pred      = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

            mse     = mean_squared_error(y_test_orig, y_pred)
            rmse    = np.sqrt(mse)
            mae     = mean_absolute_error(y_test_orig, y_pred)
            r2      = r2_score(y_test_orig, y_pred)
            train_mse = mean_squared_error(y_train_orig, y_train_pred)
            overfit = mse / train_mse if train_mse > 0 else float("inf")
            elapsed = time.time() - t0

            results.append({
                "Model": name, "Best_Params": str(best_params),
                "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
                "Train_MSE": train_mse, "Overfit_Ratio": overfit,
                "CV_Score": cv_score, "Time": elapsed, "Errors": y_test_orig - y_pred,
            })
            log.info("  %s -> RMSE=%.2f R2=%.4f (%.1fs)", name, rmse, r2, elapsed)
    else:
        optuna_models = []

    # 3. Ensemble Stacking — skipped in quick_mode (adds another ~1 min)
    if not quick_mode:
        log.info("Tuning: Ensemble_Stacking …")
        t0 = time.time()

        linear_names = list(get_linear_model_configs().keys())
        tree_names   = list(get_tree_model_configs(random_state=random_state).keys()) + optuna_models

        results_df = pd.DataFrame(results).sort_values("RMSE")
        best_linear_name = results_df[results_df["Model"].isin(linear_names)].iloc[0]["Model"]
        best_tree_name   = results_df[results_df["Model"].isin(tree_names)].iloc[0]["Model"]

        estimators = [
            ('linear', best_models[best_linear_name]),
            ('tree',   best_models[best_tree_name])
        ]

        from sklearn.linear_model import Ridge
        stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
        stacking_model.fit(X_train, y_train)
        best_models["Ensemble_Stacking"] = stacking_model

        y_pred_scaled       = stacking_model.predict(X_test)
        y_train_pred_scaled = stacking_model.predict(X_train)
        y_pred      = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        mse     = mean_squared_error(y_test_orig, y_pred)
        rmse    = np.sqrt(mse)
        mae     = mean_absolute_error(y_test_orig, y_pred)
        r2      = r2_score(y_test_orig, y_pred)
        train_mse = mean_squared_error(y_train_orig, y_train_pred)
        overfit = mse / train_mse if train_mse > 0 else float("inf")
        elapsed = time.time() - t0

        results.append({
            "Model": "Ensemble_Stacking",
            "Best_Params": f"Linear:{best_linear_name} + Tree:{best_tree_name}",
            "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
            "Train_MSE": train_mse, "Overfit_Ratio": overfit,
            "CV_Score": None, "Time": elapsed, "Errors": y_test_orig - y_pred,
        })
        log.info("  Ensemble_Stacking -> RMSE=%.2f R2=%.4f (%.1fs)", rmse, r2, elapsed)

    results_df = pd.DataFrame(results).sort_values("MSE").reset_index(drop=True)
    log.info("Best absolute model: %s (RMSE=%.2f, R²=%.4f)",
             results_df.iloc[0]["Model"],
             results_df.iloc[0]["RMSE"],
             results_df.iloc[0]["R2"])

    return results_df, best_models
