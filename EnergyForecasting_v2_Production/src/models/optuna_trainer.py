"""
src.models.optuna_trainer — Bayesian Hyperparameter Optimization
=================================================================
Uses Optuna TPE sampler with pruning for each model family.
Reports all metrics in original Trillion BTU units.
"""

import time
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, OrthogonalMatchingPursuit
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.logger import get_logger

log = get_logger(__name__)

# Silence Optuna's internal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# -----------------------------------------------------------------------
# Objective functions for each model family
# -----------------------------------------------------------------------

def _ridge_objective(trial, X, y, tscv):
    alpha = trial.suggest_float("alpha", 1e-4, 1000.0, log=True)
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _lasso_objective(trial, X, y, tscv):
    alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
    model = Lasso(alpha=alpha, max_iter=20_000)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _elasticnet_objective(trial, X, y, tscv):
    alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _omp_objective(trial, X, y, tscv):
    n_features = X.shape[1]
    n_nonzero = trial.suggest_int("n_nonzero_coefs", 3, min(n_features, 40))
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _rf_objective(trial, X, y, tscv):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
        random_state=42, n_jobs=-1,
    )
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _gbm_objective(trial, X, y, tscv):
    n_estimators = trial.suggest_int("n_estimators", 30, 300)
    max_depth = trial.suggest_int("max_depth", 2, 8)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 30)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        min_samples_leaf=min_samples_leaf, random_state=42,
    )
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _extra_trees_objective(trial, X, y, tscv):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 30)
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1,
    )
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _knn_objective(trial, X, y, tscv):
    n_neighbors = trial.suggest_int("n_neighbors", 2, 30)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _svr_objective(trial, X, y, tscv):
    C = trial.suggest_float("C", 0.01, 100.0, log=True)
    epsilon = trial.suggest_float("epsilon", 0.001, 1.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    model = SVR(C=C, epsilon=epsilon, kernel=kernel)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _xgb_objective(trial, X, y, tscv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


def _lgbm_objective(trial, X, y, tscv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    return -scores.mean()


# -----------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------

def _build_model_registry():
    """Map model names to their Optuna objective functions and final-model builders."""
    registry = {
        "Ridge": {"objective": _ridge_objective, "builder": lambda p: Ridge(**p)},
        "Lasso": {"objective": _lasso_objective, "builder": lambda p: Lasso(max_iter=20_000, **p)},
        "ElasticNet": {"objective": _elasticnet_objective, "builder": lambda p: ElasticNet(max_iter=20_000, **p)},
        "OMP": {"objective": _omp_objective, "builder": lambda p: OrthogonalMatchingPursuit(**p)},
        "RandomForest": {"objective": _rf_objective, "builder": lambda p: RandomForestRegressor(random_state=42, n_jobs=-1, **p)},
        "GradientBoosting": {"objective": _gbm_objective, "builder": lambda p: GradientBoostingRegressor(random_state=42, **p)},
        "ExtraTrees": {"objective": _extra_trees_objective, "builder": lambda p: ExtraTreesRegressor(random_state=42, n_jobs=-1, **p)},
        "KNN": {"objective": _knn_objective, "builder": lambda p: KNeighborsRegressor(**p)},
        "SVR": {"objective": _svr_objective, "builder": lambda p: SVR(**p)},
    }

    if HAS_XGBOOST:
        registry["XGBoost"] = {
            "objective": _xgb_objective,
            "builder": lambda p: xgb.XGBRegressor(random_state=42, verbosity=0, **p),
        }
        log.info("XGBoost available [OK]")

    if HAS_LIGHTGBM:
        registry["LightGBM"] = {
            "objective": _lgbm_objective,
            "builder": lambda p: lgb.LGBMRegressor(random_state=42, verbosity=-1, **p),
        }
        log.info("LightGBM available [OK]")

    return registry


# -----------------------------------------------------------------------
# Main Optuna training loop
# -----------------------------------------------------------------------

def train_all_models_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train_orig: np.ndarray,
    y_test_orig: np.ndarray,
    scaler_y,
    *,
    n_splits: int = 5,
    n_trials: int = 80,
    timeout_per_model: int = 120,
) -> tuple[pd.DataFrame, dict]:
    """
    Train every model via Optuna Bayesian optimization with TimeSeriesSplit CV.

    Parameters
    ----------
    n_trials : max number of Optuna trials per model
    timeout_per_model : max seconds per model optimization

    Returns
    -------
    results_df : pd.DataFrame with all metrics per model
    best_models : dict mapping model name -> fitted best estimator
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    registry = _build_model_registry()

    results = []
    best_models = {}

    log.info("=" * 70)
    log.info("OPTUNA BAYESIAN OPTIMIZATION  (%d models, %d trials each, %d-fold TSCV)",
             len(registry), n_trials, n_splits)
    log.info("=" * 70)

    for name, spec in registry.items():
        log.info("Optimizing: %s ...", name)
        t0 = time.time()

        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        # Optimize
        study.optimize(
            lambda trial: spec["objective"](trial, X_train, y_train, tscv),
            n_trials=n_trials,
            timeout=timeout_per_model,
            show_progress_bar=False,
        )

        best_params = study.best_params
        log.info("  Best params: %s (CV MSE=%.6f)", best_params, study.best_value)

        # Build final model with best params and fit on full training set
        best_model = spec["builder"](best_params)
        best_model.fit(X_train, y_train)
        best_models[name] = best_model

        # Predictions -> inverse transform to original units
        y_pred_scaled = best_model.predict(X_test)
        y_train_pred_scaled = best_model.predict(X_train)

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        # Metrics in original Trillion BTU units
        mse = mean_squared_error(y_test_orig, y_pred)
        rmse_val = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        train_mse = mean_squared_error(y_train_orig, y_train_pred)
        overfit = mse / train_mse if train_mse > 0 else float("inf")
        elapsed = time.time() - t0

        # MAPE
        eps = 1e-8
        mape_val = float(np.mean(np.abs((y_test_orig - y_pred) / (np.abs(y_test_orig) + eps))) * 100)

        results.append({
            "Model": name,
            "Best_Params": str(best_params),
            "MSE": mse,
            "RMSE": rmse_val,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape_val,
            "Train_MSE": train_mse,
            "Overfit_Ratio": overfit,
            "CV_MSE": study.best_value,
            "N_Trials": len(study.trials),
            "Time_Seconds": round(elapsed, 1),
            "Errors": y_test_orig - y_pred,
        })

        log.info("  %s -> RMSE=%.2f  R2=%.4f  MAPE=%.2f%%  (%.1fs, %d trials)",
                 name, rmse_val, r2, mape_val, elapsed, len(study.trials))

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    log.info("-" * 70)
    log.info("BEST MODEL: %s  (RMSE=%.2f, R2=%.4f, MAPE=%.2f%%)",
             results_df.iloc[0]["Model"],
             results_df.iloc[0]["RMSE"],
             results_df.iloc[0]["R2"],
             results_df.iloc[0]["MAPE"])
    log.info("-" * 70)

    return results_df, best_models
