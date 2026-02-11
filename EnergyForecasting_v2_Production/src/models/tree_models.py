"""
src.models.tree_models — Tree & Boosting Model Definitions
===========================================================
RandomForest, GradientBoosting, XGBoost, LightGBM with their grids.
Gracefully skips XGBoost / LightGBM if not installed.
"""

import warnings
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

from src.utils.logger import get_logger

log = get_logger(__name__)

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


def get_tree_model_configs(random_state: int = 42) -> dict:
    """
    Return a dict of ``{name: {model, params}}`` for tree / boosting / instance models.
    """
    configs = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=random_state),
            "params": {
                "n_estimators": [20, 50],
                "max_depth": [3, 5, 7],
                "min_samples_leaf": [5, 10],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=random_state),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "params": {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"],
            },
        },
    }

    if HAS_XGBOOST:
        configs["XGBoost"] = {
            "model": xgb.XGBRegressor(random_state=random_state, verbosity=0),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        }
        log.info("XGBoost available [OK]")
    else:
        log.warning("XGBoost not installed -- skipping")

    if HAS_LIGHTGBM:
        configs["LightGBM"] = {
            "model": lgb.LGBMRegressor(random_state=random_state, verbosity=-1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        }
        log.info("LightGBM available [OK]")
    else:
        log.warning("LightGBM not installed -- skipping")

    return configs
