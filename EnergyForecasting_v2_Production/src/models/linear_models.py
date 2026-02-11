"""
src.models.linear_models — Linear Model Definitions
=====================================================
Ridge, Lasso, ElasticNet, OMP with their hyperparameter grids.
"""

from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    OrthogonalMatchingPursuit,
)


def get_linear_model_configs() -> dict:
    """
    Return a dict of ``{name: {model, params}}`` for linear models.

    Compatible with ``sklearn.model_selection.GridSearchCV``.
    """
    return {
        "Ridge": {
            "model": Ridge(),
            "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "Lasso": {
            "model": Lasso(max_iter=10_000),
            "params": {"alpha": [0.001, 0.01, 0.1, 1.0]},
        },
        "ElasticNet": {
            "model": ElasticNet(max_iter=10_000),
            "params": {
                "alpha": [0.001, 0.01, 0.1],
                "l1_ratio": [0.2, 0.5, 0.8],
            },
        },
        "OMP": {
            "model": OrthogonalMatchingPursuit(),
            "params": {"n_nonzero_coefs": [5, 10, 15, 20, 25]},
        },
    }
