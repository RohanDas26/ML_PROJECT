"""
src.data.preprocessor — Scaling & Train/Test Splitting
=======================================================
Handles standardization, train/test temporal splitting, and inverse transforms.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

log = get_logger(__name__)


class TimeSeriesPreprocessor:
    """
    Wraps StandardScaler with temporal-aware train/test splitting.

    Attributes
    ----------
    scaler_X : fitted StandardScaler for features
    scaler_y : fitted StandardScaler for target
    split_idx : index separating train from test
    """

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.split_idx: int | None = None

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    def split(
        self,
        features_df: pd.DataFrame,
        test_fraction: float = 0.20,
    ) -> dict:
        """
        Temporal train/test split (no shuffling).

        Parameters
        ----------
        features_df : output of ``create_features()``
        test_fraction : fraction of samples to hold out

        Returns
        -------
        dict with keys: X_train, X_test, y_train, y_test,
                        y_train_orig, y_test_orig, feature_cols
        """
        from src.data.feature_engineering import get_feature_columns

        feature_cols = get_feature_columns(features_df)
        X = features_df[feature_cols].values
        y = features_df["target"].values

        self.split_idx = int(len(X) * (1 - test_fraction))
        X_train, X_test = X[: self.split_idx], X[self.split_idx:]
        y_train, y_test = y[: self.split_idx], y[self.split_idx:]

        log.info("Split: %d train / %d test (%.0f%% hold-out)",
                 len(X_train), len(X_test), test_fraction * 100)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_orig": y_train.copy(),
            "y_test_orig": y_test.copy(),
            "feature_cols": feature_cols,
        }

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------
    def fit_transform(self, data: dict) -> dict:
        """
        Fit scalers on training data and transform both train and test.

        Modifies ``data`` in-place and returns it.
        """
        data["X_train"] = self.scaler_X.fit_transform(data["X_train"])
        data["X_test"] = self.scaler_X.transform(data["X_test"])

        data["y_train"] = self.scaler_y.fit_transform(
            data["y_train"].reshape(-1, 1)
        ).ravel()
        data["y_test"] = self.scaler_y.transform(
            data["y_test"].reshape(-1, 1)
        ).ravel()

        log.info("Standardized features and target (fitted on train)")
        return data

    # ------------------------------------------------------------------
    # Stationarity & Differencing
    # ------------------------------------------------------------------
    def check_stationarity(self, series: pd.Series, p_value_threshold: float = 0.05) -> bool:
        """
        Run Augmented Dickey-Fuller test to check for stationarity.
        Returns True if stationary (p-value < threshold), False otherwise.
        """
        from statsmodels.tsa.stattools import adfuller
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            return p_value < p_value_threshold
        except Exception:
            return False  # Assume non-stationary on error

    def difference(self, series: pd.Series) -> tuple[pd.Series, float | None]:
        """
        Apply first-order differencing.
        Returns (differenced_series, first_value_for_reconstruction).
        """
        diff_series = series.diff().dropna()
        first_val = series.iloc[0]
        return diff_series, first_val

    def inverse_difference(self, diff_series: np.ndarray, first_val: float) -> np.ndarray:
        """
        Reconstruct original series from differenced data:
        y_t = y_{t-1} + diff_t
        """
        # np.r_ concatenates along the first axis
        # cumsum recovers the cumulative changes
        reconstructed = np.r_[first_val, diff_series].cumsum()
        return reconstructed[1:]  # Return aligned series (excluding the seed)

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------
    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to original Trillion BTU units."""
        return self.scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()
