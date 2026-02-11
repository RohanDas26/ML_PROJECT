"""
src.data.feature_engineering — Leak-Free Feature Construction
==============================================================
Implements all temporal, harmonic, autoregressive, rolling, cross-sector,
and momentum features with STRICT look-ahead prevention.

Rules enforced:
  • All rolling windows are shifted by ≥1 period
  • Cross-sector features use only lagged values
  • No concurrent-timestep information in any feature
"""

import numpy as np
import pandas as pd

from src.data.loader import ALL_SECTORS
from src.utils.logger import get_logger

log = get_logger(__name__)


def create_features(
    df: pd.DataFrame,
    target_sector: str,
    *,
    target_lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    ema_spans: list[int] | None = None,
    harmonic_periods: list[int] | None = None,
    cross_sector_lags: list[int] | None = None,
    use_differencing: bool = False,
    seasonal_flags: bool = True,
    momentum: bool = True,
    exog_df: pd.DataFrame | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature matrix for a given target sector.

    Parameters
    ----------
    df : DataFrame with columns Month + all sectors
    target_sector : one of Residential / Commercial / Industrial / Transportation
    target_lags : lag orders for autoregressive features  (default: 1..24)
    rolling_windows : window sizes for rolling mean/std   (default: 3,6,12)
    ema_spans : exponential-moving-average spans           (default: 3,6,12)
    harmonic_periods : cycle lengths in months              (default: 12,6,3)
    cross_sector_lags : lags for other-sector features     (default: 1,12)
    use_differencing : first-order differencing on target
    seasonal_flags : binary winter/summer/spring/fall flags
    momentum : YoY, MoM, and k-month momentum features
    exog_df : Optional DataFrame with exogenous vars (Month + vars).
              Will be merged and LAGGED to prevent leakage.
    drop_na : whether to drop rows with NaN values (default: True).
              Set to False for inference/forecasting where target is NaN.

    Returns
    -------
    pd.DataFrame with columns: Month, target, + all feature columns
    """
    # Defaults
    if target_lags is None:
        target_lags = [1, 2, 3, 6, 12, 13, 24]
    if rolling_windows is None:
        rolling_windows = [3, 6, 12]
    if ema_spans is None:
        ema_spans = [3, 6, 12]
    if harmonic_periods is None:
        harmonic_periods = [12, 6, 3]
    if cross_sector_lags is None:
        cross_sector_lags = [1, 12]

    if target_sector not in ALL_SECTORS:
        raise ValueError(f"target_sector must be one of {ALL_SECTORS}, got '{target_sector}'")

    feat = pd.DataFrame()
    feat["Month"] = df["Month"]
    target = df[target_sector].copy()

    # --- Target (optionally differenced) ---------------------------------
    if use_differencing:
        feat["target"] = target.diff()
        log.info("Applied first-order differencing to target")
    else:
        feat["target"] = target

    # --- Temporal features -----------------------------------------------
    month_num = df["Month"].dt.month
    feat["year"] = df["Month"].dt.year
    feat["month"] = month_num
    feat["quarter"] = df["Month"].dt.quarter
    log.info("Temporal features: year, month, quarter")

    # --- Multi-harmonic seasonality --------------------------------------
    for period in harmonic_periods:
        k = 12 / period  # frequency multiplier
        feat[f"month_sin_{period}"] = np.sin(2 * k * np.pi * month_num / 12)
        feat[f"month_cos_{period}"] = np.cos(2 * k * np.pi * month_num / 12)
    log.info("Harmonic features: periods %s", harmonic_periods)

    # --- Autoregressive lags ---------------------------------------------
    for lag in target_lags:
        feat[f"target_lag_{lag}"] = target.shift(lag)
    log.info("Autoregressive lags: %s", target_lags)

    # --- Rolling statistics (shifted by 1 to prevent look-ahead) ---------
    shifted_target = target.shift(1)
    for w in rolling_windows:
        feat[f"rolling_mean_{w}"] = shifted_target.rolling(window=w).mean()
        feat[f"rolling_std_{w}"] = shifted_target.rolling(window=w).std()
    log.info("Rolling mean/std (shifted): windows %s", rolling_windows)

    # --- Exponential moving averages (shifted) ---------------------------
    for span in ema_spans:
        feat[f"ema_{span}"] = shifted_target.ewm(span=span).mean()
    log.info("EMA features (shifted): spans %s", ema_spans)

    # --- Total energy lags -----------------------------------------------
    total_energy = df[ALL_SECTORS].sum(axis=1)
    for lag in cross_sector_lags:
        feat[f"total_energy_lag_{lag}"] = total_energy.shift(lag)
    log.info("Total energy lags: %s", cross_sector_lags)

    # --- Other-sector lags -----------------------------------------------
    other_sectors = [s for s in ALL_SECTORS if s != target_sector]
    for sector in other_sectors:
        for lag in cross_sector_lags:
            feat[f"{sector.lower()}_lag_{lag}"] = df[sector].shift(lag)
    log.info("Cross-sector lags (%s): %s", other_sectors, cross_sector_lags)

    # --- Momentum / change features --------------------------------------
    if momentum:
        feat["yoy_change"] = (target.shift(1) - target.shift(13)) / target.shift(13)
        feat["mom_change"] = (target.shift(1) - target.shift(2)) / target.shift(2)
        feat["momentum_3"] = target.shift(1) - target.shift(4)
        feat["momentum_6"] = target.shift(1) - target.shift(7)
        log.info("Momentum features: YoY, MoM, 3M, 6M")

    # --- Seasonal binary flags -------------------------------------------
    if seasonal_flags:
        feat["is_winter"] = month_num.isin([12, 1, 2]).astype(int)
        feat["is_summer"] = month_num.isin([6, 7, 8]).astype(int)
        feat["is_spring"] = month_num.isin([3, 4, 5]).astype(int)
        feat["is_fall"] = month_num.isin([9, 10, 11]).astype(int)
        log.info("Seasonal flags: winter, summer, spring, fall")

    # --- Exogenous Variables (Lagged) ------------------------------------
    if exog_df is not None:
        log.info("Merging exogenous variables...")
        # Merge on Month
        feat = feat.merge(exog_df, on="Month", how="left")
        
        # Identify exogenous columns (those not already in feat or created above)
        # Actually easiest is to assume everything from exog_df minus Month is a feature
        exog_cols = [c for c in exog_df.columns if c != "Month"]
        
        for col in exog_cols:
            # Create lags to prevent leakage. Direct concurrent value is FORBIDDEN.
            # We use lags 1 (t-1 available at prediction time t) and 12 (seasonality)
            feat[f"{col}_lag1"] = feat[col].shift(1)
            feat[f"{col}_lag12"] = feat[col].shift(12)
            
            # YoY % change for non-temperature vars
            if "HDD" not in col and "CDD" not in col and "temp" not in col:
                 feat[f"{col}_yoy"] = feat[col].pct_change(12).shift(1) # Shift 1 to be safe
            
            # Drop the raw concurrent column to enforce leakage safety
            feat.drop(columns=[col], inplace=True)
            
        log.info(f"Added lagged exogenous features for: {exog_cols}")

    # --- Drop NaN rows created by lagging --------------------------------
    n_before = len(feat)
    if drop_na:
        feat = feat.dropna().reset_index(drop=True)
    else:
        # Reset index but keep NaNs
        feat = feat.reset_index(drop=True)
        
    n_after = len(feat)

    feature_cols = [c for c in feat.columns if c not in ("Month", "target")]
    log.info("Feature matrix: %d features × %d samples (dropped %d lag rows)",
             len(feature_cols), n_after, n_before - n_after)

    return feat


def select_features_lasso(df: pd.DataFrame, target_col: str = "target", min_features: int = 10) -> list[str]:
    """
    Selects the most predictive features using LassoCV (L1 regularization).
    Retains at least `min_features` features.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    
    # Prepare X and y
    feature_cols = [c for c in df.columns if c not in ("Month", target_col)]
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Scale features (crucial for Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit LassoCV (automatically finds best alpha)
    # n_alphas=100 usually enough. cv=5 for robustness
    model = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000).fit(X_scaled, y)
    
    # Get coefficients
    coefs = np.abs(model.coef_)
    
    # Sort features by importance
    importance = pd.DataFrame({"feature": feature_cols, "coef": coefs})
    importance = importance.sort_values("coef", ascending=False)
    
    # Select features with non-zero coefficients
    selected = importance[importance["coef"] > 1e-5]["feature"].tolist()
    
    # If too few selected, take top N
    if len(selected) < min_features:
        log.warning(f"Lasso selected only {len(selected)} features. Enforcing min {min_features}.")
        selected = importance.head(min_features)["feature"].tolist()
        
    log.info(f"Lasso selection: kept {len(selected)}/{len(feature_cols)} features.")
    return selected


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes Month and target)."""
    return [c for c in df.columns if c not in ("Month", "target")]
