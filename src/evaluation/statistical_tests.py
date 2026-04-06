"""
src.evaluation.statistical_tests — Forecast Comparison Tests
=============================================================
Diebold-Mariano and related tests for statistically rigorous
model comparison on time-series data.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

log = get_logger(__name__)


# -----------------------------------------------------------------------
# Diebold-Mariano Test
# -----------------------------------------------------------------------

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: Both forecasts have equal accuracy
    H1: Forecast 1 is more accurate than Forecast 2

    Parameters
    ----------
    e1 : errors from model 1 (the better model)
    e2 : errors from model 2 (the baseline)
    h  : forecast horizon (determines HAC bandwidth)
    loss : 'squared' (MSE) or 'absolute' (MAE)

    Returns
    -------
    (dm_statistic, p_value)
    """
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    mean_d = np.mean(d)

    # Long-run variance (Newey-West HAC estimator)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = gamma_0
    for k in range(1, h):
        if len(d) > k:
            gamma_k = np.cov(d[:-k], d[k:], ddof=0)[0, 1]
            gamma_sum += 2 * gamma_k

    # DM statistic
    dm_stat = mean_d / np.sqrt(gamma_sum / n) if gamma_sum > 0 else 0.0

    # Two-sided p-value
    p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))

    return float(dm_stat), p_value


# -----------------------------------------------------------------------
# Batch DM Tests vs. Baseline
# -----------------------------------------------------------------------

def run_dm_tests(
    results_df: pd.DataFrame,
    baseline_name: str = "Ridge",
) -> pd.DataFrame:
    """
    Run Diebold-Mariano tests for every model vs. a baseline.

    Parameters
    ----------
    results_df : output of ``train_all_models()`` — must have 'Model' and 'Errors' columns
    baseline_name : model to use as baseline

    Returns
    -------
    pd.DataFrame with columns: Model, DM_Statistic, P_Value, Significance, Better
    """
    baseline_row = results_df[results_df["Model"] == baseline_name]
    if baseline_row.empty:
        raise ValueError(f"Baseline model '{baseline_name}' not found in results")

    baseline_errors = baseline_row.iloc[0]["Errors"]

    records = []
    for _, row in results_df.iterrows():
        if row["Model"] == baseline_name:
            continue

        dm_stat, p_val = diebold_mariano_test(baseline_errors, row["Errors"])

        sig = ""
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"

        records.append({
            "Model": row["Model"],
            "DM_Statistic": round(dm_stat, 4),
            "P_Value": round(p_val, 4),
            "Significance": sig,
            "Better": "Yes" if dm_stat > 0 and p_val < 0.05 else "No",
        })

        log.info("  DM(%s vs %s): stat=%.3f  p=%.4f %s",
                 row["Model"], baseline_name, dm_stat, p_val, sig)

    return pd.DataFrame(records)
