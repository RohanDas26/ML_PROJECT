"""
src.evaluation.diagnostics — Residual & Model Diagnostics
==========================================================
Ljung-Box, Durbin-Watson, normality tests, and stationarity checks.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

log = get_logger(__name__)


def ljung_box_test(residuals: np.ndarray, lags: int = 20) -> dict:
    """
    Ljung-Box test for residual autocorrelation.

    H0: residuals are independently distributed (no autocorrelation)
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        lb_stat = float(result["lb_stat"].values[0])
        lb_pvalue = float(result["lb_pvalue"].values[0])
    except ImportError:
        log.warning("statsmodels not available — skipping Ljung-Box test")
        lb_stat, lb_pvalue = float("nan"), float("nan")

    return {"LB_Statistic": lb_stat, "LB_P_Value": lb_pvalue,
            "No_Autocorrelation": lb_pvalue > 0.05}


def durbin_watson(residuals: np.ndarray) -> float:
    """
    Durbin-Watson statistic for first-order autocorrelation.

    Values near 2 -> no autocorrelation.
    Values near 0 -> positive autocorrelation.
    Values near 4 -> negative autocorrelation.
    """
    diff = np.diff(residuals)
    dw = float(np.sum(diff ** 2) / np.sum(residuals ** 2))
    return dw


def shapiro_wilk_normality(residuals: np.ndarray) -> dict:
    """Test residuals for normality (Shapiro-Wilk)."""
    # Shapiro-Wilk can be slow for n > 5000; subsample if needed
    sample = residuals[:5000] if len(residuals) > 5000 else residuals
    sw_stat, sw_pvalue = stats.shapiro(sample)
    return {
        "SW_Statistic": float(sw_stat),
        "SW_P_Value": float(sw_pvalue),
        "Normal_Residuals": sw_pvalue > 0.05,
    }


def residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Full residual analysis: autocorrelation, normality, heteroscedasticity proxy.
    """
    residuals = y_true - y_pred

    report = {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "durbin_watson": durbin_watson(residuals),
    }

    report.update(ljung_box_test(residuals))
    report.update(shapiro_wilk_normality(residuals))

    log.info("Residual diagnostics: mean=%.2f, DW=%.3f, LB_p=%.4f, SW_p=%.4f",
             report["mean_residual"], report["durbin_watson"],
             report.get("LB_P_Value", float("nan")),
             report.get("SW_P_Value", float("nan")))

    return report


def stationarity_tests(series: pd.Series) -> dict:
    """
    Run ADF and KPSS stationarity tests on a series.

    Returns
    -------
    dict with ADF / KPSS statistics, p-values, and verdicts
    """
    report = {}

    try:
        from statsmodels.tsa.stattools import adfuller, kpss

        # ADF test (H0: unit root present -> non-stationary)
        adf_result = adfuller(series.dropna(), autolag="AIC")
        report["ADF_Statistic"] = float(adf_result[0])
        report["ADF_P_Value"] = float(adf_result[1])
        report["ADF_Stationary"] = adf_result[1] < 0.05

        # KPSS test (H0: series is stationary)
        kpss_result = kpss(series.dropna(), regression="ct", nlags="auto")
        report["KPSS_Statistic"] = float(kpss_result[0])
        report["KPSS_P_Value"] = float(kpss_result[1])
        report["KPSS_Stationary"] = kpss_result[1] > 0.05

        # Combined verdict
        report["Stationary"] = report["ADF_Stationary"] and report["KPSS_Stationary"]

        log.info("Stationarity: ADF p=%.4f (%s)  KPSS p=%.4f (%s)  -> %s",
                 report["ADF_P_Value"],
                 "stationary" if report["ADF_Stationary"] else "non-stationary",
                 report["KPSS_P_Value"],
                 "stationary" if report["KPSS_Stationary"] else "non-stationary",
                 "STATIONARY" if report["Stationary"] else "NON-STATIONARY")

    except ImportError:
        log.warning("statsmodels not available — skipping stationarity tests")
        report["error"] = "statsmodels not installed"

    return report
