"""
src.data.loader — Data Loading & Validation
=============================================
Handles raw Excel ingestion, schema validation, and temporal integrity checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

# Canonical column mapping from raw Excel headers
_RAW_COL_MAP = {
    "Total Energy Consumed by the Residential Sector": "Residential",
    "Total Energy Consumed by the Commercial Sector": "Commercial",
    "Total Energy Consumed by the Industrial Sector": "Industrial",
    "Total Energy Consumed by the Transportation Sector": "Transportation",
}

ALL_SECTORS = list(_RAW_COL_MAP.values())


def load_raw_data(filepath: str | Path,
                  sheet_name: str = "Sheet1",
                  skiprows: int = 1) -> pd.DataFrame:
    """
    Load the EIA energy dataset from Excel.

    Parameters
    ----------
    filepath : path to the .xlsx file
    sheet_name : Excel sheet to read
    skiprows : rows to skip (metadata header)

    Returns
    -------
    pd.DataFrame with columns: Month, Residential, Commercial, Industrial, Transportation
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df_raw = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=skiprows)
    df_raw = df_raw.iloc[1:].reset_index(drop=True)
    df_raw.columns = ["Month"] + list(df_raw.columns[1:])

    # Select and rename relevant columns
    keep_cols = ["Month"] + list(_RAW_COL_MAP.keys())
    missing = [c for c in keep_cols if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")

    df = df_raw[keep_cols].copy()
    df = df.rename(columns=_RAW_COL_MAP)

    # Parse datetime
    df["Month"] = pd.to_datetime(df["Month"])

    # Coerce sector columns to numeric
    for col in ALL_SECTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df = df.sort_values("Month").reset_index(drop=True)

    log.info("Loaded %d rows  (%s -> %s)", len(df),
             df["Month"].min().strftime("%Y-%m"),
             df["Month"].max().strftime("%Y-%m"))
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run quality checks and return a validation report.

    Returns
    -------
    dict with keys: n_rows, date_range, missing_pct, duplicates,
                    monotonic, frequency, outlier_counts
    """
    report = {}

    report["n_rows"] = len(df)
    report["date_range"] = (str(df["Month"].min()), str(df["Month"].max()))

    # Missing values
    report["missing_pct"] = df.isnull().mean().to_dict()

    # Duplicate dates
    report["duplicates"] = int(df["Month"].duplicated().sum())
    if report["duplicates"] > 0:
        log.warning("Found %d duplicate dates!", report["duplicates"])

    # Monotonic index
    report["monotonic"] = bool(df["Month"].is_monotonic_increasing)
    if not report["monotonic"]:
        log.warning("Date column is NOT monotonically increasing!")

    # Frequency check
    diffs = df["Month"].diff().dropna()
    median_diff = diffs.median()
    report["frequency"] = str(median_diff)

    # IQR-based outlier detection per sector
    outlier_counts = {}
    for col in ALL_SECTORS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_counts[col] = n_outliers
    report["outlier_counts"] = outlier_counts

    # Log summary
    total_outliers = sum(outlier_counts.values())
    log.info("Validation: %d rows, %d duplicates, %d total outliers",
             report["n_rows"], report["duplicates"], total_outliers)

    return report
