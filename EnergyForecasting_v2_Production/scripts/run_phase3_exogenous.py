"""
run_phase3_exogenous.py — Phase 3: Exogenous Variables
=======================================================
Downloads external data (GDP proxy, Oil prices, CPI, HDD/CDD) from FRED,
integrates with existing energy dataset, and re-runs Optuna evaluation
to measure improvement from exogenous features.

Data Sources:
  - INDPRO: Industrial Production Index (monthly, 1919-present) [FRED]
  - MCOILWTICO: WTI Crude Oil price (monthly, 1986-present) [FRED]
  - CPIAUCSL: Consumer Price Index (monthly, 1947-present) [FRED]
  - HDD/CDD: Computed from seasonal temperature model (synthetic but realistic)

Usage:
    python run_phase3_exogenous.py
"""

import sys
import warnings
import json
import time
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from sklearn.metrics import mean_squared_error, r2_score
from src.data.loader import load_raw_data, validate_data, ALL_SECTORS
from src.data.feature_engineering import create_features
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.optuna_trainer import train_all_models_optuna
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.statistical_tests import diebold_mariano_test
from src.utils.io import save_json, save_dataframe
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("Results/phase3_exogenous")
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"


# ======================================================================
# DATA FETCHING
# ======================================================================

def fetch_fred_series(series_id: str, start: str = "1973-01-01",
                      end: str = "2025-09-01") -> pd.DataFrame:
    """
    Download a FRED series as CSV via direct URL.
    Returns DataFrame with 'Month' (datetime) and the series column.
    """
    import urllib.request
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}&fq=Monthly&fam=avg"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["Month", series_id]
        df["Month"] = pd.to_datetime(df["Month"])
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df
    except Exception as e:
        print(f"  [WARN] Could not fetch {series_id}: {e}")
        return pd.DataFrame(columns=["Month", series_id])


def compute_hdd_cdd(dates, base_temp: float = 65.0) -> pd.DataFrame:
    """
    Generate realistic monthly HDD/CDD based on a sinusoidal US-average
    temperature model. This approximates NOAA national averages well:
      T(month) ~ 52 + 22*cos(2*pi*(month - 7)/12)
    giving ~34F in Jan, ~74F in Jul.

    HDD = max(0, base - T) * days_in_month
    CDD = max(0, T - base) * days_in_month
    """
    dt_index = pd.DatetimeIndex(dates)
    months = dt_index.month
    # Sinusoidal US average temp (F)
    avg_temp = 52.0 + 22.0 * np.cos(2 * np.pi * (months - 7) / 12.0)
    # Add slow climate trend (~+0.03F/year since 1973)
    years_since_base = (dt_index.year - 1973) + (dt_index.month - 1) / 12.0
    avg_temp = avg_temp + 0.03 * years_since_base
    # Add small random variation
    np.random.seed(42)
    noise = np.random.normal(0, 2.5, len(dt_index))
    avg_temp = avg_temp + noise

    days_in_month = dt_index.days_in_month
    hdd = np.maximum(0, base_temp - avg_temp) * days_in_month
    cdd = np.maximum(0, avg_temp - base_temp) * days_in_month

    return pd.DataFrame({
        "Month": dates.values if hasattr(dates, 'values') else dates,
        "avg_temp_F": np.round(avg_temp, 1),
        "HDD": np.round(hdd, 0).astype(int),
        "CDD": np.round(cdd, 0).astype(int),
    })


def fetch_all_exogenous(dates: pd.DatetimeIndex, log) -> pd.DataFrame:
    """
    Fetch and merge all exogenous data sources.
    Returns a single DataFrame indexed by Month with all external features.
    """
    log.info("Fetching exogenous data from external sources...")

    # 1. Industrial Production Index (proxy for GDP, monthly)
    log.info("  Downloading INDPRO (Industrial Production Index)...")
    df_indpro = fetch_fred_series("INDPRO", "1973-01-01", "2025-09-01")
    log.info("    -> %d rows", len(df_indpro))

    # 2. WTI Crude Oil (monthly avg, starts 1986)
    log.info("  Downloading MCOILWTICO (WTI Crude Oil)...")
    df_oil = fetch_fred_series("MCOILWTICO", "1973-01-01", "2025-09-01")
    log.info("    -> %d rows", len(df_oil))

    # 3. Consumer Price Index (monthly)
    log.info("  Downloading CPIAUCSL (Consumer Price Index)...")
    df_cpi = fetch_fred_series("CPIAUCSL", "1973-01-01", "2025-09-01")
    log.info("    -> %d rows", len(df_cpi))

    # 4. Natural Gas Price (monthly, starts 1997)
    log.info("  Downloading MHHNGSP (Henry Hub Natural Gas)...")
    df_gas = fetch_fred_series("MHHNGSP", "1973-01-01", "2025-09-01")
    log.info("    -> %d rows", len(df_gas))

    # 5. HDD/CDD (computed from temperature model)
    log.info("  Computing HDD/CDD from temperature model...")
    df_hdd_cdd = compute_hdd_cdd(dates)
    log.info("    -> %d rows", len(df_hdd_cdd))

    # 6. Population (monthly, interpolated from annual)
    log.info("  Downloading B230RC0Q173SBEA (US Population, quarterly)...")
    df_pop = fetch_fred_series("B230RC0Q173SBEA", "1973-01-01", "2025-09-01")
    log.info("    -> %d rows", len(df_pop))

    # Merge everything on Month
    base = pd.DataFrame({"Month": dates})
    for df_ext in [df_indpro, df_oil, df_cpi, df_gas, df_hdd_cdd, df_pop]:
        if len(df_ext) > 0:
            base = base.merge(df_ext, on="Month", how="left")

    # Rename columns for clarity
    rename_map = {
        "INDPRO": "industrial_production",
        "MCOILWTICO": "oil_price_wti",
        "CPIAUCSL": "cpi",
        "MHHNGSP": "natgas_price",
        "B230RC0Q173SBEA": "us_population",
    }
    base = base.rename(columns=rename_map)

    # Interpolate missing values (oil starts 1986, gas starts 1997)
    exog_cols = [c for c in base.columns if c != "Month"]
    for col in exog_cols:
        base[col] = base[col].interpolate(method="linear").bfill().ffill()

    log.info("  Exogenous dataset: %d rows x %d columns", len(base), len(exog_cols))
    log.info("  Columns: %s", exog_cols)

    return base


# ======================================================================
# FEATURE ENGINEERING WITH EXOGENOUS
# ======================================================================

def create_features_with_exogenous(
    df: pd.DataFrame,
    exog_df: pd.DataFrame,
    sector: str,
    feat_cfg: dict,
) -> pd.DataFrame:
    """
    Create standard features + exogenous features.
    Adds lagged versions of exogenous variables to prevent leakage.
    """
    # Standard features
    features_df = create_features(
        df, sector,
        target_lags=feat_cfg.get("target_lags"),
        rolling_windows=feat_cfg.get("rolling_windows"),
        ema_spans=feat_cfg.get("ema_spans"),
        harmonic_periods=feat_cfg.get("harmonic_periods"),
        cross_sector_lags=feat_cfg.get("cross_sector_lags"),
        use_differencing=feat_cfg.get("use_differencing", False),
        seasonal_flags=feat_cfg.get("seasonal_flags", True),
        momentum=feat_cfg.get("momentum", True),
    )

    # Merge exogenous data
    merged = features_df.merge(exog_df, on="Month", how="left")

    # Create LAGGED exogenous features (shift by 1 to prevent leakage)
    exog_cols = [c for c in exog_df.columns if c != "Month"]
    for col in exog_cols:
        merged[f"{col}_lag1"] = merged[col].shift(1)
        merged[f"{col}_lag12"] = merged[col].shift(12)
        # Year-over-year change
        if col not in ["HDD", "CDD", "avg_temp_F"]:
            merged[f"{col}_yoy_pct"] = merged[col].pct_change(12) * 100

    # Drop unlagged exogenous (to prevent leakage — only use lagged versions)
    merged = merged.drop(columns=exog_cols, errors="ignore")

    # Drop rows with NaN (from lagging)
    merged = merged.dropna().reset_index(drop=True)

    return merged


# ======================================================================
# EVALUATION PIPELINE
# ======================================================================

def run_sector_evaluation(
    cfg, df, exog_df, sector, log,
    n_trials=50, timeout=60,
):
    """Run Optuna evaluation for one sector with exogenous features."""
    feat_cfg = cfg.get("features", {})

    # --- WITHOUT exogenous ---
    log.info("  [%s] Training WITHOUT exogenous features...", sector)
    features_base = create_features(
        df, sector,
        target_lags=feat_cfg.get("target_lags"),
        rolling_windows=feat_cfg.get("rolling_windows"),
        ema_spans=feat_cfg.get("ema_spans"),
        harmonic_periods=feat_cfg.get("harmonic_periods"),
        cross_sector_lags=feat_cfg.get("cross_sector_lags"),
        use_differencing=feat_cfg.get("use_differencing", False),
        seasonal_flags=feat_cfg.get("seasonal_flags", True),
        momentum=feat_cfg.get("momentum", True),
    )
    preproc_base = TimeSeriesPreprocessor()
    split_cfg = cfg.get("split", {})
    data_base = preproc_base.split(features_base, test_fraction=split_cfg.get("test_fraction", 0.20))
    data_base = preproc_base.fit_transform(data_base)

    results_base, models_base = train_all_models_optuna(
        data_base["X_train"], data_base["y_train"],
        data_base["X_test"], data_base["y_test"],
        data_base["y_train_orig"], data_base["y_test_orig"],
        preproc_base.scaler_y,
        n_splits=5, n_trials=n_trials, timeout_per_model=timeout,
    )

    # --- WITH exogenous ---
    log.info("  [%s] Training WITH exogenous features...", sector)
    features_exog = create_features_with_exogenous(df, exog_df, sector, feat_cfg)
    preproc_exog = TimeSeriesPreprocessor()
    data_exog = preproc_exog.split(features_exog, test_fraction=split_cfg.get("test_fraction", 0.20))
    data_exog = preproc_exog.fit_transform(data_exog)

    results_exog, models_exog = train_all_models_optuna(
        data_exog["X_train"], data_exog["y_train"],
        data_exog["X_test"], data_exog["y_test"],
        data_exog["y_train_orig"], data_exog["y_test_orig"],
        preproc_exog.scaler_y,
        n_splits=5, n_trials=n_trials, timeout_per_model=timeout,
    )

    # Add source labels
    results_base["Features"] = "Baseline"
    results_base["Sector"] = sector
    results_exog["Features"] = "Exogenous"
    results_exog["Sector"] = sector

    return results_base, results_exog, data_base, data_exog


def plot_comparison(results_base, results_exog, sector, save_dir):
    """Plot side-by-side RMSE and R2 comparison."""
    combined = pd.concat([results_base, results_exog], ignore_index=True)

    models = results_base["Model"].values
    base_rmse = results_base.set_index("Model")["RMSE"]
    exog_rmse = results_exog.set_index("Model")["RMSE"]
    base_r2 = results_base.set_index("Model")["R2"]
    exog_r2 = results_exog.set_index("Model")["R2"]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE
    b1 = ax1.bar(x - width / 2, [base_rmse.get(m, 0) for m in models],
                 width, label="Baseline", color="steelblue", edgecolor="k")
    b2 = ax1.bar(x + width / 2, [exog_rmse.get(m, 0) for m in models],
                 width, label="+ Exogenous", color="coral", edgecolor="k")
    ax1.set_ylabel("RMSE (Trillion BTU)")
    ax1.set_title(f"{sector} -- RMSE Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # R2
    b3 = ax2.bar(x - width / 2, [base_r2.get(m, 0) for m in models],
                 width, label="Baseline", color="steelblue", edgecolor="k")
    b4 = ax2.bar(x + width / 2, [exog_r2.get(m, 0) for m in models],
                 width, label="+ Exogenous", color="coral", edgecolor="k")
    ax2.set_ylabel("R2")
    ax2.set_title(f"{sector} -- R2 Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / f"{sector}_exogenous_comparison.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main():
    t0 = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log = get_logger(level="INFO")
    np.random.seed(42)

    log.info("=" * 70)
    log.info("PHASE 3: EXOGENOUS VARIABLES INTEGRATION")
    log.info("=" * 70)

    # Load energy data
    df = load_raw_data(
        cfg["data"]["raw_path"],
        sheet_name=cfg["data"].get("sheet_name", "Sheet1"),
        skiprows=cfg["data"].get("skiprows", 1),
    )
    validate_data(df)

    # Fetch exogenous data
    exog_df = fetch_all_exogenous(df["Month"], log)
    exog_df.to_csv(TABLE_DIR / "exogenous_data.csv", index=False)
    log.info("Exogenous data saved to %s", TABLE_DIR / "exogenous_data.csv")

    # Plot exogenous data overview
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    exog_cols = [c for c in exog_df.columns if c != "Month"]
    for i, col in enumerate(exog_cols[:6]):
        ax = axes[i // 2, i % 2]
        ax.plot(exog_df["Month"], exog_df[col], lw=1.2)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Exogenous Variables (1973-2025)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exogenous_overview.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Run evaluation for each sector
    all_base = []
    all_exog = []
    comparison_rows = []

    for sector in ALL_SECTORS:
        log.info("-" * 50)
        log.info("SECTOR: %s", sector)
        log.info("-" * 50)

        results_base, results_exog, data_base, data_exog = run_sector_evaluation(
            cfg, df, exog_df, sector, log,
            n_trials=50, timeout=60,
        )

        all_base.append(results_base)
        all_exog.append(results_exog)

        # Plot comparison
        plot_comparison(results_base, results_exog, sector, FIG_DIR)

        # Best model comparison
        best_base = results_base.sort_values("RMSE").iloc[0]
        best_exog = results_exog.sort_values("RMSE").iloc[0]

        comparison_rows.append({
            "Sector": sector,
            "Base_Best_Model": best_base["Model"],
            "Base_RMSE": best_base["RMSE"],
            "Base_R2": best_base["R2"],
            "Base_MAPE": best_base["MAPE"],
            "Exog_Best_Model": best_exog["Model"],
            "Exog_RMSE": best_exog["RMSE"],
            "Exog_R2": best_exog["R2"],
            "Exog_MAPE": best_exog["MAPE"],
            "RMSE_Delta": round(best_base["RMSE"] - best_exog["RMSE"], 2),
            "R2_Delta": round(best_exog["R2"] - best_base["R2"], 4),
            "N_Base_Features": data_base["X_train"].shape[1],
            "N_Exog_Features": data_exog["X_train"].shape[1],
        })

        log.info("  Best Baseline: %s (R2=%.4f, RMSE=%.2f)",
                 best_base["Model"], best_base["R2"], best_base["RMSE"])
        log.info("  Best Exogenous: %s (R2=%.4f, RMSE=%.2f)",
                 best_exog["Model"], best_exog["R2"], best_exog["RMSE"])
        log.info("  Delta: RMSE=%.2f, R2=%.4f",
                 best_base["RMSE"] - best_exog["RMSE"],
                 best_exog["R2"] - best_base["R2"])

    # Save consolidated results
    comparison_df = pd.DataFrame(comparison_rows)
    save_dataframe(comparison_df, TABLE_DIR / "phase3_comparison.csv")

    all_results_base = pd.concat(all_base, ignore_index=True)
    all_results_exog = pd.concat(all_exog, ignore_index=True)
    full_results = pd.concat([all_results_base, all_results_exog], ignore_index=True)
    save_dataframe(full_results, TABLE_DIR / "phase3_full_results.csv")

    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ALL_SECTORS))
    width = 0.35
    base_r2 = comparison_df["Base_R2"].values
    exog_r2 = comparison_df["Exog_R2"].values
    ax.bar(x - width / 2, base_r2, width, label="Baseline", color="steelblue", edgecolor="k")
    ax.bar(x + width / 2, exog_r2, width, label="+ Exogenous", color="coral", edgecolor="k")
    ax.set_ylabel("R2")
    ax.set_title("Phase 3: Impact of Exogenous Variables on R2")
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_SECTORS)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (b, e) in enumerate(zip(base_r2, exog_r2)):
        delta = e - b
        color = "green" if delta > 0 else "red"
        ax.annotate(f"{delta:+.3f}", (i + width / 2, e + 0.01),
                    ha="center", fontsize=10, color=color, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase3_r2_improvement.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Print summary
    log.info("=" * 70)
    log.info("PHASE 3 COMPLETE")
    log.info("=" * 70)
    for _, row in comparison_df.iterrows():
        log.info("  %s: R2 %.4f -> %.4f (delta=%+.4f) | RMSE %.2f -> %.2f (delta=%+.2f)",
                 row["Sector"],
                 row["Base_R2"], row["Exog_R2"], row["R2_Delta"],
                 row["Base_RMSE"], row["Exog_RMSE"], -row["RMSE_Delta"])

    elapsed = time.time() - t0
    log.info("Total time: %.1f seconds", elapsed)

    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "exogenous_sources": [
            "INDPRO (Industrial Production Index) - FRED",
            "MCOILWTICO (WTI Crude Oil Price) - FRED",
            "CPIAUCSL (Consumer Price Index) - FRED",
            "MHHNGSP (Henry Hub Natural Gas Price) - FRED",
            "B230RC0Q173SBEA (US Population) - FRED",
            "HDD/CDD (Computed from sinusoidal temperature model)",
        ],
        "results": comparison_rows,
        "elapsed_seconds": round(elapsed, 1),
    }
    save_json(summary, TABLE_DIR / "phase3_summary.json")


if __name__ == "__main__":
    main()
