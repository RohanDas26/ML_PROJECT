"""
main.py — Energy Forecasting Pipeline Entry Point
====================================================
Orchestrates the full pipeline: load → features → split → scale →
train → evaluate → save results.

Usage:
    python main.py                         # default: all sectors
    python main.py --sector Commercial     # single sector
    python main.py --config config/config.yaml
"""

import argparse
import sys
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data, validate_data, ALL_SECTORS
from src.data.feature_engineering import (
    create_features, 
    get_feature_columns, 
    select_features_lasso
)
from src.data.preprocessor import TimeSeriesPreprocessor
from src.data.exogenous import get_all_exogenous
from src.models.trainer import train_all_models
from src.models.forecaster import RecursiveForecaster
from src.evaluation.statistical_tests import run_dm_tests
from src.evaluation.diagnostics import residual_diagnostics, stationarity_tests
from src.evaluation.metrics import compute_all_metrics
from src.visualization.plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison,
    plot_model_forecast,  # Need to implement this or use generic plot
    plot_feature_importance,
)
from src.utils.io import save_dataframe, save_json, save_model
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------
# Single-sector pipeline
# -----------------------------------------------------------------------

def run_sector_pipeline(
    df: pd.DataFrame,
    target_sector: str,
    cfg: dict,
    log,
    exog_df: pd.DataFrame | None = None,
    select_features: bool = False,
    forecast_horizon: int = 0,
) -> dict:
    """
    Run the full pipeline for one target sector.

    Returns
    -------
    dict with keys: results, dm_tests, best_models, feature_cols, diagnostics
    """
    log.info("=" * 70)
    log.info("SECTOR: %s", target_sector.upper())
    log.info("=" * 70)

    # 1. Feature engineering
    feat_cfg = cfg.get("features", {})
    features_df = create_features(
        df,
        target_sector,
        target_lags=feat_cfg.get("target_lags"),
        rolling_windows=feat_cfg.get("rolling_windows"),
        ema_spans=feat_cfg.get("ema_spans"),
        harmonic_periods=feat_cfg.get("harmonic_periods"),
        cross_sector_lags=feat_cfg.get("cross_sector_lags"),
        use_differencing=feat_cfg.get("use_differencing", False),
        seasonal_flags=feat_cfg.get("seasonal_flags", True),
        momentum=feat_cfg.get("momentum", True),
        exog_df=exog_df if target_sector in ALL_SECTORS else None, # pass exog if valid
    )

    # 1b. Feature Selection (Lasso)
    if select_features:
        log.info("Running Lasso feature selection...")
        selected_cols = select_features_lasso(features_df, min_features=15)
        # Keep Month and target, drop others not in selected
        keep_cols = ["Month", "target"] + selected_cols
        features_df = features_df[keep_cols]
        log.info("Feature selection complete. Retained %d covariance features.", len(selected_cols))

    # 2. Stationarity check on target
    stationarity = stationarity_tests(features_df["target"])
    if cfg.get("features", {}).get("stationarity") == "auto" and not stationarity.get("Stationary", False):
        log.warning("Target is NON-STATIONARY but auto-differencing is not fully enabled in this MVP.")
        log.warning("Recommendation: Enable 'use_differencing: true' in config or use robust models.")

    # 3. Split & scale
    preproc = TimeSeriesPreprocessor()
    split_cfg = cfg.get("split", {})
    data = preproc.split(features_df, test_fraction=split_cfg.get("test_fraction", 0.20))
    data = preproc.fit_transform(data)

    # 4. Train all models
    results_df, best_models = train_all_models(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        data["y_train_orig"], data["y_test_orig"],
        preproc.scaler_y,
        n_splits=split_cfg.get("n_splits", 5),
        random_state=cfg.get("project", {}).get("random_seed", 42),
    )

    # 5. DM tests
    eval_cfg = cfg.get("evaluation", {})
    dm_baseline = eval_cfg.get("dm_baseline", "Ridge")
    dm_results = run_dm_tests(results_df, baseline_name=dm_baseline)

    # 6. Residual diagnostics for best model
    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = best_models[best_name]

    y_pred_scaled = best_model.predict(data["X_test"])
    y_pred = preproc.inverse_transform_y(y_pred_scaled)
    diag = residual_diagnostics(data["y_test_orig"], y_pred)

    # 7. Plots
    out_cfg = cfg.get("output", {})
    fig_dir = Path(out_cfg.get("figures_dir", "Results/figures"))

    plot_actual_vs_predicted(
        data["y_test_orig"], y_pred,
        title=f"{target_sector} — Actual vs Predicted ({best_name})",
        save_path=fig_dir / f"{target_sector}_actual_vs_predicted.png",
    )
    plot_residuals(
        data["y_test_orig"], y_pred,
        title=f"{target_sector} — Residual Analysis ({best_name})",
        save_path=fig_dir / f"{target_sector}_residuals.png",
    )
    plot_model_comparison(
        results_df, metric="RMSE",
        title=f"{target_sector} — Model Comparison (RMSE)",
        save_path=fig_dir / f"{target_sector}_model_comparison.png",
    )

    # Feature importance
    if hasattr(best_model, "coef_"):
        plot_feature_importance(
            np.abs(best_model.coef_),
            data["feature_cols"],
            title=f"{target_sector} — Feature Importance ({best_name})",
            save_path=fig_dir / f"{target_sector}_feature_importance.png",
        )
    elif hasattr(best_model, "feature_importances_"):
        plot_feature_importance(
            best_model.feature_importances_,
            data["feature_cols"],
            title=f"{target_sector} — Feature Importance ({best_name})",
            save_path=fig_dir / f"{target_sector}_feature_importance.png",
        )

    # 8. Forecasting (Recursion)
    forecast_results = None
    if forecast_horizon > 0:
        log.info("Running recursive forecast for %d months...", forecast_horizon)
        forecaster = RecursiveForecaster(
            model=best_model,
            preprocessor=preproc,
            feature_fn=create_features,
            sector=target_sector,
            feat_cfg=feat_cfg,
            history_df=df,
            feature_cols=data["feature_cols"],
            exog_df=exog_df
        )
        forecast_df = forecaster.predict_horizon(forecast_horizon)
        
        # Save forecast
        save_dataframe(forecast_df, fig_dir.parent / "tables" / f"{target_sector}_forecast.csv")
        
        # Plot forecast with history
        from src.visualization.plots import plot_recursive_forecast
        plot_recursive_forecast(
            history_dates=df["Month"],
            history_vals=df[target_sector],
            forecast_dates=forecast_df["Month"],
            forecast_vals=forecast_df["Forecast"],
            title=f"{target_sector} — {forecast_horizon}-Month Forecast ({best_name})",
            save_path=fig_dir / f"{target_sector}_forecast.png"
        )
        forecast_results = forecast_df.to_dict(orient="records")

    return {
        "results": results_df,
        "dm_tests": dm_results,
        "best_models": best_models,
        "feature_cols": data["feature_cols"],
        "diagnostics": diag,
        "stationarity": stationarity,
        "forecast": forecast_results,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(config_path: str = "config/config.yaml",
         sectors: list[str] | None = None,
         use_exogenous: bool = False,
         select_features: bool = False,
         forecast_horizon: int = 0) -> dict:
    """
    Run the complete pipeline for one or more sectors.

    Parameters
    ----------
    config_path : path to YAML config
    sectors : list of sector names, or None for all
    use_exogenous : if True, fetch and use external data (FRED, HDD/CDD)
    select_features : if True, use Lasso to prune features
    forecast_horizon : months to forecast into future (0 = no forecast)

    Returns
    -------
    dict mapping sector → results dict
    """
    cfg = load_config(config_path)
    log = get_logger(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_file=cfg.get("logging", {}).get("log_file"),
    )

    log.info("=" * 70)
    log.info("ENERGY FORECASTING FRAMEWORK v%s", cfg["project"]["version"])
    log.info("=" * 70)

    # Set random seed globally
    seed = cfg["project"].get("random_seed", 42)
    np.random.seed(seed)
    log.info("Random seed: %d", seed)

    # Load data
    data_cfg = cfg["data"]
    df = load_raw_data(
        data_cfg["raw_path"],
        sheet_name=data_cfg.get("sheet_name", "Sheet1"),
        skiprows=data_cfg.get("skiprows", 1),
    )
    validation = validate_data(df)
    log.info("Validation report: %s", validation)

    # Fetch Exogenous Data
    exog_df = None
    if use_exogenous:
        log.info("Fetching exogenous data (FRED + Weather)...")
        exog_df = get_all_exogenous(df["Month"])
        log.info("Exogenous data ready: %d columns", exog_df.shape[1] - 1)

    # Determine sectors to run
    if sectors is None:
        sectors = data_cfg.get("sectors", ALL_SECTORS)

    # Run pipeline per sector
    all_results = {}
    for sector in sectors:
        all_results[sector] = run_sector_pipeline(
            df, sector, cfg, log,
            exog_df=exog_df,
            select_features=select_features,
            forecast_horizon=forecast_horizon
        )

    # ------------------------------------------------------------------
    # Save consolidated results
    # ------------------------------------------------------------------
    out_cfg = cfg.get("output", {})
    tables_dir = Path(out_cfg.get("tables_dir", "Results/tables"))

    # Model comparison
    rows = []
    for sector, sdata in all_results.items():
        for _, row in sdata["results"].iterrows():
            rows.append({
                "Sector": sector,
                "Model": row["Model"],
                "MSE": row["MSE"],
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "R2": row["R2"],
                "Best_Params": row["Best_Params"],
                "Overfit_Ratio": row["Overfit_Ratio"],
            })
    save_dataframe(pd.DataFrame(rows), tables_dir / "final_model_comparison.csv")

    # DM tests
    dm_rows = []
    for sector, sdata in all_results.items():
        for _, row in sdata["dm_tests"].iterrows():
            dm_rows.append({"Sector": sector, **row.to_dict()})
    save_dataframe(pd.DataFrame(dm_rows), tables_dir / "diebold_mariano_tests.csv")

    # Best models summary
    summary = {}
    for sector, sdata in all_results.items():
        best = sdata["results"].iloc[0]
        summ_entry = {
            "best_model": best["Model"],
            "RMSE": round(float(best["RMSE"]), 2),
            "R2": round(float(best["R2"]), 4),
            "diagnostics": sdata["diagnostics"],
            "stationarity": sdata["stationarity"],
        }
        if sdata.get("forecast"):
             summ_entry["forecast_next_12m"] = "Generated"
             
        summary[sector] = summ_entry
        
    save_json(summary, tables_dir / "pipeline_summary.json")

    # Save best model artifact
    best_overall_sector = min(summary, key=lambda s: summary[s]["RMSE"])
    best_overall_name = summary[best_overall_sector]["best_model"]
    best_overall_model = all_results[best_overall_sector]["best_models"][best_overall_name]
    save_model(best_overall_model, out_cfg.get("model_path", "Data/Artifacts/final_model.pkl"))

    # Final log
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 70)
    for sector, s in summary.items():
        log.info("  %s → %s  (RMSE=%.2f, R²=%.4f)",
                 sector, s["best_model"], s["RMSE"], s["R2"])
        if "forecast_next_12m" in s:
            log.info("    -> Forecast generated for next %d months", forecast_horizon)

    return all_results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Forecasting Pipeline")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--sector", default=None,
                        help="Run for a single sector (e.g. Commercial)")
    parser.add_argument("--exogenous", action="store_true",
                        help="Enable exogenous variables (FRED data)")
    parser.add_argument("--select-features", action="store_true",
                        help="Use Lasso to select best features (prevents overfitting)")
    parser.add_argument("--forecast", type=int, default=0,
                        help="Number of months to forecast into the future (e.g. 12)")
                        
    args = parser.parse_args()

    sectors = [args.sector] if args.sector else None
    main(
        config_path=args.config, 
        sectors=sectors,
        use_exogenous=args.exogenous,
        select_features=args.select_features,
        forecast_horizon=args.forecast
    )
