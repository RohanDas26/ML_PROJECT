"""
run_optuna_evaluation.py — Full Optuna Evaluation Pipeline
============================================================
Runs Bayesian hyperparameter optimization across ALL 4 sectors,
generates comprehensive test reports, and saves all artifacts.

Usage:
    python run_optuna_evaluation.py
    python run_optuna_evaluation.py --sector Commercial --n_trials 100
"""

import argparse
import sys
import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from src.data.loader import load_raw_data, validate_data, ALL_SECTORS
from src.data.feature_engineering import create_features, get_feature_columns
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.optuna_trainer import train_all_models_optuna
from src.evaluation.statistical_tests import run_dm_tests
from src.evaluation.diagnostics import residual_diagnostics, stationarity_tests
from src.evaluation.metrics import compute_all_metrics
from src.visualization.plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison,
    plot_feature_importance,
    plot_time_series_forecast,
)
from src.utils.io import save_dataframe, save_json, save_model
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")


def run_full_evaluation(
    config_path: str = "config/config.yaml",
    sectors: list[str] | None = None,
    n_trials: int = 80,
    timeout_per_model: int = 120,
):
    """Run Optuna-based full evaluation across all sectors."""

    # Load config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = get_logger(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_file=cfg.get("logging", {}).get("log_file"),
    )

    log.info("=" * 70)
    log.info("ENERGY FORECASTING -- OPTUNA FULL EVALUATION")
    log.info("Date: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 70)

    # Seed
    seed = cfg["project"].get("random_seed", 42)
    np.random.seed(seed)

    # Load data
    data_cfg = cfg["data"]
    df = load_raw_data(
        data_cfg["raw_path"],
        sheet_name=data_cfg.get("sheet_name", "Sheet1"),
        skiprows=data_cfg.get("skiprows", 1),
    )
    validation = validate_data(df)
    log.info("Data validation: %d rows, %d outliers", validation["n_rows"],
             sum(validation["outlier_counts"].values()))

    if sectors is None:
        sectors = data_cfg.get("sectors", ALL_SECTORS)

    # Output dirs
    out_cfg = cfg.get("output", {})
    tables_dir = Path(out_cfg.get("tables_dir", "Results/tables"))
    fig_dir = Path(out_cfg.get("figures_dir", "Results/figures"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for sector in sectors:
        log.info("\n" + "#" * 70)
        log.info("# SECTOR: %s", sector.upper())
        log.info("#" * 70)

        # Feature engineering
        feat_cfg = cfg.get("features", {})
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

        # Stationarity
        stationarity = stationarity_tests(features_df["target"])

        # Split & scale
        preproc = TimeSeriesPreprocessor()
        split_cfg = cfg.get("split", {})
        data = preproc.split(features_df, test_fraction=split_cfg.get("test_fraction", 0.20))
        data = preproc.fit_transform(data)

        # Optuna train
        results_df, best_models = train_all_models_optuna(
            data["X_train"], data["y_train"],
            data["X_test"], data["y_test"],
            data["y_train_orig"], data["y_test_orig"],
            preproc.scaler_y,
            n_splits=split_cfg.get("n_splits", 5),
            n_trials=n_trials,
            timeout_per_model=timeout_per_model,
        )

        # DM tests
        dm_results = run_dm_tests(results_df, baseline_name="Ridge")

        # Best model diagnostics
        best_name = results_df.iloc[0]["Model"]
        best_model = best_models[best_name]
        y_pred_scaled = best_model.predict(data["X_test"])
        y_pred = preproc.inverse_transform_y(y_pred_scaled)
        y_train_pred_scaled = best_model.predict(data["X_train"])
        y_train_pred = preproc.inverse_transform_y(y_train_pred_scaled)

        diag = residual_diagnostics(data["y_test_orig"], y_pred)
        full_metrics = compute_all_metrics(
            data["y_test_orig"], y_pred, n_features=len(data["feature_cols"])
        )

        # Plots
        plot_actual_vs_predicted(
            data["y_test_orig"], y_pred,
            title=f"{sector} -- Actual vs Predicted ({best_name}, Optuna)",
            save_path=fig_dir / f"{sector}_optuna_actual_vs_predicted.png",
        )
        plot_residuals(
            data["y_test_orig"], y_pred,
            title=f"{sector} -- Residuals ({best_name})",
            save_path=fig_dir / f"{sector}_optuna_residuals.png",
        )
        plot_model_comparison(
            results_df, metric="RMSE",
            title=f"{sector} -- Model Comparison (Optuna RMSE)",
            save_path=fig_dir / f"{sector}_optuna_model_comparison.png",
        )

        if hasattr(best_model, "coef_"):
            plot_feature_importance(
                np.abs(best_model.coef_), data["feature_cols"],
                title=f"{sector} -- Feature Importance ({best_name})",
                save_path=fig_dir / f"{sector}_optuna_feature_importance.png",
            )
        elif hasattr(best_model, "feature_importances_"):
            plot_feature_importance(
                best_model.feature_importances_, data["feature_cols"],
                title=f"{sector} -- Feature Importance ({best_name})",
                save_path=fig_dir / f"{sector}_optuna_feature_importance.png",
            )

        all_results[sector] = {
            "results": results_df,
            "dm_tests": dm_results,
            "best_models": best_models,
            "feature_cols": data["feature_cols"],
            "diagnostics": diag,
            "stationarity": stationarity,
            "full_metrics": full_metrics,
        }

    # ------------------------------------------------------------------
    # Save consolidated results
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("SAVING CONSOLIDATED RESULTS")
    log.info("=" * 70)

    # Full model comparison
    rows = []
    for sector, sdata in all_results.items():
        for _, row in sdata["results"].iterrows():
            rows.append({
                "Sector": sector,
                "Model": row["Model"],
                "RMSE": round(row["RMSE"], 2),
                "MAE": round(row["MAE"], 2),
                "R2": round(row["R2"], 4),
                "MAPE": round(row["MAPE"], 2),
                "MSE": round(row["MSE"], 2),
                "Overfit_Ratio": round(row["Overfit_Ratio"], 3),
                "CV_MSE": round(row["CV_MSE"], 6),
                "N_Trials": row["N_Trials"],
                "Time_s": row["Time_Seconds"],
                "Best_Params": row["Best_Params"],
            })
    comparison_df = pd.DataFrame(rows)
    save_dataframe(comparison_df, tables_dir / "optuna_model_comparison.csv")

    # DM tests
    dm_rows = []
    for sector, sdata in all_results.items():
        for _, row in sdata["dm_tests"].iterrows():
            dm_rows.append({"Sector": sector, **row.to_dict()})
    save_dataframe(pd.DataFrame(dm_rows), tables_dir / "optuna_dm_tests.csv")

    # Best-per-sector summary
    summary = {}
    for sector, sdata in all_results.items():
        best = sdata["results"].iloc[0]
        summary[sector] = {
            "best_model": best["Model"],
            "RMSE": round(float(best["RMSE"]), 2),
            "MAE": round(float(best["MAE"]), 2),
            "R2": round(float(best["R2"]), 4),
            "MAPE": round(float(best["MAPE"]), 2),
            "best_params": best["Best_Params"],
            "diagnostics": sdata["diagnostics"],
            "stationarity": sdata["stationarity"],
            "full_metrics": sdata["full_metrics"],
        }
    save_json(summary, tables_dir / "optuna_pipeline_summary.json")

    # Save best overall model
    best_sector = min(summary, key=lambda s: summary[s]["RMSE"])
    best_model_name = summary[best_sector]["best_model"]
    best_model_obj = all_results[best_sector]["best_models"][best_model_name]
    save_model(best_model_obj, out_cfg.get("model_path", "Data/Artifacts/final_model.pkl"))

    # ------------------------------------------------------------------
    # Print final report
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("FINAL REPORT -- OPTUNA HYPERPARAMETER OPTIMIZATION")
    log.info("=" * 70)

    for sector, s in summary.items():
        log.info("\n  [%s]", sector.upper())
        log.info("    Best Model:    %s", s["best_model"])
        log.info("    RMSE:          %.2f Trillion BTU", s["RMSE"])
        log.info("    MAE:           %.2f Trillion BTU", s["MAE"])
        log.info("    R2:            %.4f (%.1f%% variance explained)", s["R2"], s["R2"] * 100)
        log.info("    MAPE:          %.2f%%", s["MAPE"])
        log.info("    Stationary:    %s", s["stationarity"].get("Stationary", "N/A"))
        log.info("    Residual DW:   %.3f", s["diagnostics"]["durbin_watson"])

    log.info("\n" + "=" * 70)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 70)

    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Full Evaluation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--sector", default=None, help="Single sector to evaluate")
    parser.add_argument("--n_trials", type=int, default=80, help="Optuna trials per model")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout (s) per model")
    args = parser.parse_args()

    sectors = [args.sector] if args.sector else None
    run_full_evaluation(
        config_path=args.config,
        sectors=sectors,
        n_trials=args.n_trials,
        timeout_per_model=args.timeout,
    )
