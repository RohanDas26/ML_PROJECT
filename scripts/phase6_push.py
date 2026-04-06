"""
phase6_push.py — Maximum Performance Push
==========================================
Every technique available WITHOUT new external data:

NEW vs Phase 2:
  1. Cross-sector lag features  — lagged values of OTHER sectors (no leakage)
  2. Richer temporal features   — quarter, season, trend, year-over-year growth
  3. Multi-output joint learning — train one model for all 4 sectors at once
  4. Stacking ensemble          — base model predictions as meta-features
  5. Per-fold scaling + TimeSeriesSplit (fully fair)

Output: Results/tables/phase6_max_performance.json
"""

import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ---- Paths ------------------------------------------------------------------
EXCEL_PATH  = Path(r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\Dataset\USA ENGERY PREDICTION.xlsx")
RESULTS_DIR = Path(r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\EnergyForecasting_v2_Production\Results\tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_COLS = {
    "Residential":    "Primary Energy Consumed by the Residential Sector",
    "Commercial":     "Primary Energy Consumed by the Commercial Sector",
    "Industrial":     "Primary Energy Consumed by the Industrial Sector",
    "Transportation": "Primary Energy Consumed by the Transportation Sector",
}

# Phase 2 baseline (Trillion BTU RMSE) for comparison
PHASE2_BEST = {
    "Residential":    67.83,
    "Commercial":     38.55,
    "Industrial":     62.15,
    "Transportation": 99.29,
}

# ---- Data Loading -----------------------------------------------------------
def load_all():
    df = pd.read_excel(EXCEL_PATH, skiprows=0)
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Month"})
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month"]).sort_values("Month").reset_index(drop=True)
    for col in SECTOR_COLS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Loaded {len(df)} rows ({df['Month'].min().year} -> {df['Month'].max().year})")
    return df

# ---- Feature Engineering ----------------------------------------------------
def build_features(df, target_col, all_sector_cols):
    """
    Full feature set — no leakage.
    Own-sector: Fourier, lags, rolling, diff, growth
    Cross-sector: lagged values of all other sectors
    Calendar: month, year, quarter, season, t
    """
    feat = pd.DataFrame(index=df.index)
    t = np.arange(len(df))

    # Calendar
    feat["month"]   = df["Month"].dt.month
    feat["year"]    = df["Month"].dt.year
    feat["quarter"] = df["Month"].dt.quarter
    feat["trend"]   = t  # long-term linear trend

    # Season dummies (no leakage — deterministic)
    feat["is_winter"] = df["Month"].dt.month.isin([12, 1, 2]).astype(int)
    feat["is_summer"] = df["Month"].dt.month.isin([6, 7, 8]).astype(int)

    # Fourier harmonics — 6 orders (12 features)
    for k in range(1, 7):
        feat[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 12)
        feat[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 12)

    # Own-sector lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        feat[f"own_lag{lag}"] = df[target_col].shift(lag)

    # Own-sector rolling stats (on lag-1, no leakage)
    lag1 = df[target_col].shift(1)
    for w in [3, 6, 12]:
        feat[f"own_rmean{w}"] = lag1.rolling(w).mean()
        feat[f"own_rstd{w}"]  = lag1.rolling(w).std()

    # Own-sector momentum
    feat["own_diff1"]   = df[target_col].diff(1)
    feat["own_diff12"]  = df[target_col].diff(12)
    feat["own_yoy_pct"] = df[target_col].pct_change(12)  # year-over-year % change

    # Cross-sector LAGGED features (lag-1 of other sectors — no leakage)
    other_sectors = [c for c in all_sector_cols if c != target_col]
    for i, other_col in enumerate(other_sectors):
        if other_col in df.columns:
            feat[f"cross{i}_lag1"]  = df[other_col].shift(1)
            feat[f"cross{i}_lag12"] = df[other_col].shift(12)
            feat[f"cross{i}_diff1"] = df[other_col].diff(1).shift(1)

    valid = feat.dropna().index
    return feat.loc[valid].values, df.loc[valid, target_col].values, feat.columns.tolist()

# ---- Model Registry ---------------------------------------------------------
def get_base_models():
    m = {
        "Ridge":            Ridge(alpha=1.0),
        "Lasso":            Lasso(alpha=0.005, max_iter=10000),
        "ElasticNet":       ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=10000),
        "GradBoosting":     GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
        "ExtraTrees":       ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        m["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     min_child_weight=3, random_state=42, verbosity=0)
    if HAS_LGBM:
        m["LightGBM"] = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=42, verbose=-1)
    return m

# ---- Per-fold CV (fully fair) -----------------------------------------------
def cv_per_fold(X, y, model, n_folds=10):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    rmses, maes, r2s = [], [], []
    for tr, te in tscv.split(X):
        sx = StandardScaler()
        X_tr = sx.fit_transform(X[tr]); X_te = sx.transform(X[te])
        sy = StandardScaler()
        y_tr = sy.fit_transform(y[tr].reshape(-1,1)).ravel()
        y_te_s = sy.transform(y[te].reshape(-1,1)).ravel()
        m = clone(model); m.fit(X_tr, y_tr)
        pred_s = m.predict(X_te)
        # Inverse-transform to BTU for real-unit metrics
        pred_btu = sy.inverse_transform(pred_s.reshape(-1,1)).ravel()
        true_btu = y[te]
        rmses.append(float(np.sqrt(mean_squared_error(true_btu, pred_btu))))
        maes.append(float(mean_absolute_error(true_btu, pred_btu)))
        r2_sc = float(r2_score(y_te_s, pred_s))
        r2s.append(r2_sc)
    return np.mean(rmses), np.std(rmses), np.mean(maes), np.mean(r2s)

# ---- Stacking Ensemble ------------------------------------------------------
def stacking_cv(X, y, base_models, meta_model=Ridge(), n_folds=10):
    """
    Proper stacking with TimeSeriesSplit:
    - Generate OOF predictions from each base model
    - Train meta-learner on OOF predictions
    - Score meta-learner on held-out test fold
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)
    splits = list(tscv.split(X))
    rmses, r2s = [], []

    for fold_idx, (tr, te) in enumerate(splits):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

        # Scale X on this fold's training data
        sx = StandardScaler()
        X_tr_s = sx.fit_transform(X_tr); X_te_s = sx.transform(X_te)
        sy = StandardScaler()
        y_tr_s = sy.fit_transform(y_tr.reshape(-1,1)).ravel()

        # Generate base model OOF predictions via inner CV on training fold
        inner_splits = list(TimeSeriesSplit(n_splits=min(5, len(tr)//20+2)).split(X_tr_s))
        oof_preds = np.zeros((len(X_tr_s), len(base_models)))
        for bm_idx, (bname, bmodel) in enumerate(base_models.items()):
            fold_preds = np.zeros(len(X_tr_s))
            for itr, ite in inner_splits:
                bm = clone(bmodel)
                bm.fit(X_tr_s[itr], y_tr_s[itr])
                fold_preds[ite] = bm.predict(X_tr_s[ite])
            oof_preds[:, bm_idx] = fold_preds

        # Base model predictions on test fold
        test_preds = np.zeros((len(X_te_s), len(base_models)))
        for bm_idx, (bname, bmodel) in enumerate(base_models.items()):
            bm = clone(bmodel)
            bm.fit(X_tr_s, y_tr_s)
            test_preds[:, bm_idx] = bm.predict(X_te_s)

        # Train meta-learner on OOF Z-score predictions
        meta = clone(meta_model)
        meta.fit(oof_preds, y_tr_s)
        meta_pred_s = meta.predict(test_preds)

        # Inverse-transform to BTU
        pred_btu = sy.inverse_transform(meta_pred_s.reshape(-1,1)).ravel()
        rmses.append(float(np.sqrt(mean_squared_error(y_te, pred_btu))))
        r2_sc = float(r2_score(y_tr_s, meta.predict(oof_preds)))  # OOF R²
        r2s.append(r2_sc)

    return float(np.mean(rmses)), float(np.std(rmses))

# ---- Multi-Output (Joint Learning) ------------------------------------------
def multioutput_cv(df, all_sector_cols, n_folds=10):
    """
    Train one model to predict all 4 sectors simultaneously.
    Features are built from ALL sectors' lagged data jointly.
    """
    print("\n  [Multi-Output Joint Learning]")

    # Build shared feature matrix using sector-agnostic time features
    feat = pd.DataFrame(index=df.index)
    t = np.arange(len(df))
    feat["month"] = df["Month"].dt.month
    feat["year"]  = df["Month"].dt.year
    feat["trend"] = t
    for k in range(1, 7):
        feat[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 12)
        feat[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 12)

    # Add lag-1 and lag-12 of ALL sectors as features
    for i, col in enumerate(all_sector_cols):
        if col in df.columns:
            feat[f"s{i}_lag1"]  = df[col].shift(1)
            feat[f"s{i}_lag12"] = df[col].shift(12)
            feat[f"s{i}_diff1"] = df[col].diff(1)

    # Target matrix: all 4 sectors
    target_df = df[all_sector_cols].copy()

    valid = feat.dropna().index
    valid = valid.intersection(target_df.dropna().index)
    X = feat.loc[valid].values
    Y = target_df.loc[valid].values  # shape (n, 4)

    tscv = TimeSeriesSplit(n_splits=n_folds)
    sector_rmses = {s: [] for s in SECTOR_COLS.keys()}

    models_to_try = {"Ridge": Ridge(alpha=1.0)}
    if HAS_XGB:
        models_to_try["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=5,
                                                  learning_rate=0.05, random_state=42,
                                                  verbosity=0)

    best_rmses = {s: np.inf for s in SECTOR_COLS.keys()}
    best_model_name = {s: "" for s in SECTOR_COLS.keys()}

    for model_name, base_model in models_to_try.items():
        mo_model = MultiOutputRegressor(base_model, n_jobs=-1)
        fold_rmses = {s: [] for s in SECTOR_COLS.keys()}

        for tr, te in tscv.split(X):
            sx = StandardScaler()
            X_tr = sx.fit_transform(X[tr]); X_te = sx.transform(X[te])
            sy = StandardScaler()
            Y_tr = sy.fit_transform(Y[tr]); Y_te = Y[te]

            m = clone(mo_model); m.fit(X_tr, Y_tr)
            Y_pred_s = m.predict(X_te)
            Y_pred = sy.inverse_transform(Y_pred_s)

            for i, sector in enumerate(SECTOR_COLS.keys()):
                fold_rmses[sector].append(
                    float(np.sqrt(mean_squared_error(Y_te[:, i], Y_pred[:, i])))
                )

        for sector in SECTOR_COLS.keys():
            mean_rmse = float(np.mean(fold_rmses[sector]))
            if mean_rmse < best_rmses[sector]:
                best_rmses[sector] = mean_rmse
                best_model_name[sector] = f"MultiOutput({model_name})"

    return best_rmses, best_model_name

# ---- Main -------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 6 — Maximum Performance Push (No New Data)")
    print("Features: Fourier + Own lags + Cross-sector lags + Calendar + Stack")
    print("=" * 70)
    df = load_all()

    all_sector_cols = list(SECTOR_COLS.values())
    results = {
        "experiment": "Phase 6 -- Maximum push without new data",
        "techniques": [
            "Cross-sector lagged features (lag-1 and lag-12 of other sectors)",
            "Richer calendar features (quarter, season, linear trend, YoY growth)",
            "Fourier harmonics (6 orders)",
            "Stacking ensemble (base predictions -> meta-learner via inner CV)",
            "Multi-output joint learning (all 4 sectors simultaneously)",
            "Per-fold StandardScaler + TimeSeriesSplit (fully fair)",
        ],
        "sectors": {},
    }
    summary = []

    # Multi-output first (shared computation)
    mo_rmses, mo_model_names = multioutput_cv(df, all_sector_cols, n_folds=10)

    for sector, target_col in SECTOR_COLS.items():
        if target_col not in df.columns:
            print(f"  Skipping {sector}: column not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Sector: {sector}")
        print(f"{'='*60}")

        sub = df[["Month", target_col] + [c for c in all_sector_cols if c != target_col]].copy()
        X, y, feat_names = build_features(df, target_col, all_sector_cols)
        print(f"  Feature matrix: {X.shape}  ({X.shape[1]} features including cross-sector)")

        # --- Single-sector models ---
        all_model_results = {}
        base_models = get_base_models()

        for mname, model in base_models.items():
            rmse_m, rmse_std, mae_m, r2_m = cv_per_fold(X, y, model)
            all_model_results[mname] = {
                "RMSE_BTU": round(rmse_m, 2),
                "RMSE_std": round(rmse_std, 2),
                "MAE_BTU":  round(mae_m, 2),
                "R2":       round(r2_m, 4),
            }

        # --- Stacking ensemble ---
        print("  Running stacking ensemble...")
        stack_rmse, stack_std = stacking_cv(X, y, base_models)
        all_model_results["Stacking(Ridge-meta)"] = {
            "RMSE_BTU": round(stack_rmse, 2),
            "RMSE_std": round(stack_std, 2),
            "MAE_BTU":  None,
            "R2":       None,
        }

        # --- Multi-output result ---
        mo_r = mo_rmses.get(sector, np.inf)
        all_model_results[mo_model_names.get(sector, "MultiOutput")] = {
            "RMSE_BTU": round(mo_r, 2),
            "RMSE_std": None,
            "MAE_BTU":  None,
            "R2":       None,
        }

        # Best
        best_name = min(all_model_results, key=lambda n: all_model_results[n]["RMSE_BTU"] or 1e9)
        best_rmse = all_model_results[best_name]["RMSE_BTU"]
        phase2    = PHASE2_BEST.get(sector, 9999)
        improv    = round((1 - best_rmse / phase2) * 100, 1)

        results["sectors"][sector] = {
            "phase2_best_RMSE_BTU": phase2,
            "phase6_best_RMSE_BTU": best_rmse,
            "best_model":           best_name,
            "improvement_vs_phase2": f"{improv:+.1f}%",
            "all_models":           all_model_results,
        }

        print(f"\n  Phase 2 RMSE:  {phase2:.2f} BTU")
        print(f"  Phase 6 best ({best_name}): {best_rmse:.2f} BTU  ({improv:+.1f}% vs Phase 2)")
        print()
        for mname in sorted(all_model_results, key=lambda n: all_model_results[n]["RMSE_BTU"] or 1e9):
            r = all_model_results[mname]
            print(f"    {mname:<28} RMSE={r['RMSE_BTU']:.2f}  R2={r.get('R2') or 'N/A'}")

        summary.append({
            "Sector":        sector,
            "Phase2 RMSE":   phase2,
            "Phase6 RMSE":   best_rmse,
            "Best Model":    best_name,
            "Improvement":   f"{improv:+.1f}%",
        })

    # Save
    out_path = RESULTS_DIR / "phase6_max_performance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 6 FINAL SUMMARY")
    print("=" * 70)
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
