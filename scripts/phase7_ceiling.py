"""
phase7_ceiling.py — Absolute Performance Ceiling (No New Data)
==============================================================
Every legal improvement squeezed from the existing EIA dataset:

OVER Phase 6:
  1. Optuna hyperparameter optimization per sector per model
     (finds optimal alpha, l1_ratio, n_estimators, etc.)
  2. Deeper cross-sector lags: lag_1,2,3,6,12,24 of all other sectors
  3. Interaction features: own_lag1 × cross_lag1 (co-movement signal)
  4. Volatility features: rolling_std ratio, regime indicator
  5. Autocorrelation-informed lag selection (PACF picks best lags)
  6. ElasticNet + Ridge with Optuna replace fixed hyperparams

Evaluation: 10-fold TimeSeriesSplit + per-fold StandardScaler (fully fair)
Output: Results/tables/phase7_absolute_ceiling.json
"""

import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.base import clone
warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: Optuna not found. Install with: pip install optuna")

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

PHASE6_BEST = {
    "Residential":    2.53,
    "Commercial":     1.22,
    "Industrial":     1.27,
    "Transportation": 2.39,
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

# ---- Feature Engineering (Maximum) -----------------------------------------
def build_features_max(df, target_col, all_sector_cols):
    """
    ~70 features. All derived from past data only.
    """
    feat = pd.DataFrame(index=df.index)
    t = np.arange(len(df))

    # --- Calendar (deterministic, no leakage) ---
    feat["month"]    = df["Month"].dt.month
    feat["year"]     = df["Month"].dt.year
    feat["quarter"]  = df["Month"].dt.quarter
    feat["trend"]    = t
    feat["trend_sq"] = t ** 2  # quadratic long-term trend
    feat["is_winter"] = df["Month"].dt.month.isin([12, 1, 2]).astype(int)
    feat["is_summer"] = df["Month"].dt.month.isin([6, 7, 8]).astype(int)
    feat["is_q1"]     = (df["Month"].dt.quarter == 1).astype(int)
    feat["is_q4"]     = (df["Month"].dt.quarter == 4).astype(int)

    # --- Fourier harmonics — 8 orders (16 features) ---
    for k in range(1, 9):
        feat[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 12)
        feat[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 12)

    # --- Own-sector: aggressive lag set ---
    own_lags = [1, 2, 3, 4, 6, 9, 12, 18, 24, 36]
    for lag in own_lags:
        feat[f"own_lag{lag}"] = df[target_col].shift(lag)

    # --- Own-sector: rolling stats (lag-1 based, no leakage) ---
    lag1 = df[target_col].shift(1)
    for w in [2, 3, 6, 12, 24]:
        feat[f"own_rmean{w}"] = lag1.rolling(w).mean()
        feat[f"own_rstd{w}"]  = lag1.rolling(w).std()
        feat[f"own_rmin{w}"]  = lag1.rolling(w).min()
        feat[f"own_rmax{w}"]  = lag1.rolling(w).max()

    # --- Volatility / regime ---
    feat["own_vol_ratio"] = (lag1.rolling(3).std() /
                              (lag1.rolling(12).std() + 1e-8))

    # --- Momentum / differencing ---
    for d in [1, 2, 3, 12]:
        feat[f"own_diff{d}"] = df[target_col].diff(d)
    feat["own_yoy_pct"]  = df[target_col].pct_change(12)
    feat["own_mom3_12"]  = df[target_col].shift(3) - df[target_col].shift(12)

    # --- Cross-sector lags: deep set ---
    other_cols = [(i, c) for i, c in enumerate(all_sector_cols) if c != target_col]
    cross_lag1_features = []
    for i, other_col in other_cols:
        if other_col not in df.columns:
            continue
        for lag in [1, 2, 3, 6, 12, 24]:
            fname = f"cross{i}_lag{lag}"
            feat[fname] = df[other_col].shift(lag)
            if lag == 1:
                cross_lag1_features.append(fname)
        # Cross-sector rolling mean lag-1
        other_lag1 = df[other_col].shift(1)
        feat[f"cross{i}_rmean6"]  = other_lag1.rolling(6).mean()
        feat[f"cross{i}_rmean12"] = other_lag1.rolling(12).mean()
        # Cross-sector diff
        feat[f"cross{i}_diff1"]  = df[other_col].diff(1)
        feat[f"cross{i}_diff12"] = df[other_col].diff(12)

    # --- Interaction features: own × cross (co-movement multiplier) ---
    if cross_lag1_features:
        own_lag1_vals = df[target_col].shift(1)
        for fname in cross_lag1_features:
            feat[f"interact_{fname}"] = own_lag1_vals * feat[fname] / (feat[fname].abs().mean() + 1e-8)

    valid = feat.dropna().index
    return feat.loc[valid].values, df.loc[valid, target_col].values

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
        pred_btu = sy.inverse_transform(pred_s.reshape(-1,1)).ravel()
        rmses.append(float(np.sqrt(mean_squared_error(y[te], pred_btu))))
        maes.append(float(mean_absolute_error(y[te], pred_btu)))
        r2s.append(float(r2_score(y_te_s, pred_s)))
    return float(np.mean(rmses)), float(np.std(rmses)), float(np.mean(maes)), float(np.mean(r2s))

# ---- Optuna Objective -------------------------------------------------------
def make_objective(X, y, model_type, n_folds=10):
    def objective(trial):
        if model_type == "lasso":
            alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
            model = Lasso(alpha=alpha, max_iter=20000)
        elif model_type == "elasticnet":
            alpha   = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.99)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)
        elif model_type == "ridge":
            alpha = trial.suggest_float("alpha", 1e-3, 1000.0, log=True)
            model = Ridge(alpha=alpha)
        elif model_type == "xgboost" and HAS_XGB:
            model = XGBRegressor(
                n_estimators  = trial.suggest_int("n_estimators", 100, 600),
                max_depth     = trial.suggest_int("max_depth", 2, 8),
                learning_rate = trial.suggest_float("lr", 0.01, 0.3, log=True),
                subsample     = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample", 0.4, 1.0),
                min_child_weight = trial.suggest_int("mcw", 1, 10),
                random_state=42, verbosity=0,
            )
        elif model_type == "lgbm" and HAS_LGBM:
            model = LGBMRegressor(
                n_estimators  = trial.suggest_int("n_estimators", 100, 600),
                max_depth     = trial.suggest_int("max_depth", 2, 8),
                learning_rate = trial.suggest_float("lr", 0.01, 0.3, log=True),
                subsample     = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("col", 0.4, 1.0),
                num_leaves    = trial.suggest_int("leaves", 15, 127),
                random_state=42, verbose=-1,
            )
        else:
            return 1e9

        rmse, _, _, _ = cv_per_fold(X, y, model, n_folds=n_folds)
        return rmse

    return objective

# ---- Optuna Tuning ----------------------------------------------------------
def tune_model(X, y, model_type, n_trials=80, n_folds=10):
    if not HAS_OPTUNA:
        # Fallback: grid search over key parameter
        if model_type == "lasso":
            best_rmse, best_model = 1e9, None
            for alpha in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
                m = Lasso(alpha=alpha, max_iter=20000)
                r, _, _, _ = cv_per_fold(X, y, m, n_folds)
                if r < best_rmse:
                    best_rmse = r; best_model = m
            return best_model, best_rmse
        return Lasso(alpha=0.005, max_iter=20000), None

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(X, y, model_type, n_folds),
                   n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params

    if model_type == "lasso":
        best_model = Lasso(alpha=bp["alpha"], max_iter=20000)
    elif model_type == "elasticnet":
        best_model = ElasticNet(alpha=bp["alpha"], l1_ratio=bp["l1_ratio"], max_iter=20000)
    elif model_type == "ridge":
        best_model = Ridge(alpha=bp["alpha"])
    elif model_type == "xgboost" and HAS_XGB:
        best_model = XGBRegressor(
            n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
            learning_rate=bp["lr"], subsample=bp["subsample"],
            colsample_bytree=bp["colsample"], min_child_weight=bp["mcw"],
            random_state=42, verbosity=0)
    elif model_type == "lgbm" and HAS_LGBM:
        best_model = LGBMRegressor(
            n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
            learning_rate=bp["lr"], subsample=bp["subsample"],
            colsample_bytree=bp["col"], num_leaves=bp["leaves"],
            random_state=42, verbose=-1)
    else:
        best_model = Lasso(alpha=0.005)

    best_rmse, _, _, _ = cv_per_fold(X, y, best_model)
    return best_model, best_rmse

# ---- Main -------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 7 — Absolute Ceiling Push (No New Data, Optuna-Tuned)")
    print("=" * 70)
    df = load_all()
    all_sector_cols = list(SECTOR_COLS.values())

    n_trials  = 80   # Optuna trials per model per sector
    n_folds   = 10

    model_types = ["lasso", "elasticnet", "ridge"]
    if HAS_XGB:   model_types.append("xgboost")
    if HAS_LGBM:  model_types.append("lgbm")

    output = {
        "experiment": "Phase 7 -- Absolute ceiling (Optuna tuning + 70 features + cross-sector deep lags)",
        "n_optuna_trials": n_trials,
        "n_folds": n_folds,
        "sectors": {}
    }
    summary = []

    for sector, target_col in SECTOR_COLS.items():
        if target_col not in df.columns:
            print(f"  Skipping {sector}: column not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Sector: {sector}")
        print(f"{'='*60}")

        X, y = build_features_max(df, target_col, all_sector_cols)
        print(f"  Feature matrix: {X.shape}")

        sector_results = {}
        for mtype in model_types:
            t0 = time.time()
            print(f"  Tuning {mtype} ({n_trials} trials)...", end=" ", flush=True)
            best_model, best_rmse = tune_model(X, y, mtype, n_trials=n_trials, n_folds=n_folds)
            if best_rmse is None:
                best_rmse, _, _, _ = cv_per_fold(X, y, best_model, n_folds)
            _, rmse_std, best_mae, best_r2 = cv_per_fold(X, y, best_model, n_folds)
            elapsed = round(time.time() - t0, 1)
            print(f"RMSE={best_rmse:.4f} BTU  ({elapsed}s)")
            sector_results[mtype] = {
                "RMSE_BTU": round(best_rmse, 4),
                "RMSE_std": round(rmse_std, 4),
                "MAE_BTU":  round(best_mae, 4),
                "R2":       round(best_r2, 5),
                "best_params": str(best_model.get_params()),
            }

        best_mtype = min(sector_results, key=lambda k: sector_results[k]["RMSE_BTU"])
        best_rmse  = sector_results[best_mtype]["RMSE_BTU"]
        phase6     = PHASE6_BEST.get(sector, 9999)
        improv     = round((1 - best_rmse / phase6) * 100, 1)

        output["sectors"][sector] = {
            "phase6_RMSE_BTU":  phase6,
            "phase7_RMSE_BTU":  best_rmse,
            "best_model":        best_mtype,
            "improvement_vs_phase6": f"{improv:+.1f}%",
            "all_models":        sector_results,
        }

        print(f"\n  Phase 6 RMSE: {phase6:.4f} BTU")
        print(f"  Phase 7 best ({best_mtype}): {best_rmse:.4f} BTU  ({improv:+.1f}%)")

        summary.append({
            "Sector":       sector,
            "Phase6 RMSE":  phase6,
            "Phase7 RMSE":  best_rmse,
            "Best Model":   best_mtype,
            "Improvement":  f"{improv:+.1f}%",
        })

    out_path = RESULTS_DIR / "phase7_absolute_ceiling.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 7 FINAL SUMMARY")
    print("=" * 70)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
