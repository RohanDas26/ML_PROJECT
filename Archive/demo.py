
import os
import json
import joblib
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, OrthogonalMatchingPursuit

import optuna

# --- Tree-based models imports ---
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")
print("running demo.py...")

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    data_path: str = "NEW DATA SET-1.xlsx"     # update if needed
    output_dir: str = "artifacts"

    # Choose ONE target column from the Excel header
    target_col: str = "Total Energy Consumed by the Residential Sector"

    # Leakage columns to drop (components of the target)
    leakage_cols: tuple = (
        "Primary Energy Consumed by the Residential Sector",
        "Electricity Sales to Ultimate Customers in the Residential Sector",
        "End-Use Energy Consumed by the Residential Sector",
        "Residential Sector Electrical System Energy Losses"
    )

    # Lag/seasonality features
    lags: tuple = (1, 12)          # previous month and previous year
    rolling_windows: tuple = (3, 12)

    # Hold-out split
    test_years: int = 14           # last N years as test
    min_rows_after_fe: int = 24    # after dropping NaNs from lags, ensure enough data


CFG = Config()


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    if ("Year" in cols and "Month" in cols):
        dt = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str), errors="coerce")
        df = df.drop(columns=["Year", "Month"])
        df.insert(0, "date", dt)
        return df

    c0, c1 = cols[0], cols[1]
    if df[c0].dtype.kind in "iu" and df[c1].dtype == object:
        dt = pd.to_datetime(df[c0].astype(str) + "-" + df[c1].astype(str), errors="coerce")
        df = df.drop(columns=[c0, c1])
        df.insert(0, "date", dt)
        return df

    for cand in ["date", "Date", "Month", "MONTH", "time", "Time"]:
        if cand in cols:
            dt = pd.to_datetime(df[cand], errors="coerce")
            df = df.drop(columns=[cand])
            df.insert(0, "date", dt)
            return df

    c0 = cols[0]
    first_val = str(df[c0].iloc[0]).strip()
    if first_val.lower() == "month":
        df = df.iloc[1:].reset_index(drop=True)
        dt = pd.to_datetime(df[c0], errors="coerce")
        df = df.drop(columns=[c0])
        df.insert(0, "date", dt)
        return df

    if pd.api.types.is_datetime64_any_dtype(df[c0]):
        df = df.rename(columns={c0: "date"})
        df = df[["date"] + [col for col in df.columns if col != "date"]]
        return df

    raise ValueError(f"Could not infer date columns. Columns found: {cols}")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != "date"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    num_cols = [c for c in df.columns if c != "date"]
    df[num_cols] = df[num_cols].mask(df[num_cols] < 0, np.nan)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int).astype(str)
    df["t"] = np.arange(len(df))
    return df


def add_lag_rolling_features(df: pd.DataFrame, target_col: str, lags=(1, 12), windows=(3, 12)) -> pd.DataFrame:
    df = df.copy()

    # Lags on target
    for L in lags:
        df[f"{target_col}__lag{L}"] = df[target_col].shift(L)

    # Rolling stats on target
    for w in windows:
        df[f"{target_col}__rollmean{w}"] = df[target_col].shift(1).rolling(window=w, min_periods=max(2, w//2)).mean()
        df[f"{target_col}__rollstd{w}"] = df[target_col].shift(1).rolling(window=w, min_periods=max(2, w//2)).std()

    # ===== FIX: LAG ALL OTHER SECTOR COLUMNS =====
    # Define columns belonging to other sectors that might leak current info
    sector_keywords = ["Commercial", "Industrial", "Transportation", "End-Use Sectors"]
    potential_leak_cols = [c for c in df.columns if any(k in c for k in sector_keywords) and c != target_col]
    
    # Lag them all by 1 month and drop original
    for col in potential_leak_cols:
        df[f"{col}__lag1"] = df[col].shift(1)
        df = df.drop(columns=[col])

    return df


def time_holdout_split(df: pd.DataFrame, test_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_date = df["date"].max()
    cutoff = last_date - pd.DateOffset(years=test_years)
    train_df = df[df["date"] <= cutoff].copy()
    test_df = df[df["date"] > cutoff].copy()
    return train_df, test_df


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if X[c].dtype.kind in "if"]
    cat_cols = [c for c in X.columns if X[c].dtype == object]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")


def evaluate(model: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred))
    }


# -----------------------------
# Main training routine
# -----------------------------
def main(cfg: Config):
    ensure_dir(cfg.output_dir)

    # 1) Load & Clean
    df_raw = pd.read_excel(cfg.data_path, sheet_name=0, header=1)
    df = make_datetime_index(df_raw)
    df = basic_clean(df)

    # 2) Remove explicit leakage columns
    leakage_cols = [c for c in cfg.leakage_cols if c in df.columns]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)

    # 3) Feature engineering
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column not found: {cfg.target_col}")

    df = add_time_features(df)
    df = add_lag_rolling_features(df, cfg.target_col, lags=cfg.lags, windows=cfg.rolling_windows)
    
    # Drop rows with NaNs created by lagging/rolling
    df = df.dropna().reset_index(drop=True)

    if len(df) < cfg.min_rows_after_fe:
        raise ValueError(f"Too few rows after feature engineering: {len(df)}")

    # 4) Save Clean Data
    df.to_csv(os.path.join(cfg.output_dir, "clean_data.csv"), index=False)

    # 5) Split
    train_df, test_df = time_holdout_split(df, cfg.test_years)
    train_df.to_csv(os.path.join(cfg.output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(cfg.output_dir, "test.csv"), index=False)

    # 6) Prepare X/y
    drop_cols = ["date", cfg.target_col]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[cfg.target_col].values
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[cfg.target_col].values

    preprocess = build_preprocess(X_train)
    joblib.dump(preprocess, os.path.join(cfg.output_dir, "preprocess_pipeline.pkl"))

    # 7) Baselines (Added XGBoost & LightGBM)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-3, max_iter=20000),
        "ElasticNet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    
    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, max_depth=-1, num_leaves=31, learning_rate=0.1, random_state=42)

    baseline_rows = []
    print("\nRunning Baselines...")
    for name, reg in models.items():
        print(f"  Evaluating {name}...")
        pipe = Pipeline([("prep", preprocess), ("reg", reg)])
        metrics = evaluate(pipe, X_train, y_train, X_test, y_test)
        baseline_rows.append({"model": name, **metrics})

    baseline_df = pd.DataFrame(baseline_rows).sort_values("rmse")
    baseline_df.to_csv(os.path.join(cfg.output_dir, "baseline_metrics.csv"), index=False)
    print("Baseline metrics saved.\n")
    print(baseline_df)

    # 8) Optuna tuning - Focused on Tree Models
    tscv = TimeSeriesSplit(n_splits=5)

    def tune_xgboost(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": 42
        }
        model = Pipeline([("prep", preprocess), ("reg", xgb.XGBRegressor(**params, verbosity=0))])
        scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=1)
        return -float(np.mean(scores))

    def tune_lightgbm(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 25, 60),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": 42,
            "verbose": -1,
            "force_row_wise": True
        }
        model = Pipeline([("prep", preprocess), ("reg", lgb.LGBMRegressor(**params))])
        scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=1)
        return -float(np.mean(scores))

    # Tune best performing tree model
    best_model_name = "ElasticNet"  # fallback
    best_params = {}
    
    if HAS_XGBOOST:
        print("\nTuning XGBoost (20 trials)...")
        study_xgb = optuna.create_study(direction="minimize")
        study_xgb.optimize(tune_xgboost, n_trials=20, show_progress_bar=False)
        print(f"Best XGBoost RMSE: {study_xgb.best_value:.4f}")
        best_model_name = "XGBoost"
        best_params = study_xgb.best_params
    
    if HAS_LIGHTGBM:
        print("\nTuning LightGBM (20 trials)...")
        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(tune_lightgbm, n_trials=20, show_progress_bar=False)
        print(f"Best LightGBM RMSE: {study_lgb.best_value:.4f}")
        
        # Compare and pick winner
        if HAS_XGBOOST and study_lgb.best_value < study_xgb.best_value:
            best_model_name = "LightGBM"
            best_params = study_lgb.best_params
            print(f"LightGBM selected (better than XGBoost)")
        elif not HAS_XGBOOST:
            best_model_name = "LightGBM"
            best_params = study_lgb.best_params
        else:
            print(f"XGBoost selected (better than LightGBM)")

    # 9) Retrain Best Model
    print(f"\nFinal training with {best_model_name}...")
    
    if best_model_name == "XGBoost":
        final_regressor = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
    elif best_model_name == "LightGBM":
        final_regressor = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
    else:
        final_regressor = ElasticNet(alpha=1.0, max_iter=20000)

    final_model = Pipeline([("prep", preprocess), ("reg", final_regressor)])
    final_metrics = evaluate(final_model, X_train, y_train, X_test, y_test)

    # Save
    joblib.dump(final_model, os.path.join(cfg.output_dir, "final_model.pkl"))
    with open(os.path.join(cfg.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f)

    print("\n------------------------------------------------")
    print(f"FINAL RESULTS ({best_model_name}):")
    print(f"RMSE: {final_metrics['rmse']:.4f}")
    print(f"MAE:  {final_metrics['mae']:.4f}")
    print(f"R2:   {final_metrics['r2']:.4f}")
    print("------------------------------------------------")

if __name__ == "__main__":
    main(CFG)