"""
ablation_test.py
================
Empirical proof: shows WHY Fourier > raw month integers for energy forecasting.

Ablation design:
  - Model: Ridge (SAME model the base paper used as winner)
  - CV: 10-fold TimeSeriesSplit + per-fold scaler (fully fair)
  - Dataset: same EIA data

  Condition A: Raw month integer (1-12) + year + lag_1 + lag_12
               --> closest to base paper's feature set
  Condition B: Fourier harmonics + year + lag_1 + lag_12
               --> our approach (just replacing month integer with Fourier)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

EXCEL_PATH = Path(r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\Dataset\USA ENGERY PREDICTION.xlsx")

SECTORS = {
    "Residential":    "Primary Energy Consumed by the Residential Sector",
    "Commercial":     "Primary Energy Consumed by the Commercial Sector",
    "Industrial":     "Primary Energy Consumed by the Industrial Sector",
    "Transportation": "Primary Energy Consumed by the Transportation Sector",
}

def load():
    df = pd.read_excel(EXCEL_PATH, skiprows=0)
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Month"})
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month"]).sort_values("Month").reset_index(drop=True)
    for col in SECTORS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_A_raw_month(df, target_col):
    """Condition A: raw month integer + year + lag_1 + lag_12."""
    feat = pd.DataFrame(index=df.index)
    feat["month"]  = df["Month"].dt.month    # 1-12 integer <- base paper
    feat["year"]   = df["Month"].dt.year
    feat["lag_1"]  = df[target_col].shift(1)
    feat["lag_12"] = df[target_col].shift(12)
    valid = feat.dropna().index
    return feat.loc[valid].values, df.loc[valid, target_col].values

def build_B_fourier(df, target_col):
    """Condition B: Fourier harmonics + year + lag_1 + lag_12."""
    feat = pd.DataFrame(index=df.index)
    t = np.arange(len(df))
    for k in range(1, 7):
        feat[f"sin_{k}"] = np.sin(2 * np.pi * k * t / 12)
        feat[f"cos_{k}"] = np.cos(2 * np.pi * k * t / 12)
    feat["year"]   = df["Month"].dt.year
    feat["lag_1"]  = df[target_col].shift(1)
    feat["lag_12"] = df[target_col].shift(12)
    valid = feat.dropna().index
    return feat.loc[valid].values, df.loc[valid, target_col].values

def cv_score(X, y, n_folds=10):
    """Per-fold scaling + TimeSeriesSplit. Returns (mean_rmse, std_rmse)."""
    tscv = TimeSeriesSplit(n_splits=n_folds)
    rmses = []
    for tr, te in tscv.split(X):
        sx = StandardScaler()
        X_tr = sx.fit_transform(X[tr]); X_te = sx.transform(X[te])
        sy = StandardScaler()
        y_tr = sy.fit_transform(y[tr].reshape(-1,1)).ravel()
        y_te = sy.transform(y[te].reshape(-1,1)).ravel()
        m = Ridge(alpha=1.0); m.fit(X_tr, y_tr)
        rmses.append(float(np.sqrt(mean_squared_error(y_te, m.predict(X_te)))))
    return float(np.mean(rmses)), float(np.std(rmses))

def main():
    df = load()
    print("ABLATION: Raw Month Integer vs Fourier Harmonics")
    print("Same model (Ridge), same CV (10-fold TimeSeriesSplit + per-fold scaler)")
    print("Only difference: how time / seasonality is encoded")
    print("=" * 70)

    rows = []
    for sector, col in SECTORS.items():
        if col not in df.columns:
            continue
        sub = df[["Month", col]].dropna().copy()
        X_a, y_a = build_A_raw_month(sub, col)
        X_b, y_b = build_B_fourier(sub, col)

        rmse_a, std_a = cv_score(X_a, y_a)
        rmse_b, std_b = cv_score(X_b, y_b)
        improv = round((1 - rmse_b / rmse_a) * 100, 1)

        print(f"\n  {sector}")
        print(f"    A) month-int + lag:  RMSE={rmse_a:.4f}  std={std_a:.4f}")
        print(f"    B) fourier  + lag:   RMSE={rmse_b:.4f}  std={std_b:.4f}")
        print(f"    Fourier improvement: {improv:+.1f}%")

        rows.append({"Sector": sector,
                     "A_Raw_Month_RMSE": round(rmse_a, 4),
                     "B_Fourier_RMSE": round(rmse_b, 4),
                     "Improvement": f"{improv:+.1f}%"})

    print("\n" + "=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
