# Energy Forecasting -- Full Test Report

**Generated**: 2026-02-11 13:55
**Dataset**: 633 samples (1973-2025), 4 sectors
**Methodology**: Optuna Bayesian HP tuning, 80 trials/model, 5-fold TimeSeriesSplit
**Leakage Tests**: 36/36 PASSED

## 1. Executive Summary

| Sector | Best Model | RMSE (TBTU) | R2 | MAPE (%) |
|--------|-----------|-------------|------|----------|
| Residential | SVR | 123.59 | 0.8784 | 5.6 |
| Commercial | OMP | 61.03 | 0.8132 | 3.51 |
| Industrial | KNN | 67.61 | 0.643 | 1.88 |
| Transportation | Lasso | 94.5 | 0.636 | 2.42 |

## 2. Good vs Bad Model Comparison

| Sector | Good Model | Good R2 | Bad Model | Bad R2 | Gap |
|--------|-----------|---------|----------|--------|-----|
| Residential | SVR | 0.8784 | KNN | 0.8235 | 0.0549 |
| Commercial | OMP | 0.8132 | KNN | 0.6742 | 0.139 |
| Industrial | KNN | 0.643 | RandomForest | 0.5232 | 0.1198 |
| Transportation | Lasso | 0.636 | RandomForest | 0.2763 | 0.3597 |

## 3. Stationarity Analysis (Phase 2)

| Sector | ADF p-value | KPSS p-value | Stationary? | After Diff? | Trend Str. | Seasonal Str. |
|--------|------------|-------------|-------------|-------------|------------|--------------|
| Residential | 0.3707 | 0.0100 | False | True | 0.809 | 0.938 |
| Commercial | 0.4068 | 0.0100 | False | True | 0.979 | 0.916 |
| Industrial | 0.1502 | 0.0100 | False | True | 0.874 | 0.748 |
| Transportation | 0.5494 | 0.0100 | False | True | 0.972 | 0.701 |

## 4. SHAP Interpretability (Phase 6)

### Residential -- SVR (GOOD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | month_cos_12 | 0.3997 |
| 2 | target_lag_1 | 0.1687 |
| 3 | rolling_mean_6 | 0.0969 |
| 4 | is_spring | 0.094 |
| 5 | month_cos_6 | 0.0838 |

### Residential -- KNN (BAD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | is_winter | 0.0307 |
| 2 | year | 0.028 |
| 3 | total_energy_lag_12 | 0.027 |
| 4 | transportation_lag_12 | 0.0252 |
| 5 | month_cos_12 | 0.0247 |

### Commercial -- OMP (GOOD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | ema_12 | 0.7029 |
| 2 | rolling_mean_12 | 0.6631 |
| 3 | target_lag_1 | 0.4879 |
| 4 | is_fall | 0.1497 |
| 5 | month_cos_12 | 0.1472 |

### Commercial -- KNN (BAD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | year | 0.1043 |
| 2 | transportation_lag_1 | 0.0524 |
| 3 | transportation_lag_12 | 0.0446 |
| 4 | target_lag_24 | 0.0416 |
| 5 | ema_12 | 0.0368 |

### Industrial -- KNN (GOOD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | yoy_change | 0.0205 |
| 2 | target_lag_12 | 0.0188 |
| 3 | target_lag_1 | 0.0181 |
| 4 | month_sin_3 | 0.018 |
| 5 | target_lag_2 | 0.017 |

### Industrial -- RandomForest (BAD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | target_lag_12 | 0.1545 |
| 2 | ema_3 | 0.1283 |
| 3 | ema_6 | 0.0689 |
| 4 | month_sin_12 | 0.06 |
| 5 | rolling_mean_12 | 0.0444 |

### Transportation -- Lasso (GOOD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | target_lag_12 | 0.2832 |
| 2 | target_lag_2 | 0.2704 |
| 3 | target_lag_3 | 0.2464 |
| 4 | rolling_mean_12 | 0.1885 |
| 5 | is_spring | 0.1004 |

### Transportation -- RandomForest (BAD)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | target_lag_12 | 0.9757 |
| 2 | rolling_mean_12 | 0.1041 |
| 3 | target_lag_2 | 0.0492 |
| 4 | target_lag_24 | 0.0321 |
| 5 | ema_6 | 0.0293 |


## 5. Robustness Testing (Phase 7)

### 5a. Bootstrap 95% Confidence Intervals

| Sector | Model | Type | RMSE CI | R2 CI |
|--------|-------|------|---------|-------|
| Residential | SVR | good | [106.53, 142.02] | [0.8276, 0.9127] |
| Residential | KNN | bad | [117.78, 169.2] | [0.7458, 0.8964] |
| Commercial | OMP | good | [52.94, 68.89] | [0.74, 0.8633] |
| Commercial | KNN | bad | [70.11, 90.61] | [0.5496, 0.7658] |
| Industrial | KNN | good | [52.37, 85.01] | [0.508, 0.7408] |
| Industrial | RandomForest | bad | [62.34, 91.06] | [0.4256, 0.6679] |
| Transportation | Lasso | good | [55.12, 135.71] | [0.4536, 0.8252] |
| Transportation | RandomForest | bad | [92.37, 174.55] | [-0.0045, 0.5065] |

### 5b. Noise Sensitivity (RMSE at noise levels)

| Sector | Model | 0% | 1% | 5% | 10% | 20% | 50% |
|--------|-------|----|----|----|----|-----|-----|
| Residential | SVR | 123.07 | 122.88 | 124.31 | 134.59 | 153.0 | 264.89 |
| Residential | KNN | 143.87 | 143.95 | 145.7 | 144.34 | 144.99 | 147.05 |
| Commercial | OMP | 61.03 | 61.31 | 64.29 | 71.24 | 108.42 | 194.48 |
| Commercial | KNN | 80.6 | 80.24 | 79.97 | 82.31 | 80.23 | 75.83 |
| Industrial | KNN | 68.74 | 68.87 | 67.81 | 69.18 | 69.01 | 68.34 |
| Industrial | RandomForest | 75.02 | 75.5 | 75.48 | 76.23 | 82.03 | 97.53 |
| Transportation | Lasso | 94.42 | 94.37 | 94.77 | 93.49 | 99.01 | 120.15 |
| Transportation | RandomForest | 133.25 | 133.47 | 134.31 | 137.38 | 137.3 | 158.3 |

## 6. Residual Diagnostics

| Sector | DW | LB p | Autocorr? | SW p | Normal? |
|--------|----|----|-----------|------|---------|
| Residential | 1.999 | 0.9020 | No | 0.0090 | False |
| Commercial | 1.768 | 0.9169 | No | 0.4414 | True |
| Industrial | 1.486 | 0.2499 | No | 0.0000 | False |
| Transportation | 0.888 | 0.0000 | YES | 0.0000 | False |

## 7. Conclusions & Recommendations

1. **Linear models dominate**: SVR(linear), OMP, and Lasso outperform trees on 633 samples.
2. **Residential/Commercial well-modeled** (R2 > 0.81) due to strong seasonal structure.
3. **Industrial/Transportation harder** (R2 ~ 0.64) -- need exogenous variables (GDP, HDD/CDD).
4. **All sectors non-stationary** -- differencing improves stationarity (Phase 2 confirmed).
5. **Transportation has autocorrelated residuals** -- needs temporal structure (ARIMA residuals).
6. **Leakage-free**: 36/36 automated tests confirm no data contamination.
