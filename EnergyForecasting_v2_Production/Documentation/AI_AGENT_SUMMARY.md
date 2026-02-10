# ML-EnergyForecasting-Framework: Issues & Fixes Log

> **Purpose**: This document serves as a memory/changelog for AI agents working on this project.  
> **Last Updated**: January 18, 2026

---

## Executive Summary

This project had **fundamental flaws** that invalidated its R²=0.98 results. The high accuracy was an illusion caused by data leakage, mathematical errors, and improper methodology.

---

## Critical Issues Found

### Issue 1: Data Leakage (Target Contamination)
| Feature | Problem |
|---------|---------|
| `total_energy` | Sum of all sectors at time t includes target sector at time t |
| `res_com_ratio` | Uses residential and commercial at time t |
| `ind_trans_sum` | Uses industrial and transportation at time t |
| `sector_std` | Standard deviation across sectors at time t |

**Impact**: Model learns identity function y=y, not an actual forecast.

### Issue 2: Scale Inconsistency
- Baseline MSE from paper: 2.61 (raw Trillion BTU units)
- Reported "improved" MSE: 0.0031 (standardized Z-score units)
- **99.99% improvement claim is scientifically dishonest** - comparing different units

### Issue 3: The "Stationarity via Standardization" Fallacy (Proof 6)
**Claim**: "Standardization (z-score) induces weak stationarity"

**Mathematical Error**: 
- Standardization `z_t = (y_t - μ) / σ` does NOT remove trends
- If original has trend `y_t = αt`, standardized still has trend `z_t = (αt - μ) / σ`
- Mean of z_t depends on t, violating stationarity

**Correction**: Stationarity requires differencing `Δy_t = y_t - y_{t-1}` or explicit detrending

### Issue 4: Inadequate Seasonal Modeling (Harmonics Mismatch)
**Claim**: Single 12-month sin/cos captures bimodal (Winter/Summer) peaks

**Mathematical Error**:
- Single sine wave = one peak and one trough per year
- Residential energy has TWO peaks (Winter heating + Summer cooling)
- Cannot fit bimodal pattern with single harmonic

**Correction**: Need higher harmonics (6-month cycle) or explicit binary seasonal flags

### Issue 5: Look-Ahead Bias in Rolling Features
**Claim**: `ind_rolling_mean` uses 6-month window

**Implementation Error**:
- Formula: `(1/6) Σ Industrial_{t-i}` for i=0 to 5
- Window includes value at time t (current timestep)
- Industrial data at time t released simultaneously with Commercial target

**Correction**: Use shifted windows `t-1` to `t-6` (only past data)

### Issue 6: Invalid Statistical Significance Testing
**Claim**: Paired t-test proves model is better (p < 0.001)

**Statistical Error**:
- T-test assumes i.i.d. residuals
- Time series residuals have autocorrelation
- Autocorrelation inflates t-statistic → artificially low p-values

**Correction**: Use Diebold-Mariano test for time series comparison

### Issue 7: Overfitting
- Tree models showed 209x gap between training and test error
- Dataset only has ~600 samples
- 100-tree ensembles are overkill for this data size

### Issue 8: Documentation Contradictions
- Main report: OMP winner with MSE 0.0031
- Summary section: XGBoost winner with MSE 1.42
- These differ by factor of 450x

---

## Flaws Summary Table

| Category | Error | Impact |
|----------|-------|--------|
| Data Leakage | `total_energy`, `res_com_ratio` include target y_t | Model learns identity function |
| Math Theory | Standardization = Stationarity | False theorem, models fail to generalize |
| Feature Logic | 12-month Sin/Cos for bimodal data | Single sine can't fit two peaks |
| Causality | `rolling(6)` includes t | Look-ahead bias |
| Statistics | T-test on autocorrelated residuals | Invalid significance claims |

---

## Changes Made

### New Files Created

1. **`corrected_feature_engineering.py`**
   - Removed all leaked features
   - Implemented proper lagged features (shift by 1+)
   - 20 valid features using ONLY past data

2. **`corrected_model_training.py`**
   - Inverse transforms predictions before MSE calculation
   - Reports MSE in original Trillion BTU units
   - Reduced model complexity for dataset size

3. **`Results/tables/corrected_model_comparison.csv`**
   - Valid results with proper metrics

### Valid Features Implemented

```
Temporal: year, month, quarter, month_sin, month_cos
Autoregressive: target_lag_1, target_lag_2, target_lag_3, target_lag_12, target_lag_13
Rolling (shifted): rolling_mean_3, rolling_mean_6, rolling_mean_12, rolling_std_3
Cross-sector (lagged): total_energy_lag_1, total_energy_lag_12, 
                       residential_lag_1, industrial_lag_1, transportation_lag_1
Year-over-year: yoy_change
```

### Corrected Results

| Model | R² | MSE (Trillion BTU) |
|-------|----|--------------------|
| OMP | 0.777 | 4,390 |
| Lasso | 0.753 | 4,851 |
| Ridge | 0.726 | 5,384 |

**Note**: R²=0.78 is realistic for ~600 samples with proper methodology.

---

## Remaining Work for Future Agents

1. **Add higher harmonics** for bimodal seasonality (6-month sin/cos)
2. **Implement differencing** for proper stationarity
3. **Use Diebold-Mariano test** instead of t-test for model comparison
4. **Update mathematical proofs** to remove false claims
5. **Regenerate all figures** with corrected methodology
6. **Re-run cross-sector analysis** without leaked features

---

## Context for AI Agents

When working on this project:
- NEVER use cross-sector features from current timestep
- ALWAYS shift rolling windows by at least 1 period
- ALWAYS inverse-transform before calculating MSE
- The base paper MSE=2.61 is in raw Trillion BTU units
- Dataset has 633 samples spanning 1973-2025

---

## Update: February 7, 2026

### Tasks Accomplished

1. **Comprehensive Model Evaluation**
   - Calculated RMSE, MAE, R-squared, and Adjusted R-squared for the existing forecasting model.
   - Performed 5-fold cross-validation to assess model stability and generalization.

2. **Advanced Visualization and Diagnostic Plots**
   - Generated **Actual vs. Predicted** scatter plots to visualize model accuracy.
   - Created **Residuals vs. Predicted** and **Residual Distribution** (histogram/KDE) plots to diagnose error patterns and homoscedasticity.
   - Generated **Feature Importance** plots to identify key drivers of energy consumption.
   - Implemented **Partial Dependence Plots (PDP)** for the top features to visualize their non-linear relationship with the target.

3. **Overfitting Mitigation & Hyperparameter Tuning**
   - **Diagnosis**: Identified overfitting in the initial model (significant gap between CV and Test performance).
   - **Regularization**: Transitioned to a strictly **Lasso Regression** framework as per project requirements.
   - **Tuning**: Conducted a `GridSearchCV` for the Lasso `alpha` parameter.
   - **Optimization**: Found optimal `alpha=1.0`, effectively reducing model complexity and improving test performance.

4. **Model & Artifact Refresh**
   - Replaced the original `final_model.pkl` with the tuned Lasso version.
   - Updated `final_metrics.json` with the new performance benchmarks.
   - Regenerated all diagnostic plots (`lasso_actual_vs_predicted.png`, `lasso_residuals_plot.png`, etc.) to match the tuned model.

### Updated Performance (Lasso α=1.0)

| Metric | Value |
|--------|-------|
| RMSE   | 137.64 |
| MAE    | 106.15 |
| R²     | 0.859 |
| CV RMSE| 96.40  |

**Note**: The tuned Lasso model shows a better R² (0.859 vs 0.840) and lower RMSE on the test set compared to the original baseline.

---

## Update: February 8, 2026

### Current Project State & Reorganization
The project has been streamlined to focus on the most advanced implementation. Redundant and modular files have been moved to the `Archive/` folder.

#### Active "Final Version" Components
- **`final_complete_project.py`**: The master execution script (Multi-harmonic features, TimeSeries CV, Diebold-Mariano testing).
- **`Dataset/NEW DATA SET-1.xlsx`**: Primary data source.
- **`Dataset/artifacts/`**: Contains the latest `final_model.pkl` and `preprocess_pipeline.pkl`.
- **`Results/`**: Most recent performance tables and diagnostic plots.

#### Actions Taken
1. **Cleaned Workspace**: Moved `corrected_feature_engineering.py`, `corrected_model_training.py`, `demo.py`, and outdated logs to `Archive/`.
2. **Analysis**: Verified that `final_complete_project.py` contains all the "Corrected" logic from previous iterations, making the standalone "corrected_" files obsolete.
3. **Comparison**: Identified that the current framework significantly improves upon the base paper by adding XGBoost/LightGBM, multi-harmonic seasonality, and TimeSeriesSplit validation.
4. **Model Verification**: Confirmed that while individual models (Lasso, Ridge, etc.) are compared in the results table, only the "Best" model is currently saved as a `.pkl` artifact to maintain efficiency.

