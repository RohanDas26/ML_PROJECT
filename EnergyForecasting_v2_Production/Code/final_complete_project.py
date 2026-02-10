"""
Complete Energy Forecasting Project - Final Implementation
===========================================================

This script implements ALL fixes and finds the BEST model:
1. Proper lagged features (no data leakage)
2. 6-month harmonics for bimodal seasonality
3. Differencing for stationarity
4. Hyperparameter tuning via cross-validation
5. Diebold-Mariano test for statistical significance
6. Multiple model comparison including XGBoost

Author: Final Corrected Version
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, OrthogonalMatchingPursuit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not available, will skip")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠ LightGBM not available, will skip")

print("=" * 80)
print("COMPLETE ENERGY FORECASTING - FINAL IMPLEMENTATION")
print("=" * 80)


# Base paper reference
BASE_PAPER_MSE_RIDGE = 2.61


def load_and_preprocess_data(filepath):
    """Load and preprocess the energy data."""
    df_raw = pd.read_excel(filepath, sheet_name='Sheet1', skiprows=1)
    df_raw = df_raw.iloc[1:].reset_index(drop=True)
    df_raw.columns = ['Month'] + list(df_raw.columns[1:])
    
    df = df_raw[[
        'Month',
        'Total Energy Consumed by the Residential Sector',
        'Total Energy Consumed by the Commercial Sector',
        'Total Energy Consumed by the Industrial Sector',
        'Total Energy Consumed by the Transportation Sector'
    ]].copy()
    
    df.columns = ['Month', 'Residential', 'Commercial', 'Industrial', 'Transportation']
    df['Month'] = pd.to_datetime(df['Month'])
    
    for col in ['Residential', 'Commercial', 'Industrial', 'Transportation']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values('Month').reset_index(drop=True)
    
    return df


def create_complete_features(df, target_sector, use_differencing=False):
    """
    Create complete feature set with ALL fixes:
    - Proper lagged features (no leakage)
    - 6-month AND 12-month harmonics (bimodal seasonality)
    - Optional differencing for stationarity
    """
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING: {target_sector}")
    print(f"{'='*60}")
    
    features_df = pd.DataFrame()
    features_df['Month'] = df['Month']
    
    target = df[target_sector].copy()
    
    # ============================================
    # DIFFERENCING FOR STATIONARITY (Optional)
    # ============================================
    if use_differencing:
        target_diff = target.diff()
        features_df['target'] = target_diff
        print("✓ Using first-order differencing for stationarity")
    else:
        features_df['target'] = target
    
    # ============================================
    # TEMPORAL FEATURES
    # ============================================
    features_df['year'] = df['Month'].dt.year
    features_df['month'] = df['Month'].dt.month
    features_df['quarter'] = df['Month'].dt.quarter
    
    # ============================================
    # MULTI-HARMONIC SEASONALITY (FIX #2)
    # 12-month cycle (annual) + 6-month cycle (bimodal)
    # ============================================
    month = df['Month'].dt.month
    
    # Annual cycle (12-month)
    features_df['month_sin_12'] = np.sin(2 * np.pi * month / 12)
    features_df['month_cos_12'] = np.cos(2 * np.pi * month / 12)
    
    # Semi-annual cycle (6-month) - CRITICAL for bimodal patterns
    features_df['month_sin_6'] = np.sin(4 * np.pi * month / 12)
    features_df['month_cos_6'] = np.cos(4 * np.pi * month / 12)
    
    # Quarterly cycle (3-month) for additional resolution
    features_df['month_sin_3'] = np.sin(8 * np.pi * month / 12)
    features_df['month_cos_3'] = np.cos(8 * np.pi * month / 12)
    
    print("✓ Multi-harmonic seasonality: 12-month, 6-month, 3-month cycles")
    
    # ============================================
    # AUTOREGRESSIVE FEATURES (Lagged Target)
    # ============================================
    for lag in [1, 2, 3, 6, 12, 13, 24]:
        features_df[f'target_lag_{lag}'] = target.shift(lag)
    
    print("✓ Autoregressive lags: 1, 2, 3, 6, 12, 13, 24 months")
    
    # ============================================
    # ROLLING STATISTICS (FIX #3 - Properly shifted)
    # ============================================
    # CRITICAL: shift(1) ensures we only use PAST data
    features_df['rolling_mean_3'] = target.shift(1).rolling(window=3).mean()
    features_df['rolling_mean_6'] = target.shift(1).rolling(window=6).mean()
    features_df['rolling_mean_12'] = target.shift(1).rolling(window=12).mean()
    features_df['rolling_std_3'] = target.shift(1).rolling(window=3).std()
    features_df['rolling_std_6'] = target.shift(1).rolling(window=6).std()
    features_df['rolling_std_12'] = target.shift(1).rolling(window=12).std()
    
    # Exponential moving average (more recent data weighted higher)
    features_df['ema_3'] = target.shift(1).ewm(span=3).mean()
    features_df['ema_6'] = target.shift(1).ewm(span=6).mean()
    features_df['ema_12'] = target.shift(1).ewm(span=12).mean()
    
    print("✓ Rolling/EMA features (properly shifted): 3, 6, 12 months")
    
    # ============================================
    # LAGGED TOTAL ENERGY (Valid)
    # ============================================
    total_energy = df['Residential'] + df['Commercial'] + df['Industrial'] + df['Transportation']
    
    features_df['total_energy_lag_1'] = total_energy.shift(1)
    features_df['total_energy_lag_12'] = total_energy.shift(12)
    
    print("✓ Total energy lags: 1, 12 months")
    
    # ============================================
    # LAGGED OTHER SECTORS
    # ============================================
    other_sectors = ['Residential', 'Commercial', 'Industrial', 'Transportation']
    other_sectors.remove(target_sector)
    
    for sector in other_sectors:
        features_df[f'{sector.lower()}_lag_1'] = df[sector].shift(1)
        features_df[f'{sector.lower()}_lag_12'] = df[sector].shift(12)
    
    print(f"✓ Other sector lags: {', '.join(other_sectors)}")
    
    # ============================================
    # MOMENTUM AND CHANGE FEATURES
    # ============================================
    features_df['yoy_change'] = (target.shift(1) - target.shift(13)) / target.shift(13)
    features_df['mom_change'] = (target.shift(1) - target.shift(2)) / target.shift(2)
    features_df['momentum_3'] = target.shift(1) - target.shift(4)
    features_df['momentum_6'] = target.shift(1) - target.shift(7)
    
    print("✓ Momentum features: YoY, MoM, 3-month, 6-month")
    
    # ============================================
    # SEASONAL FLAGS (Binary)
    # ============================================
    features_df['is_winter'] = month.isin([12, 1, 2]).astype(int)
    features_df['is_summer'] = month.isin([6, 7, 8]).astype(int)
    features_df['is_spring'] = month.isin([3, 4, 5]).astype(int)
    features_df['is_fall'] = month.isin([9, 10, 11]).astype(int)
    
    print("✓ Seasonal binary flags: Winter, Summer, Spring, Fall")
    
    # Drop NaN rows
    n_before = len(features_df)
    features_df = features_df.dropna().reset_index(drop=True)
    n_after = len(features_df)
    
    feature_cols = [c for c in features_df.columns if c not in ['Month', 'target']]
    print(f"\n✓ Total features: {len(feature_cols)}")
    print(f"✓ Samples: {n_after} (dropped {n_before - n_after} due to lagging)")
    
    return features_df


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for forecast comparison (FIX #4).
    
    H0: Both forecasts have equal accuracy
    H1: Forecast 1 is more accurate
    
    Args:
        e1: Errors from model 1
        e2: Errors from model 2
        h: Forecast horizon
    
    Returns:
        DM statistic, p-value
    """
    d = e1**2 - e2**2
    n = len(d)
    
    # Calculate autocovariance
    mean_d = np.mean(d)
    gamma_0 = np.var(d)
    
    # Long-run variance estimate
    gamma_sum = gamma_0
    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0,1] if len(d) > k else 0
        gamma_sum += 2 * gamma_k
    
    # DM statistic
    dm_stat = mean_d / np.sqrt(gamma_sum / n)
    
    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def train_with_hyperparameter_tuning(X_train, y_train, X_test, y_test,
                                      y_train_original, y_test_original, scaler_y):
    """Train models with hyperparameter tuning and find the best one."""
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define models and their hyperparameter grids
    model_configs = {
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        'Lasso': {
            'model': Lasso(max_iter=10000),
            'params': {'alpha': [0.001, 0.01, 0.1, 1.0]}
        },
        'ElasticNet': {
            'model': ElasticNet(max_iter=10000),
            'params': {'alpha': [0.001, 0.01, 0.1], 'l1_ratio': [0.2, 0.5, 0.8]}
        },
        'OMP': {
            'model': OrthogonalMatchingPursuit(),
            'params': {'n_nonzero_coefs': [5, 10, 15, 20, 25]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {'n_estimators': [20, 50], 'max_depth': [3, 5, 7], 'min_samples_leaf': [5, 10]}
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'params': {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance']}
        }
    }
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        model_configs['XGBoost'] = {
            'model': xgb.XGBRegressor(random_state=42, verbosity=0),
            'params': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
        }
    
    # Add LightGBM if available
    if HAS_LIGHTGBM:
        model_configs['LightGBM'] = {
            'model': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'params': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
        }
    
    results = []
    best_models = {}
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING & MODEL SELECTION")
    print("=" * 80)
    
    for name, config in model_configs.items():
        print(f"\n{'-'*40}")
        print(f"Tuning: {name}")
        print(f"{'-'*40}")
        
        start_time = time.time()
        
        # Grid search with time series CV
        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        # Predictions
        y_pred_scaled = best_model.predict(X_test)
        y_train_pred_scaled = best_model.predict(X_train)
        
        # Inverse transform
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred_original = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        
        # Metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        train_mse = mean_squared_error(y_train_original, y_train_pred_original)
        
        train_time = time.time() - start_time
        
        results.append({
            'Model': name,
            'Best_Params': str(grid.best_params_),
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Train_MSE': train_mse,
            'Overfit_Ratio': mse / train_mse if train_mse > 0 else float('inf'),
            'CV_Score': -grid.best_score_,
            'Time': train_time,
            'Errors': y_test_original - y_pred_original
        })
        
        print(f"  Best params: {grid.best_params_}")
        print(f"  MSE: {mse:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    
    return pd.DataFrame(results), best_models


def perform_dm_tests(results_df, baseline_name='Ridge'):
    """Perform Diebold-Mariano tests against baseline."""
    print("\n" + "=" * 80)
    print(f"DIEBOLD-MARIANO TESTS (vs {baseline_name})")
    print("=" * 80)
    
    baseline_errors = results_df[results_df['Model'] == baseline_name]['Errors'].values[0]
    
    dm_results = []
    for _, row in results_df.iterrows():
        if row['Model'] == baseline_name:
            continue
            
        dm_stat, p_value = diebold_mariano_test(baseline_errors, row['Errors'])
        
        significance = ""
        if p_value < 0.01:
            significance = "***"
        elif p_value < 0.05:
            significance = "**"
        elif p_value < 0.10:
            significance = "*"
        
        dm_results.append({
            'Model': row['Model'],
            'DM_Statistic': dm_stat,
            'P_Value': p_value,
            'Significance': significance,
            'Better': 'Yes' if dm_stat > 0 and p_value < 0.05 else 'No'
        })
        
        print(f"  {row['Model']}: DM={dm_stat:.3f}, p={p_value:.4f} {significance}")
    
    return pd.DataFrame(dm_results)


def main():
    """Main execution - complete project."""
    
    # Load data
    print("\n" + "=" * 80)
    print("DATA LOADING")
    print("=" * 80)
    df = load_and_preprocess_data('Dataset/NEW DATA SET-1.xlsx')
    print(f"✓ Loaded {len(df)} samples: {df['Month'].min()} to {df['Month'].max()}")
    
    # Process all sectors
    all_results = {}
    
    for target_sector in ['Commercial', 'Residential', 'Industrial', 'Transportation']:
        print(f"\n\n{'#'*80}")
        print(f"# SECTOR: {target_sector.upper()}")
        print(f"{'#'*80}")
        
        # Create features
        features_df = create_complete_features(df, target_sector, use_differencing=False)
        
        # Train/test split
        feature_cols = [c for c in features_df.columns if c not in ['Month', 'target']]
        X = features_df[feature_cols].values
        y = features_df['target'].values
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        y_train_orig, y_test_orig = y_train.copy(), y_test.copy()
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        
        # Train and evaluate
        results_df, best_models = train_with_hyperparameter_tuning(
            X_train_s, y_train_s, X_test_s, y_test_s,
            y_train_orig, y_test_orig, scaler_y
        )
        
        # DM tests
        dm_results = perform_dm_tests(results_df)
        
        all_results[target_sector] = {
            'results': results_df,
            'dm_tests': dm_results,
            'best_models': best_models,
            'feature_cols': feature_cols
        }
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    for sector, data in all_results.items():
        results = data['results']
        best_idx = results['MSE'].idxmin()
        best = results.loc[best_idx]
        
        print(f"\n{sector}:")
        print(f"  Best Model: {best['Model']}")
        print(f"  MSE: {best['MSE']:.2f} | RMSE: {best['RMSE']:.2f} | R²: {best['R2']:.4f}")
        print(f"  Best Params: {best['Best_Params']}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Combine all sector results
    final_results = []
    for sector, data in all_results.items():
        for _, row in data['results'].iterrows():
            final_results.append({
                'Sector': sector,
                'Model': row['Model'],
                'MSE': row['MSE'],
                'RMSE': row['RMSE'],
                'MAE': row['MAE'],
                'R2': row['R2'],
                'Best_Params': row['Best_Params'],
                'Overfit_Ratio': row['Overfit_Ratio']
            })
    
    final_df = pd.DataFrame(final_results)
    final_df.to_csv('Results/tables/final_model_comparison.csv', index=False)
    print("✓ Saved Results/tables/final_model_comparison.csv")
    
    # Save DM test results
    dm_all = []
    for sector, data in all_results.items():
        for _, row in data['dm_tests'].iterrows():
            dm_all.append({
                'Sector': sector,
                'Model': row['Model'],
                'DM_Statistic': row['DM_Statistic'],
                'P_Value': row['P_Value'],
                'Significant': row['Better']
            })
    
    dm_df = pd.DataFrame(dm_all)
    dm_df.to_csv('Results/tables/diebold_mariano_tests.csv', index=False)
    print("✓ Saved Results/tables/diebold_mariano_tests.csv")
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETE")
    print("=" * 80)
    print("""
All fixes implemented:
  ✓ No data leakage (proper lagged features)
  ✓ Multi-harmonic seasonality (6, 12-month cycles)
  ✓ Properly shifted rolling features
  ✓ Diebold-Mariano statistical tests
  ✓ Hyperparameter tuning with time-series CV
  ✓ MSE in original Trillion BTU units
""")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
