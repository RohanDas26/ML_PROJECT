"""
Corrected Model Training for Energy Consumption Forecasting
============================================================

This script trains models using the corrected features and properly
calculates MSE in ORIGINAL UNITS for valid comparison with the base paper.

Key Fixes:
- Simpler models (Ridge, OMP) appropriate for ~600 samples
- Inverse transform predictions before calculating MSE
- Report MSE in Trillion BTU units (same as base paper)
- Proper time-series cross-validation

Base Paper Reference:
- Malakouti et al. (2025)
- Ridge Regression with 2 features
- Baseline MSE: 2.61 (in Trillion BTU units)

Author: Corrected Version
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, OrthogonalMatchingPursuit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import time
import warnings
warnings.filterwarnings('ignore')

# Import our corrected feature engineering
from corrected_feature_engineering import load_and_preprocess_data, create_valid_features, prepare_train_test_split, standardize_data

print("=" * 80)
print("CORRECTED MODEL TRAINING")
print("Proper MSE Calculation in Original Units (Trillion BTU)")
print("=" * 80)


# Base paper reference MSE (Malakouti et al., 2025)
BASE_PAPER_MSE_RIDGE = 2.61  # Trillion BTU


def train_and_evaluate_models(X_train, X_test, y_train, y_test, 
                               y_train_original, y_test_original, scaler_y):
    """
    Train multiple models and evaluate with PROPER metrics.
    
    CRITICAL: We inverse-transform predictions to calculate MSE in original units!
    """
    
    # Model configurations - SIMPLIFIED for 600-sample dataset
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'OMP': OrthogonalMatchingPursuit(n_nonzero_coefs=10),
        # Reduced complexity for tree models
        'RandomForest': RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=42),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5)
    }
    
    results = []
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{'-' * 40}")
        print(f"Training: {name}")
        print(f"{'-' * 40}")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict (in standardized space)
        y_pred_scaled = model.predict(X_test)
        y_train_pred_scaled = model.predict(X_train)
        
        # ============================================
        # CRITICAL FIX: Inverse transform to original units
        # ============================================
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred_original = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        
        # Calculate metrics in ORIGINAL UNITS (Trillion BTU)
        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        r2_original = r2_score(y_test_original, y_pred_original)
        
        # Training MSE for overfitting detection
        train_mse_original = mean_squared_error(y_train_original, y_train_pred_original)
        
        # Calculate improvement over base paper
        improvement_pct = (BASE_PAPER_MSE_RIDGE - mse_original) / BASE_PAPER_MSE_RIDGE * 100
        
        # Overfitting ratio (train vs test MSE)
        overfit_ratio = mse_original / train_mse_original if train_mse_original > 0 else float('inf')
        
        results.append({
            'Model': name,
            'MSE (Trillion BTU)': mse_original,
            'RMSE (Trillion BTU)': rmse_original,
            'MAE (Trillion BTU)': mae_original,
            'R²': r2_original,
            'Train MSE': train_mse_original,
            'Overfit Ratio': overfit_ratio,
            'Base Paper MSE': BASE_PAPER_MSE_RIDGE,
            'Improvement (%)': improvement_pct,
            'Time (sec)': train_time
        })
        
        print(f"  MSE (Original Units): {mse_original:.4f} Trillion BTU")
        print(f"  RMSE:                 {rmse_original:.4f} Trillion BTU")
        print(f"  R²:                   {r2_original:.4f}")
        print(f"  Train MSE:            {train_mse_original:.4f}")
        print(f"  Overfit Ratio:        {overfit_ratio:.2f}x")
        print(f"  Base Paper MSE:       {BASE_PAPER_MSE_RIDGE:.2f}")
        print(f"  Improvement:          {improvement_pct:.2f}%")
        
        # Warning for overfitting
        if overfit_ratio > 3:
            print(f"  ⚠ WARNING: High overfit ratio ({overfit_ratio:.1f}x) suggests overfitting!")
    
    return pd.DataFrame(results)


def cross_validate_models(X_train, y_train, y_train_original, scaler_y, n_splits=5):
    """
    Time-series cross-validation with proper inverse transforms.
    """
    print("\n" + "=" * 80)
    print("TIME-SERIES CROSS-VALIDATION")
    print("=" * 80)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = {
        'Ridge': Ridge(alpha=1.0),
        'OMP': OrthogonalMatchingPursuit(n_nonzero_coefs=10),
    }
    
    cv_results = []
    
    for name, model in models.items():
        fold_mses = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            y_val_orig = y_train_original[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred_scaled = model.predict(X_val)
            
            # Inverse transform
            y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            fold_mse = mean_squared_error(y_val_orig, y_pred_orig)
            fold_mses.append(fold_mse)
        
        cv_results.append({
            'Model': name,
            'CV MSE Mean': np.mean(fold_mses),
            'CV MSE Std': np.std(fold_mses),
            'CV Folds': fold_mses
        })
        
        print(f"\n{name}:")
        print(f"  CV MSE: {np.mean(fold_mses):.4f} ± {np.std(fold_mses):.4f}")
    
    return pd.DataFrame(cv_results)


def main():
    """Main training execution."""
    
    # Load and process data
    print("\nLoading and processing data...")
    df = load_and_preprocess_data('Dataset/NEW DATA SET-1.xlsx')
    
    # Create corrected features
    target_sector = 'Commercial'
    features_df = create_valid_features(df, target_sector)
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(features_df)
    
    # Keep original values for proper MSE calculation
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()
    
    # Standardize
    X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = standardize_data(
        X_train, X_test, y_train, y_test
    )
    
    # Train and evaluate
    results_df = train_and_evaluate_models(
        X_train_s, X_test_s, y_train_s, y_test_s,
        y_train_original, y_test_original, scaler_y
    )
    
    # Cross-validation
    cv_results = cross_validate_models(X_train_s, y_train_s, y_train_original, scaler_y)
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('Results/tables/corrected_model_comparison.csv', index=False)
    print("\n✓ Results saved to Results/tables/corrected_model_comparison.csv")
    
    cv_results.to_csv('Results/tables/corrected_cv_results.csv', index=False)
    print("✓ CV results saved to Results/tables/corrected_cv_results.csv")
    
    # Find best model
    best_model = results_df.loc[results_df['MSE (Trillion BTU)'].idxmin()]
    
    print("\n" + "=" * 80)
    print("BEST MODEL SUMMARY")
    print("=" * 80)
    print(f"\n  Model:               {best_model['Model']}")
    print(f"  MSE (Trillion BTU):  {best_model['MSE (Trillion BTU)']:.4f}")
    print(f"  R²:                  {best_model['R²']:.4f}")
    print(f"  Base Paper MSE:      {BASE_PAPER_MSE_RIDGE:.2f}")
    print(f"  Improvement:         {best_model['Improvement (%)']:.2f}%")
    
    if best_model['Improvement (%)'] > 0:
        print(f"\n  ✓ VALID improvement of {best_model['Improvement (%)']:.2f}% over base paper")
    else:
        print(f"\n  ✗ No improvement over base paper (this is REALISTIC given the data)")
    
    print("\n" + "=" * 80)
    print("IMPORTANT NOTE")
    print("=" * 80)
    print("""
The previous results showed 99.99% improvement due to DATA LEAKAGE.
These corrected results are scientifically valid because:
  1. No features use current-timestep cross-sector data
  2. MSE is calculated in original Trillion BTU units
  3. Proper comparison against base paper's MSE = 2.61
  
A modest improvement (or even no improvement) is REALISTIC for this dataset.
""")
    
    return results_df, cv_results


if __name__ == "__main__":
    results_df, cv_results = main()
