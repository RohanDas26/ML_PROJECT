"""
Corrected Feature Engineering for Energy Consumption Forecasting
================================================================

This script implements PROPER feature engineering that avoids data leakage.
All features use ONLY information available at prediction time (lagged values).

Key Fixes:
- No cross-sector features using current timestep (removes data leakage)
- Proper lagged features (Lag-1, Lag-12) for autocorrelation
- Temporal features (month, year, quarter) for seasonality
- Rolling statistics using ONLY past data

Author: Corrected Version
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CORRECTED FEATURE ENGINEERING - NO DATA LEAKAGE")
print("=" * 80)


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


def create_valid_features(df, target_sector):
    """
    Create features WITHOUT data leakage.
    
    CRITICAL: All features must use ONLY information available at prediction time.
    This means we can only use lagged values, not current values from other sectors.
    
    Args:
        df: DataFrame with all sectors
        target_sector: The sector we want to predict ('Commercial', etc.)
    
    Returns:
        DataFrame with valid features
    """
    print(f"\nCreating features for {target_sector} sector (NO DATA LEAKAGE)")
    print("-" * 60)
    
    features_df = pd.DataFrame()
    features_df['Month'] = df['Month']
    
    # =====================================
    # TEMPORAL FEATURES (Always Valid)
    # =====================================
    features_df['year'] = df['Month'].dt.year
    features_df['month'] = df['Month'].dt.month
    features_df['quarter'] = df['Month'].dt.quarter
    
    # Cyclical encoding for month (captures seasonality better)
    features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
    features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
    
    print("✓ Temporal features: year, month, quarter, month_sin, month_cos")
    
    # =====================================
    # AUTOREGRESSIVE FEATURES (Lagged Target)
    # =====================================
    # These are the most important predictors for time series
    target = df[target_sector]
    
    # Lag-1: Previous month (captures momentum)
    features_df['target_lag_1'] = target.shift(1)
    
    # Lag-2: Two months ago
    features_df['target_lag_2'] = target.shift(2)
    
    # Lag-3: Three months ago  
    features_df['target_lag_3'] = target.shift(3)
    
    # Lag-12: Same month last year (captures annual seasonality)
    features_df['target_lag_12'] = target.shift(12)
    
    # Lag-13: Previous month, previous year
    features_df['target_lag_13'] = target.shift(13)
    
    print("✓ Autoregressive features: target_lag_1, target_lag_2, target_lag_3, target_lag_12, target_lag_13")
    
    # =====================================
    # ROLLING STATISTICS (Using PAST Data Only)
    # =====================================
    # Rolling mean of past 3 months
    features_df['target_rolling_mean_3'] = target.shift(1).rolling(window=3).mean()
    
    # Rolling mean of past 6 months
    features_df['target_rolling_mean_6'] = target.shift(1).rolling(window=6).mean()
    
    # Rolling mean of past 12 months
    features_df['target_rolling_mean_12'] = target.shift(1).rolling(window=12).mean()
    
    # Rolling std of past 3 months (captures volatility)
    features_df['target_rolling_std_3'] = target.shift(1).rolling(window=3).std()
    
    print("✓ Rolling statistics: rolling_mean_3, rolling_mean_6, rolling_mean_12, rolling_std_3")
    
    # =====================================
    # LAGGED TOTAL ENERGY (Valid Cross-Sector Feature)
    # =====================================
    # IMPORTANT: We can use total_energy ONLY as a lagged feature (from previous periods)
    total_energy = df['Residential'] + df['Commercial'] + df['Industrial'] + df['Transportation']
    
    # Total energy from previous month
    features_df['total_energy_lag_1'] = total_energy.shift(1)
    
    # Total energy from same month last year
    features_df['total_energy_lag_12'] = total_energy.shift(12)
    
    print("✓ Lagged total energy: total_energy_lag_1, total_energy_lag_12")
    
    # =====================================
    # LAGGED OTHER SECTORS (Valid if from past)
    # =====================================
    other_sectors = ['Residential', 'Commercial', 'Industrial', 'Transportation']
    other_sectors.remove(target_sector)
    
    for sector in other_sectors:
        # Only use LAG-1 values from other sectors
        features_df[f'{sector.lower()}_lag_1'] = df[sector].shift(1)
    
    print(f"✓ Lagged other sectors (lag-1): {', '.join([s.lower() + '_lag_1' for s in other_sectors])}")
    
    # =====================================
    # YEAR-OVER-YEAR CHANGE (Valid)
    # =====================================
    features_df['yoy_change'] = (target.shift(1) - target.shift(13)) / target.shift(13)
    
    print("✓ Year-over-year change: yoy_change")
    
    # =====================================
    # TARGET VARIABLE
    # =====================================
    features_df['target'] = target
    
    # Drop rows with NaN values (due to lagging)
    n_before = len(features_df)
    features_df = features_df.dropna().reset_index(drop=True)
    n_after = len(features_df)
    
    print(f"\n✓ Created {len(features_df.columns) - 2} valid features")  # -2 for Month and target
    print(f"✓ Dropped {n_before - n_after} rows due to lagging (using {n_after} samples)")
    
    return features_df


def prepare_train_test_split(features_df, test_size=0.2):
    """
    Prepare train/test split respecting time ordering.
    
    For time series, we should NOT use random splitting.
    We use the most recent data for testing.
    """
    # Features and target
    feature_cols = [col for col in features_df.columns if col not in ['Month', 'target']]
    X = features_df[feature_cols].values
    y = features_df['target'].values
    dates = features_df['Month'].values
    
    # Time-ordered split (last 20% for testing)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"\n✓ Train/Test split: {len(X_train)} training, {len(X_test)} testing samples")
    print(f"✓ Training period: {dates_train[0]} to {dates_train[-1]}")
    print(f"✓ Testing period: {dates_test[0]} to {dates_test[-1]}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def standardize_data(X_train, X_test, y_train, y_test):
    """Standardize features and target, returning scalers for inverse transform."""
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def main():
    """Main execution function."""
    # Load data
    print("\nLoading data...")
    df = load_and_preprocess_data('Dataset/NEW DATA SET-1.xlsx')
    print(f"✓ Loaded {len(df)} samples from {df['Month'].min()} to {df['Month'].max()}")
    
    # Process Commercial sector as example
    target_sector = 'Commercial'
    features_df = create_valid_features(df, target_sector)
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(features_df)
    
    # Standardize
    X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = standardize_data(
        X_train, X_test, y_train, y_test
    )
    
    # Save processed data for model training
    processed_data = {
        'X_train': X_train_s,
        'X_test': X_test_s,
        'y_train': y_train_s,
        'y_test': y_test_s,
        'y_train_original': y_train,
        'y_test_original': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols
    }
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print("\nFeatures created (all valid, no data leakage):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nTotal: {len(feature_cols)} features")
    print("\n✓ Ready for model training with corrected features")
    
    return processed_data, features_df


if __name__ == "__main__":
    processed_data, features_df = main()
