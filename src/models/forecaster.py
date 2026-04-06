"""
src/models/forecaster.py

Recursive Forecaster for Multi-Step Predictions.
================================================
Iteratively predicts one step ahead, appends prediction to history,
re-calculates features (rolling, lags, etc.), and predicts the next step.
"""

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict

from src.utils.logger import get_logger

log = get_logger(__name__)

class RecursiveForecaster:
    """
    Simulates a production forecasting loop.
    
    Attributes
    ----------
    model : Trained sklearn-like model (must have .predict())
    preprocessor : Fitted TimeSeriesPreprocessor
    feature_fn : Function to call for feature engineering (create_features)
    sector : Target sector name
    feat_cfg : Configuration dictionary for feature engineering
    history_df : DataFrame containing historical data (Month + sectors)
    exog_df : DataFrame containing future exogenous variables (optional)
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        feature_fn: Callable,
        sector: str,
        feat_cfg: Dict[str, Any],
        history_df: pd.DataFrame,
        feature_cols: list[str],
        exog_df: pd.DataFrame | None = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_fn = feature_fn
        self.sector = sector
        self.feat_cfg = feat_cfg.copy()
        self.history_df = history_df.copy()
        self.feature_cols = feature_cols
        self.exog_df = exog_df.copy() if exog_df is not None else None
        
        # Ensure Month is datetime
        self.history_df["Month"] = pd.to_datetime(self.history_df["Month"])
        if self.exog_df is not None:
             self.exog_df["Month"] = pd.to_datetime(self.exog_df["Month"])

    def predict_horizon(self, steps: int = 12) -> pd.DataFrame:
        """
        Predict 'steps' months into the future.
        """
        predictions = []
        future_dates = []
        
        # Determine start date (next month after last history)
        last_date = self.history_df["Month"].max()
        
        log.info(f"Starting forecast from {last_date} for {steps} steps...")

        current_history = self.history_df.copy()
        
        for i in range(steps):
            next_date = last_date + pd.DateOffset(months=i+1)
            future_dates.append(next_date)
            
            # 1. Create PLACEHOLDER row for next_date
            # We need this row to exist so create_features can calculate lags/rolling
            # derived from previous rows.
            new_row = {"Month": next_date}
            # Fill other cols with NaN or 0 (won't be used for features value, 
            # only for target which we are predicting)
            for col in current_history.columns:
                if col != "Month":
                    new_row[col] = np.nan 
            
            # Append placeholder
            current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)
            
            # 2. Re-create features for the ENTIRE history (including new row)
            # This ensures all rolling windows and lags are correct for the new row
            feats_df = self.feature_fn(
                current_history, 
                self.sector, 
                **self.feat_cfg,
                exog_df=self.exog_df,  # Pass full exog df (should cover future dates)
                drop_na=False          # Keep NaNs for inference!
            )
            
            # IMPROVEMENT: Fill NaNs (e.g. at start of series or missing exog)
            # Forward fill is logical for time series (propagate last known state)
            feats_df = feats_df.ffill().bfill()
            
            # 3. Extract Feature Vector for the NEW row (last row)
            # We expect the last row of feats_df to correspond to next_date
            # Check if feature engineering dropped rows (due to lags)
            # We need the row matching next_date
            
            target_feat_row = feats_df[feats_df["Month"] == next_date]
            
            if target_feat_row.empty:
                raise ValueError(f"Feature engineering dropped the target date {next_date}. Increase history length?")
            
            # Filter to match model's features
            X_row_df = target_feat_row[self.feature_cols]
            
            # Check for NaNs
            if X_row_df.isnull().values.any():
                null_cols = X_row_df.columns[X_row_df.isnull().any()].tolist()
                log.warning(f"  Warning: NaNs found in features for {next_date.date()}: {null_cols}. Filling with 0.0.")
                X_row_df = X_row_df.fillna(0.0)
            
            X_cols = X_row_df.columns.tolist()
            X_vals = X_row_df.values
            
            # 4. Scale
            X_scaled = self.preprocessor.scaler_X.transform(X_vals)
            
            # 5. Predict
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)[0]
            
            # Handle negative predictions (energy can't be negative)
            y_pred = max(0.0, y_pred)
            
            predictions.append(y_pred)
            
            # 6. Update History with Prediction
            # We fill the 'target_sector' column for that new row with our prediction
            # so next iteration's lags will see it.
            current_history.loc[current_history["Month"] == next_date, self.sector] = y_pred
            
            # Also need to fill 'Total Energy' if used?
            # Approximation: If we only predict one sector, we can't perfectly update Total.
            # But we can update the Total column by adding this sector's predicted val 
            # + previous values of other sectors (naive) or just leave others as NaN.
            # src.data.feature_engineering handles total_energy = sum(axis=1). 
            # If other sectors are NaN, sum is partial. 
            # For this MVP, we assume interaction is weak or we accept the limitation.
            
            # Log progress
            log.debug(f"  Step {i+1}/{steps}: {next_date.date()} -> {y_pred:.2f}")

        results = pd.DataFrame({
            "Month": future_dates,
            "Forecast": predictions
        })
        return results
