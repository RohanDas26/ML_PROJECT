
import re

PATH = r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\EnergyForecasting_v2_Production\Documentation\Final_Project_Report.md"

with open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Inject code snippet into 5.3
feature_eng_code = """

**Code Snippet: Chronologically Safe Feature Engineering**
The following snippet from our `feature_engineering.py` module mathematically guarantees that no future data leaks into the training row at index `t`.
```python
def engineer_features(df: pd.DataFrame, target_col: str, max_lag: int = 24) -> pd.DataFrame:
    df_feat = df.copy()
    
    # 1. Autoregressive Lags (strictly shifting past records into current row)
    for i in range(1, max_lag + 1):
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
        
    # 2. Rolling Statistics (computed exclusively on past windows)
    # Using shift(1) ensures the rolling window never sees the current month's truth
    for window in [3, 6, 12]:
        df_feat[f'rolling_mean_{window}_lag_1'] = df_feat[target_col].shift(1).rolling(window=window).mean()
        df_feat[f'rolling_std_{window}_lag_1']  = df_feat[target_col].shift(1).rolling(window=window).std()
        
    # 3. Fourier Seasonality (Extracting the cyclical 6-mo and 12-mo patterns)
    month = df_feat.index.month
    df_feat['sin_m12'] = np.sin(2 * np.pi * month / 12)
    df_feat['cos_m12'] = np.cos(2 * np.pi * month / 12)
    df_feat['sin_m6']  = np.sin(2 * np.pi * month / 6)
    df_feat['cos_m6']  = np.cos(2 * np.pi * month / 6)
    
    # 4. Drop any rows containing NaNs introduced by the shifting process
    df_feat.dropna(inplace=True)
    return df_feat
```
"""
text = text.replace("6. **Interaction Terms**: Integration of primary directional momentum (`lag_1`) with immediate local variance (`rolling_std_3`) for stability.", 
                    "6. **Interaction Terms**: Integration of primary directional momentum (`lag_1`) with immediate local variance (`rolling_std_3`) for stability.\n\n" + feature_eng_code)

# 2. Inject code snippet into 5.4
cv_opt_code = """

**Code Snippet: Strict Chronological Validation via Optuna**
The following implementation from our `optuna_trainer.py` module proves our adherence to strict time-series cross-validation. Random K-Fold shuffling is strictly explicitly avoided.
```python
def objective(trial, X, y):
    # 1. Trial Hyperparameters (e.g., for Lasso Regularization)
    alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
    model = Lasso(alpha=alpha, max_iter=10000)
    
    # 2. Strict Chronological Split (n_splits = 10)
    tscv = TimeSeriesSplit(n_splits=10)
    fold_mses = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 3. Scaling strictly fit on the training fold to prevent distribution leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        
        # 4. Training and Evaluation
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        fold_mses.append(mean_squared_error(y_test, preds))
        
    # 5. Optuna minimizes the average out-of-fold generalization error
    return np.mean(fold_mses)
```
"""
text = text.replace("For hyperparameter determination, we utilized Optuna's Tree-structured Parzen Estimator (TPE). Each architecture underwent 80 independent computational trials. The objective function strictly minimized the out-of-fold Mean Squared Error (MSE) computed across the chronological splits.",
                    "For hyperparameter determination, we utilized Optuna's Tree-structured Parzen Estimator (TPE). Each architecture underwent 80 independent computational trials. The objective function strictly minimized the out-of-fold Mean Squared Error (MSE) computed across the chronological splits.\n\n" + cv_opt_code)


# 3. Inject visual references into 6.3
vis_injections = """
*(Refer to `Results/figures/` for visual actual-vs-predicted plots across these final architectures.)*

#### 6.3.1 Visual Proof: Industrial Sector Performance
The optimal model natively aligns with historical variations without oscillating wildly, indicating profound noise rejection.

*Figure 6.1: Industrial Sector Optuna Optimized Model (Lasso) vs Actual Truth*  
*(Location: `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Industrial_optuna_actual_vs_predicted.png`)*
![Industrial Actual vs Predicted](../../Results/figures/Industrial_optuna_actual_vs_predicted.png)

*Figure 6.2: Industrial Sector Feature Importance Analysis*  
*(Location: `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Industrial_optuna_feature_importance.png`)*
![Industrial Feature Importance](../../Results/figures/Industrial_optuna_feature_importance.png)

#### 6.3.2 Visual Proof: Commercial Sector Performance
Notice the tight alignment to actual data even during macroeconomic shocks, validating the robustness of the recursive structure.

*Figure 6.3: Commercial Sector Optuna Optimized Model vs Actual Truth*  
*(Location: `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Commercial_optuna_actual_vs_predicted.png`)*
![Commercial Actual vs Predicted](../../Results/figures/Commercial_optuna_actual_vs_predicted.png)

#### 6.3.3 Visual Proof: Residential & Transportation
*(Additional Visualizations for Residential and Transportation can be found encoded at:)*
- **Residential Model Plot**: `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Residential_optuna_actual_vs_predicted.png`
- **Residential Features**: `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Residential_optuna_feature_importance.png`
- **Transportation Plot**: `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Transportation_optuna_actual_vs_predicted.png`

"""
text = text.replace("*(Refer to `Results/figures/` for visual actual-vs-predicted plots across these final architectures.)*",
                    vis_injections)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("Injected successfully.")
