
PATH = r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\EnergyForecasting_v2_Production\Documentation\Final_Project_Report.md"

s = """

## 4. PROBLEM STATEMENT AND OBJECTIVES

### 4.1 Problem Definition
The primary problem addressed by this research is the **accurate, reliable, and mathematically sound forecasting of sector-level U.S. energy consumption** using historical macroeconomic and consumption data. 

The secondary, yet equally critical, problem is the pervasive issue of **methodological flaws in existing energy forecasting literature**. As demonstrated by our replication of Malakouti et al. (2025), significant portions of published research rely on data leakage and unit-scale obfuscation to claim near-perfect prediction accuracy (R² = 0.999), which is impossible to achieve in a live forecasting environment. The problem entails building a rigorously evaluated, leak-free pipeline that establishes the *true* predictable ceiling of energy demand.

### 4.2 The "Leakage Illusion" Audit Outcomes
Before building our forecasting pipeline, we audited the baseline paper to understand exactly why their results were irreproducible under honest conditions. Our forensic replication (`notebooks/04_audit_and_replication.ipynb`) identified eight specific mathematical and methodological flaws:

1. **Massive Data Leakage**: The total energy consumption of all sectors at month `t` was used as a feature to predict a specific sector at month `t`. This allows the model to "peek" at the answer.
2. **Invalid Z-Score Error Reporting**: RMSE was calculated on standardized (Z-score) data (range -3 to +3) but reported as if it were in original Trillion BTU units (range 1000 to 3000).
3. **Improper Time-Series Cross-Validation**: Basic K-Fold cross-validation was used, randomizing the temporal order and allowing models to predict the past using the future.
4. **Information Leakage in Scaling**: `StandardScaler` was fit on the *entire* dataset before splitting, leaking future distribution statistics (mean/variance) into the training set.
5. **Lack of Statistical Significance Testing**: Differences between model RMSEs (e.g., 0.0031 vs 0.0032) were claimed as "superiority" without applying the Diebold-Mariano test to verify if the difference was statistically distinct from random noise.
6. **False Equivalence in Comparisons**: The base paper compared their Z-scaled RMSE values against historical literature that reported RMSE in actual Trillion BTUs, fabricating a "99.99% improvement."
7. **Overfitting Small Data with Deep Ensembles**: High-variance models were deployed on just 600 samples without adequate structural regularization.
8. **Ignoring Domain Constraints**: Total sector forecasts did not equal total aggregate energy.

*Figure 4.1 (see `Results/visualizations/audit_z_score_vs_actual_rmse_comparison.png`): The "Leakage Illusion." The base paper reported an RMSE of ~0.003 units. Our honest replication shows the true RMSE is ~0.4 units in the same Z-scale — meaning their reported error was artificially compressed by a factor of 142×.*

### 4.3 Research Objectives
To solve both the forecasting challenge and the methodological crisis, our objectives are:
1. Process the 1973–2021 EIA dataset ensuring absolute strict temporal separation (no future data leaks into past predictions).
2. Engineer up to 70 robust time-series features (autoregressive lags, rolling statistics, Fourier harmonics) that rely *only* on data available strictly before the forecast target date.
3. Compare 11 standard machine learning algorithms (Linear, Tree-based, SVM, KNN) using a 10-fold strict chronological `TimeSeriesSplit`.
4. Apply Optuna Bayesian optimization to find the absolute performance ceiling for each sector.
5. Statistically validate all findings using the Diebold-Mariano test.
6. Deploy the final models into a production-ready interactive dashboard to demonstrate live predictive capability.

---

## 5. METHODOLOGY

The project methodology is structured as a 10-phase progression, ensuring that every transformation, feature addition, and model optimization is incrementally validated against strict anti-leakage principles.

### 5.1 System Architecture and Data Pipeline
The system is built as a modular Python application (`EnergyForecasting_v2_Production/src/`). The overarching pipeline is as follows:
1. **Data Ingestion (`data_loader.py`)**: Loads the EIA JSON/CSV files, parsing date indices and standardizing column names.
2. **Feature Engineering (`feature_engineering.py`)**: Generates 70 distinct time-series features (detailed below).
3. **Leakage Verification (`tests/test_leakage.py`)**: A suite of 36 unit tests programmatically verifies that no feature contains data from time `t` when predicting time `t`.
4. **Model Training (`model_factory.py`)**: Instantiates models spanning regularized linear regression (Lasso, Ridge), ensemble trees (XGBoost, Random Forest), and deep learning (PyTorch LSTM).
5. **Cross-Validation (`optuna_trainer.py`)**: Executes 80 trials per model using `TimeSeriesSplit(n_splits=10)` to optimize hyperparameters.
6. **Statistical Validation (`statistical_tests.py`)**: Performs Diebold-Mariano tests on out-of-fold predictions.

### 5.2 The 10-Phase Experimental Progression
A core tenet of our methodology is isolated, incremental complexity:

- **Phase 1 (Honest Baseline)**: 12-month lags only, no feature engineering. Establishes the true, legal baseline.
- **Phase 2 (Ensemble/Stacking)**: Basic stacking (Lasso + XGBoost) to test model combination efficacy.
- **Phase 3 (Fourier & Statistics)**: Introduction of 6-month and 12-month Fourier sine/cosine terms and 3/6-month rolling means.
- **Phase 4 (Sector-Specific Optimization)**: Tuning algorithms independently for each sector's unique dynamics.
- **Phase 5 (Full Scale Optuna)**: 80-trial Bayesian search across all models. SVR emerges as a top contender here.
- **Phase 6 (Deep Lags - The Breakthrough)**: Extension of autoregressive lags from `t-12` back to `t-24`, dropping all traditional tree models (XGBoost/RF) in favor of strictly penalized linear models (Lasso/Ridge/ElasticNet/OMP). This shift yielded the largest performance jump in the project.
- **Phase 7 (The Absolute Ceiling)**: Hyper-optimized regularized linear models combined with `t-24` lags, Fourier terms, and structural-break adjustments. This represents the maximum achievable accuracy on the dataset.
- **Phase 8 (LSTM / Deep Learning Evaluation)**: Implementation of PyTorch-based Long Short-Term Memory networks to empirically test deep learning against the Phase 7 benchmark.
- **Phase 9 (Exogenous Variables Simulation)**: Evaluation of how external economic indicators (dummy variables) impact predictability.
- **Phase 10 (Z-Scale Fair Comparison)**: Standardizing our Phase 7 models to Z-scores to directly, mathematically compare our honest models against the base paper's leaky models.

### 5.3 Feature Engineering Strategy (Strictly Leak-Free)
To ensure absolute mathematical integrity, we constructed a 70-feature matrix where every predictor is explicitly shifted by at least 1 temporal step (`t-1` or older).

1. **Autoregressive Lags (24 Features)**: `lag_1` through `lag_24`. The inclusion of `lag_12` and `lag_24` captures strict annual seasonality, while `lag_1` captures immediate momentum.
2. **Rolling Statistics (12 Features)**: Moving averages and exponentially weighted moving averages (EMA) over 3, 6, and 12-month windows (e.g., `rolling_mean_3_lag_1`) to smooth volatility.
3. **Volatility Indicators (4 Features)**: Rolling standard deviations (`rolling_std_6_lag_1`) to explicitly model heteroskedasticity (variance in consumption patterns).
4. **Fourier Seasonality (4 Features)**: Explicit mathematical encoding of the bimodal (Summer AC / Winter Heating) pattern using sine and cosine waves:
   - `sin_m12` = $\sin(2\pi \cdot \text{month} / 12)$
   - `cos_m12` = $\cos(2\pi \cdot \text{month} / 12)$
   - `sin_m6` = $\sin(2\pi \cdot \text{month} / 6)$
   - `cos_m6` = $\cos(2\pi \cdot \text{month} / 6)$
5. **Year-over-Year Delta (1 Feature)**: `yoy_change_lag_1` calculates the percentage difference between `t-1` and `t-13`.
6. **Interaction Terms**: Multiplication of the primary lag (`lag_1`) by immediate volatility (`rolling_std_3`), allowing the model to dampen its response to high-variance periods.

### 5.4 Cross-Validation and Hyperparameter Optimization
To prevent "Look-Ahead Bias", standard K-Fold cross-validation was strictly prohibited. All models were evaluated using **Chronological TimeSeriesSplit**.
- `n_splits = 10`
- Fold 1 trains on 1973–1978, tests on 1979.
- Fold 2 trains on 1973–1979, tests on 1980.
- ...and so on.

**Optuna Optimization**:
For hyperparameter tuning, we utilized Optuna's Tree-structured Parzen Estimator (TPE). Each model underwent 80 trials. The objective function strictly minimized the out-of-fold Mean Squared Error (MSE) computed across the 10 chronological splits.

### 5.5 Deep Learning (LSTM) Implementation Details
To rigorously answer whether deep learning adds value to this dataset size, we built a PyTorch LSTM:
- **Architecture**: 1 to 3 LSTM layers followed by a fully connected output layer.
- **Sequence Length**: 12 to 24 months (tuned).
- **Hidden Dimensions**: 32 to 128 units.
- **Regularization**: Dropout (0.1 to 0.4) and L2 weight decay.
- **Optimizer**: AdamW with learning rate scheduling.
- **Loss Function**: Smooth L1 Loss (Huber) to prevent large gradient updates from outlier consumption spikes.

The LSTM was subjected to the exact same 10-fold chronological validation constraints as the classical machine learning models.

"""

with open(PATH, "a", encoding="utf-8") as f:
    f.write(s)
print("Sections 4-5 appended successfully.")
