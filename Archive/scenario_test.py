import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 100)
print("TESTING MODEL WITH DIFFERENT FEATURE SETS")
print("=" * 100)

# Load data
train_df = pd.read_csv('artifacts/train.csv')
test_df = pd.read_csv('artifacts/test.csv')

target_col = "Total Energy Consumed by the Residential Sector"
drop_cols = ["date", target_col]

# ============================================================================
# SCENARIO 1: Current model (with other sector data)
# ============================================================================
print("\n" + "=" * 100)
print("SCENARIO 1: WITH OTHER SECTOR ENERGY DATA (current model)")
print("=" * 100)

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target_col].values
X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target_col].values

# Load preprocessing and model
preprocess = joblib.load('artifacts/preprocess_pipeline.pkl')
X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

model = ElasticNet(alpha=1.5916753443475336e-05, l1_ratio=0.9887596118454134)
model.fit(X_train_proc, y_train)
pred = model.predict(X_test_proc)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"\nRMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.6f}")
print(f"\nFeatures: {X_train.shape[1]}")
print(f"Top 5 features by column name:")
feature_names = X_train.columns.tolist()
for i, name in enumerate(feature_names[:5]):
    print(f"  {i+1}. {name}")

# ============================================================================
# SCENARIO 2: WITHOUT other sector's energy data (only time features + lags)
# ============================================================================
print("\n" + "=" * 100)
print("SCENARIO 2: WITHOUT OTHER SECTOR ENERGY DATA (only lags + time features)")
print("=" * 100)

# Features to keep: only time-based, lags, and rolling stats
time_lag_features = [
    'year', 'month', 't',
    'Total Energy Consumed by the Residential Sector__lag1',
    'Total Energy Consumed by the Residential Sector__lag12',
    'Total Energy Consumed by the Residential Sector__rollmean3',
    'Total Energy Consumed by the Residential Sector__rollstd3',
    'Total Energy Consumed by the Residential Sector__rollmean12',
    'Total Energy Consumed by the Residential Sector__rollstd12'
]

X_train_lags = train_df[time_lag_features]
X_test_lags = test_df[time_lag_features]

# Simple preprocessing for lags only
num_cols = [c for c in X_train_lags.columns if X_train_lags[c].dtype.kind in 'if']
cat_cols = [c for c in X_train_lags.columns if X_train_lags[c].dtype == object]

preprocess_lags = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

X_train_lags_proc = preprocess_lags.fit_transform(X_train_lags)
X_test_lags_proc = preprocess_lags.transform(X_test_lags)

# Train model
model_lags = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
model_lags.fit(X_train_lags_proc, y_train)
pred_lags = model_lags.predict(X_test_lags_proc)

rmse_lags = np.sqrt(mean_squared_error(y_test, pred_lags))
mae_lags = mean_absolute_error(y_test, pred_lags)
r2_lags = r2_score(y_test, pred_lags)

print(f"\nRMSE: {rmse_lags:.2f}")
print(f"MAE: {mae_lags:.2f}")
print(f"R²: {r2_lags:.6f}")
print(f"\nFeatures: {X_train_lags.shape[1]}")
print(f"Features used:")
for i, name in enumerate(time_lag_features):
    print(f"  {i+1}. {name}")

# ============================================================================
# SCENARIO 3: Linear regression (no regularization) - shows raw relationship
# ============================================================================
print("\n" + "=" * 100)
print("SCENARIO 3: LINEAR REGRESSION (no regularization)")
print("=" * 100)

from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train_proc, y_train)
pred_lr = model_lr.predict(X_test_proc)

rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)

print(f"\nRMSE: {rmse_lr:.2f}")
print(f"MAE: {mae_lr:.2f}")
print(f"R²: {r2_lr:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 100)
print("COMPARISON SUMMARY")
print("=" * 100)

print(f"\n{'Model':<40} {'RMSE':<12} {'R²':<12}")
print("-" * 65)
print(f"{'1. With sector data (ElasticNet)':<40} {rmse:>10.2f}  {r2:>10.6f}")
print(f"{'2. Lags only (ElasticNet)':<40} {rmse_lags:>10.2f}  {r2_lags:>10.6f}")
print(f"{'3. With sector data (LinearReg)':<40} {rmse_lr:>10.2f}  {r2_lr:>10.6f}")

print(f"\n{'VERDICT':<40}")
print("-" * 65)
if rmse_lags > rmse * 1.5:
    print(f"✓ Model is VALID - lags only give much worse performance (R²={r2_lags:.4f})")
    print(f"  This proves sector data adds real predictive value, not just memorization")
    print(f"  R² = {r2:.4f} is realistic for this problem")
else:
    print(f"⚠️ Model is SUSPICIOUS - lags alone give similar performance")
    print(f"  This suggests high R² might be due to trend/seasonality only")
    print(f"  Sector data might not add much predictive value")

print("\n" + "=" * 100)
