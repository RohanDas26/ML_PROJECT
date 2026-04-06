
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, OrthogonalMatchingPursuit, ElasticNet
from sklearn.preprocessing import StandardScaler
import os
import sys

# Define proper directories
BASE_DIR = r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\EnergyForecasting_v2_Production"
DATA_PATH = os.path.join(BASE_DIR, "Data", "Artifacts", "clean_data.csv")
FIG_DIR = os.path.join(BASE_DIR, "Results", "figures", "Phase7_Final")

os.makedirs(FIG_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Could not find data at", DATA_PATH)
    sys.exit(1)

# Setup style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1c1c1c", "figure.facecolor": "#1c1c1c", "grid.color": "#333333", "text.color":"white", "axes.labelcolor":"white", "xtick.color":"white", "ytick.color":"white"})

df = pd.read_csv(DATA_PATH)
if 'Month' in df.columns:
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
elif 'YYYYMM' in df.columns:
    df['YYYYMM'] = pd.to_datetime(df['YYYYMM'], format='%Y%m')
    df.set_index('YYYYMM', inplace=True)
else:
    df.index = pd.date_range(start='1973-01-01', periods=len(df), freq='M')

df.sort_index(inplace=True)

SECTORS = {
    'Industrial': ('Industrial Energy Consumption', Lasso(alpha=0.01)),
    'Commercial': ('Commercial Energy Consumption', Ridge(alpha=1.0)),
    'Residential': ('Residential Energy Consumption', OrthogonalMatchingPursuit(n_nonzero_coefs=15)),
    'Transportation': ('Transportation Energy Consumption', ElasticNet(alpha=0.01, l1_ratio=0.5))
}

def create_features(y_series, name):
    # Leak-free feature creation
    df_feat = pd.DataFrame(y_series.values, index=y_series.index, columns=[name])
    for i in range(1, 25): df_feat[f'lag_{i}'] = df_feat[name].shift(i)
    for w in [3, 6, 12]:
        df_feat[f'rolling_mean_{w}_lag_1'] = df_feat[name].shift(1).rolling(w).mean()
        df_feat[f'rolling_std_{w}_lag_1']  = df_feat[name].shift(1).rolling(w).std()
    months = df_feat.index.month
    df_feat['sin_m12'] = np.sin(2 * np.pi * months / 12)
    df_feat['cos_m12'] = np.cos(2 * np.pi * months / 12)
    df_feat['sin_m6']  = np.sin(2 * np.pi * months / 6)
    df_feat['cos_m6']  = np.cos(2 * np.pi * months / 6)
    df_feat.dropna(inplace=True)
    return df_feat

for sector, (col_name, base_model) in SECTORS.items():
    actual_col = [c for c in df.columns if sector.split(" ")[0] in c]
    if not actual_col: continue
    col_to_use = actual_col[0]
    
    print(f"Processing {sector} with target {col_to_use}...")
    df_features = create_features(df[col_to_use], sector)
    
    X = df_features.drop(columns=[sector])
    y = df_features[sector]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = base_model
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    
    # 1. Forecast Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Data', color='#00d4ff', linewidth=2)
    plt.plot(y_test.index, y_pred, label=f'Phase 7 {model.__class__.__name__} Forecast', color='#ff007f', linestyle='--', linewidth=2)
    plt.title(f"{sector} Sector - Phase 7 Ceiling Forecast vs Actual", fontsize=16, pad=15, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{sector}_Phase7_Forecast.png"), dpi=300)
    plt.close()
    
    # 2. Importance Plot
    coefs = getattr(model, 'coef_', getattr(model, 'feature_importances_', np.zeros(len(X.columns))))
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': coefs})
    imp_df['Abs_Imp'] = imp_df['Importance'].abs()
    top_imp = imp_df.sort_values(by='Abs_Imp', ascending=False).head(15)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_imp, x='Importance', y='Feature', palette='viridis')
    plt.title(f"{sector} Sector - Phase 7 {model.__class__.__name__} Feature Coefficients", fontsize=16, pad=15, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{sector}_Phase7_Importance.png"), dpi=300)
    plt.close()
    
print("ALL DONE")
