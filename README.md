# Energy Consumption Forecasting Using Machine Learning
## A Rigorous, Leak-Free Multi-Sector Analysis of US Energy Demand (1973-2021)

**B.Tech Final Year Project | Department of Computer Science / Data Science | March 2026**

---

## Project Overview

This project develops a rigorous, production-grade machine learning pipeline for forecasting
sector-level US energy demand across four sectors: Residential, Commercial, Industrial, and
Transportation. The dataset spans 633 monthly observations (January 1973 - December 2021)
from the US Energy Information Administration (EIA).

A core contribution of this work is a forensic audit of an existing peer-reviewed baseline,
which identified eight distinct methodological flaws including pervasive data leakage.
Our clean, leak-free pipeline reduces prediction error by 98.3% to 99.2% over the baseline.

---

## Project Structure

```
EnergyForecasting_ML_PROJECT/
├── app.py              # Streamlit dashboard (main entry point)
├── main.py             # Pipeline entry point
├── system_run.py       # System runner
├── requirements.txt    # Python dependencies
├── config/             # Configuration files (config.yaml)
├── src/                # Core source code
│   ├── data/           # Data loading, preprocessing, feature engineering
│   ├── models/         # Model definitions (Lasso, Ridge, XGBoost, LSTM, etc.)
│   ├── evaluation/     # Metrics, diagnostics, Diebold-Mariano tests
│   ├── visualization/  # Plot generation
│   ├── utils/          # IO and logging utilities
│   └── frontend/       # Web dashboard (HTML/CSS/JS)
├── tests/              # Anti-leakage unit tests (36 tests)
├── Data/
│   ├── Dataset/        # Raw EIA dataset (xlsx)
│   └── Processed/      # Cleaned CSVs, trained model files (pkl), metrics (json)
├── Results/
│   ├── figures/        # Per-sector model output plots
│   │   └── Phase7_Final/ # Phase 7 optimized model forecast plots
│   ├── tables/         # Numerical results (CSV, JSON)
│   └── full_report/    # SHAP, ablation, STL decomposition, robustness plots
├── Visualizations/
│   ├── EDA/            # 20 exploratory data analysis plots
│   └── Model_Performance/ # Cross-model comparison figures and tables
└── scripts/            # Full phase-by-phase analysis and evaluation scripts
```

---

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run the full pipeline
```
python main.py
```

### 3. Launch the interactive dashboard
```
streamlit run app.py
```

---

## Key Results (Phase 7 - Optimized Models)

| Sector         | Best Model  | RMSE (Trillion BTU) | R2   |
|:---------------|:------------|:-------------------:|:----:|
| Industrial     | Lasso       | 0.0296              | 1.00 |
| Transportation | ElasticNet  | 0.0385              | 1.00 |
| Commercial     | Ridge       | 0.0626              | 1.00 |
| Residential    | OMP         | 0.1094              | 1.00 |

---

## Methodology Highlights

- **70-feature leak-free engineering**: autoregressive lags (t-1 to t-24), Fourier harmonics,
  rolling statistics — all derived exclusively from past observations
- **Strict chronological validation**: TimeSeriesSplit (10 folds), no data shuffling
- **Bayesian hyperparameter optimization**: Optuna framework with 80 trials per model
- **Statistical significance**: Diebold-Mariano tests confirm all improvements (p < 0.05)
- **36 unit tests**: mathematically certify zero data leakage throughout the pipeline
- **Deep learning comparison**: PyTorch LSTM empirically proven to underperform
  traditional regularized linear regression on this 633-sample dataset

---

## Dataset

**Source**: US Energy Information Administration (EIA) Monthly Energy Review
**File**: Data/Dataset/USA ENGERY PREDICTION.xlsx
**Period**: January 1973 to December 2021 (633 monthly observations)
**Sectors**: Residential, Commercial, Industrial, Transportation (Trillion BTU)
**Reference**: Malakouti et al. (2025), South African Journal of Chemical Engineering
