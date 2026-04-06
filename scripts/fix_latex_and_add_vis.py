
import os

PATH = r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\EnergyForecasting_v2_Production\Documentation\Final_Project_Report.md"

with open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Fix LaTeX issues caused by string escaping in previous Python scripts
# '\t' became a tab, '\a' became an ASCII bell
text = text.replace("\text", "\\text")
text = text.replace("\t", "\\t") # Just in case

# Python executes \t as tab (which is chr(9))
text = text.replace(chr(9) + "ext", "\\text")
text = text.replace(chr(9) + "imes", "\\times")

# Python executes \a as bell (which is chr(7))
text = text.replace(chr(7) + "pprox", "\\approx")

# Let's also do direct replacements just in case
text = text.replace(" pprox", "\\approx")
text = text.replace("	imes", "\\times")
text = text.replace("	ext", "\\text")


# We want to add a 6.7 Comprehensive Visual Evidence
visual_section = """
### 6.7 Comprehensive Visual Evidence

To fully substantiate the claims of robust predictive behavior across all sectors, the following visual proofs are provided. These representations detail the alignment between actual and Optuna-optimized predicted values, the relative feature importances, and the structural decomposition of the underlying historical data. 

*(Note for formatting: If rendering fails, the direct local File Locations are provided for straightforward inclusion into external documents.)*

#### 6.7.1 Sector Forecasts (Actual vs. Predicted)
These graphs showcase the performance of the Phase 7 Absolute Ceiling models per sector over the unseen testing timeline.

**Industrial Sector (Lasso)**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Industrial_optuna_actual_vs_predicted.png`
![Industrial Forecast](../../Results/figures/Industrial_optuna_actual_vs_predicted.png)

**Commercial Sector (Ridge)**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Commercial_optuna_actual_vs_predicted.png`
![Commercial Forecast](../../Results/figures/Commercial_optuna_actual_vs_predicted.png)

**Residential Sector (OMP)**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Residential_optuna_actual_vs_predicted.png`
![Residential Forecast](../../Results/figures/Residential_optuna_actual_vs_predicted.png)

**Transportation Sector (ElasticNet)**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Transportation_optuna_actual_vs_predicted.png`
![Transportation Forecast](../../Results/figures/Transportation_optuna_actual_vs_predicted.png)

#### 6.7.2 Key Drivers (Feature Importance)
The feature importance plots detail the specific lagged and Fourier variables the linear regularizers utilized to define the predictions. Notice the overwhelming importance of the $t-12$ and $t-24$ lags in seasonal structures.

**Industrial Feature Importance**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Industrial_optuna_feature_importance.png`
![Industrial Features](../../Results/figures/Industrial_optuna_feature_importance.png)

**Commercial Feature Importance**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Commercial_optuna_feature_importance.png`
![Commercial Features](../../Results/figures/Commercial_optuna_feature_importance.png)

**Residential Feature Importance**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Residential_optuna_feature_importance.png`
![Residential Features](../../Results/figures/Residential_optuna_feature_importance.png)

**Transportation Feature Importance**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\figures\\Transportation_optuna_feature_importance.png`
![Transportation Features](../../Results/figures/Transportation_optuna_feature_importance.png)

#### 6.7.3 Structural Decompositions
The structural time series decompositions (STL) isolate the trend, seasonality, and residual noise for macro-analysis contexts. These are crucial for observing macroeconomic "structural breaks", particularly the 2020 Transportation dip.

**Industrial Components**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\full_report\\figures\\Industrial_stl_decomposition.png`
![Industrial STL](../../Results/full_report/figures/Industrial_stl_decomposition.png)

**Commercial Components**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\full_report\\figures\\Commercial_stl_decomposition.png`
![Commercial STL](../../Results/full_report/figures/Commercial_stl_decomposition.png)

**Residential Components**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\full_report\\figures\\Residential_stl_decomposition.png`
![Residential STL](../../Results/full_report/figures/Residential_stl_decomposition.png)

**Transportation Components**
- *Location:* `c:\\Users\\Rohan Das\\Desktop\\ML PROJECT CODE\\ML_PROJECT\\EnergyForecasting_v2_Production\\Results\\full_report\\figures\\Transportation_stl_decomposition.png`
![Transportation STL](../../Results/full_report/figures/Transportation_stl_decomposition.png)

"""

# Insert visual_section right before "## 7. CONCLUSION"
text = text.replace("## 7. CONCLUSION", visual_section + "\n\n## 7. CONCLUSION")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("Latex fixed and full visuals appended successfully.")
