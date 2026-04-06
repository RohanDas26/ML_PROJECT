
import pandas as pd
import re

PATH = r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\EnergyForecasting_v2_Production\Documentation\Final_Project_Report.md"

with open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

sectors = ['Industrial', 'Commercial', 'Residential', 'Transportation']

for sector in sectors:
    # 1. Actual vs Predicted -> Forecast
    old_av_p1 = f"Results/figures/{sector}_optuna_actual_vs_predicted.png"
    old_av_p2 = f"Results\\\\figures\\\\{sector}_optuna_actual_vs_predicted.png"
    
    new_av_p1 = f"Results/figures/Phase7_Final/{sector}_Phase7_Forecast.png"
    new_av_p2 = f"Results\\\\figures\\\\Phase7_Final\\\\{sector}_Phase7_Forecast.png"
    
    text = text.replace(old_av_p1, new_av_p1)
    text = text.replace(old_av_p2, new_av_p2)
    
    # 2. Feature Importance
    old_fi_1 = f"Results/figures/{sector}_optuna_feature_importance.png"
    old_fi_2 = f"Results\\\\figures\\\\{sector}_optuna_feature_importance.png"
    
    old_fi_3 = f"Results/figures/{sector}_feature_importance.png"
    old_fi_4 = f"Results\\\\figures\\\\{sector}_feature_importance.png"
    
    new_fi_1 = f"Results/figures/Phase7_Final/{sector}_Phase7_Importance.png"
    new_fi_2 = f"Results\\\\figures\\\\Phase7_Final\\\\{sector}_Phase7_Importance.png"
    
    text = text.replace(old_fi_1, new_fi_1)
    text = text.replace(old_fi_2, new_fi_2)
    text = text.replace(old_fi_3, new_fi_1)
    text = text.replace(old_fi_4, new_fi_2)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("Linked to Phase 7 images.")
