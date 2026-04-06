
import re

PATH = r"c:\Users\Rohan Das\Desktop\ML PROJECT CODE\ML_PROJECT\EnergyForecasting_v2_Production\Documentation\Final_Project_Report.md"

with open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

sectors = ['Industrial', 'Commercial', 'Residential', 'Transportation']

for sector in sectors:
    # 1. Fixing the text Location string for Forecasts
    old_text_fc_1 = f"Results\\figures\\{sector}_optuna_actual_vs_predicted.png"
    new_text_fc   = f"Results\\figures\\Phase7_Final\\{sector}_Phase7_Forecast.png"
    text = text.replace(old_text_fc_1, new_text_fc)
    
    # 2. Fixing the text Location string for Importance
    old_text_fi_1 = f"Results\\figures\\{sector}_optuna_feature_importance.png"
    old_text_fi_2 = f"Results\\figures\\{sector}_feature_importance.png"
    new_text_fi   = f"Results\\figures\\Phase7_Final\\{sector}_Phase7_Importance.png"

    text = text.replace(old_text_fi_1, new_text_fi)
    text = text.replace(old_text_fi_2, new_text_fi)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("Text metadata locations in MD file updated successfully!")
