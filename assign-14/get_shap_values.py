import shap
import pandas as pd
import joblib
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

# Force stdout to utf-8 just in case, though simple print is safer
sys.stdout.reconfigure(encoding='utf-8')

try:
    model = joblib.load("house_price_model.pkl")
    df = pd.read_csv("train_cleaned.csv")
    X = df[['GrLivArea', 'OverallQual', 'GarageCars']]
    
    # Consistent sample
    X_sample = X.sample(n=50, random_state=42)
    
    # Explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
    except:
         # Fallback
        background = shap.sample(X_sample, 10)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample)

    # Print clean values
    print("--- DATA ---")
    row = X_sample.iloc[0]
    print(f"GrLivArea: {row['GrLivArea']}")
    print(f"OverallQual: {row['OverallQual']}")
    print(f"GarageCars: {row['GarageCars']}")
    
    print("--- SHAP ---")
    vals = shap_values[0]
    print(f"GrLivArea_SHAP: {vals[0]}")
    print(f"OverallQual_SHAP: {vals[1]}")
    print(f"GarageCars_SHAP: {vals[2]}")

    print("--- MODEL ---")
    if hasattr(model, 'feature_importances_'):
        print(f"FeatureImportances: {model.feature_importances_}")
    
except Exception as e:
    print(f"ERROR: {e}")
