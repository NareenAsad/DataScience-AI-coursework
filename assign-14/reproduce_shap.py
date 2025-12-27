import shap
import pandas as pd
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading model and data...")
try:
    model = joblib.load("house_price_model.pkl")
    df = pd.read_csv("train_cleaned.csv")
    
    # Select features
    X = df[['GrLivArea', 'OverallQual', 'GarageCars']]
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Use SMALLER sample
    X_sample = X.sample(n=50, random_state=42)
    print(f"Using {X_sample.shape[0]} samples for SHAP analysis")
    
    print("Creating SHAP explainer...")
    try:
        explainer = shap.TreeExplainer(model)
        print("Using TreeExplainer")
    except:
        print("TreeExplainer failed, using KernelExplainer...")
        background = shap.sample(X_sample, 10)
        explainer = shap.KernelExplainer(model.predict, background)
        
    print("Calculating SHAP values...")
    try:
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
    except Exception as e:
        print(f"Error: {e}")
        shap_values = np.array([explainer.shap_values(X_sample.iloc[[i]], check_additivity=False)[0] 
                                for i in range(len(X_sample))])

    print("\n[6/7] Local explanation for first sample:")
    print("-" * 60)
    print("Sample Input:")
    for col in X_sample.columns:
        print(f"  {col:15s}: {X_sample.iloc[0][col]}")

    print("\nSHAP Contributions:")
    # Handle expected_value being scalar or list
    if hasattr(explainer, 'expected_value'):
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0] # assuming single output regression
    else:
        base_value = 0

    for i, col in enumerate(X_sample.columns):
        contribution = shap_values[0][i]
        direction = "↑" if contribution > 0 else "↓"
        print(f"  {col:15s}: {contribution:+8.2f} {direction}")

    print("-" * 60)
    
    # inspect the model
    print("\nModel Info:")
    print(type(model))
    if hasattr(model, 'feature_importances_'):
        print("Feature Importances:", model.feature_importances_)
    if hasattr(model, 'coef_'):
        print("Coefficients:", model.coef_)

except Exception as e:
    print(f"Fatal Error: {e}")
