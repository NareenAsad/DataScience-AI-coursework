# Week 14 – Ethics & Explainability (SHAP)

## 📋 Overview

This week focuses on making machine learning models **interpretable and explainable** using SHAP (SHapley Additive exPlanations). The goal is to understand **why** the model makes certain predictions, not just **what** it predicts.

---

## 🎯 Objectives

- Understand the importance of explainable AI in real-world applications
- Implement SHAP to interpret model predictions
- Analyze global feature importance across the dataset
- Explain individual predictions at the local level
- Discuss ethical implications of AI in sensitive domains like real estate

---

## 📁 Repository Structure

```
Week14_Ethics_Explainability/
│
├── Week14_Ethics&Explainability.ipynb    # Main Jupyter notebook
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── Data/
│   ├── train_cleaned.csv                  # Cleaned training data
│   └── house_price_model.pkl              # Trained model
│
├── Results/
│   ├── shap_values.csv                    # Computed SHAP values
│   ├── shap_sample_data.csv               # Sample data used for analysis
│   ├── feature_importance.csv             # Feature importance rankings
│   │
│   └── Visualizations/
│       ├── shap_global_importance.png     # Global feature importance
│       ├── shap_bar_importance.png        # Bar chart of importance
│       ├── shap_local_explanation.png     # Individual prediction explanation
│       └── shap_manual_importance.png     # Manual bar chart (fallback)
│
└── Assignment_14_Report.md                # Detailed assignment writeup
```

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 - 3.11 (SHAP may not work with Python 3.12+)
- Jupyter Notebook or JupyterLab
- Trained house price prediction model

### Install Dependencies

**Option 1: Quick Install**
```bash
pip install shap pandas numpy matplotlib scikit-learn joblib scipy
```

**Option 2: Using requirements.txt**
```bash
pip install -r requirements.txt
```

**Option 3: Conda (if using Anaconda)**
```bash
conda install -c conda-forge shap pandas numpy matplotlib scikit-learn
```

### Verify Installation

```python
import shap
print("SHAP version:", shap.__version__)
```

---

## 🚀 Usage

### Step 1: Load Model and Data

```python
import shap
import pandas as pd
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# Load dataset
df = pd.read_csv("train_cleaned.csv")
X = df[['GrLivArea', 'OverallQual', 'GarageCars']]
```

### Step 2: Create SHAP Explainer

```python
# Use a sample to prevent crashes
X_sample = X.sample(n=50, random_state=42)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample, check_additivity=False)
```

### Step 3: Generate Visualizations

```python
# Global importance
shap.summary_plot(shap_values, X_sample)

# Feature importance bar chart
shap.summary_plot(shap_values, X_sample, plot_type="bar")

# Local explanation for one prediction
shap.plots.waterfall(shap_values[0])
```

---

## 📊 Features Analyzed

| Feature | Description | Type |
|---------|-------------|------|
| **GrLivArea** | Above-ground living area (sq ft) | Continuous |
| **OverallQual** | Overall material and finish quality (1-10) | Ordinal |
| **GarageCars** | Garage capacity (number of cars) | Discrete |

---

## 🔍 Key Findings

### Global Feature Importance

1. **GrLivArea** (Living Area) - **Highest Impact**
   - Larger homes consistently command higher prices
   - Most influential feature in the model

2. **OverallQual** (Quality Rating) - **High Impact**
   - Quality improvements significantly boost valuations
   - Second most important predictor

3. **GarageCars** (Garage Size) - **Moderate Impact**
   - Garage capacity adds value but is secondary to size and quality

### Example Prediction Explanation

**Input:**
- GrLivArea: 1,800 sq ft
- OverallQual: 7 (Good)
- GarageCars: 2

**SHAP Breakdown:**
- Base prediction: $180,000
- GrLivArea contribution: +$25,000
- OverallQual contribution: +$12,000
- GarageCars contribution: +$3,000
- **Final prediction: ~$220,000**

---

## ⚖️ Ethics & Explainability

### Why Explainability Matters

1. **Transparency**: Stakeholders deserve to understand how predictions are made
2. **Trust**: Explainable models build confidence in AI systems
3. **Bias Detection**: SHAP reveals if models rely on problematic features
4. **Accountability**: Clear explanations enable responsible AI deployment

### Ethical Considerations

**Potential Biases:**
- Historical data may reflect past discrimination
- Quality ratings may be subjective and inconsistent
- Model may undervalue properties in certain neighborhoods

**Mitigation Strategies:**
- Regular bias audits using SHAP across demographic groups
- Transparent communication of model limitations
- Human oversight for high-stakes decisions
- Continuous monitoring for disparate impact

---

## 📝 Class Task

**Objective:** Use SHAP to explain model predictions

**Deliverables:**
1. Global feature importance visualization
2. Local explanation for individual predictions
3. Written interpretation of SHAP results

**Reflection:**
> SHAP analysis revealed that living area is the most influential factor in house price predictions, followed by overall quality and garage capacity. Global explanations showed feature importance across the dataset, while local explanations demonstrated how individual feature values contribute to specific predictions. This task highlighted the importance of model transparency in building trust and identifying potential biases.

---

## 📄 Assignment 14

**Task:** Explain predictions of your model (why X → Y)

**Requirements:**
1. Implement SHAP for model interpretation
2. Generate global and local explanations
3. Discuss ethical implications
4. Create visualizations of feature importance
5. Write comprehensive analysis report

**Deliverables:**
- ✅ Jupyter notebook with SHAP implementation
- ✅ SHAP visualizations (PNG files)
- ✅ Feature importance analysis
- ✅ Ethics discussion
- ✅ Assignment report (markdown/PDF)

---

## 🛠️ Troubleshooting

### Kernel Crashes

**Problem:** Jupyter kernel crashes when running SHAP
**Solution:**
- Reduce sample size: `X_sample = X.sample(n=30)`
- Use non-interactive backend: `matplotlib.use('Agg')`
- Run plotting separately from calculations

### Installation Issues

**Problem:** SHAP installation fails
**Solution:**
```bash
pip install shap --no-deps
pip install pandas numpy scipy scikit-learn
```

### Memory Errors

**Problem:** Out of memory errors
**Solution:**
- Use smaller sample size
- Close other applications
- Restart Python kernel
- Use `check_additivity=False`

### Visualization Errors

**Problem:** Plots don't display or save
**Solution:**
- Save data first: `shap_df.to_csv("shap_values.csv")`
- Create plots separately in new notebook
- Use manual matplotlib plots as fallback

---

## 📚 References

### Documentation
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)
- [Google AI Principles](https://ai.google/principles/)

### Research Papers
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"
- Molnar, C. (2020). "Interpretable Machine Learning"

### Additional Resources
- [SHAP GitHub Repository](https://github.com/slundberg/shap)
- [Explainable AI Tutorials](https://github.com/interpretml/interpret)

---

## 🎓 Learning Outcomes

After completing Week 14, you should be able to:

- ✅ Explain why model interpretability is crucial in AI
- ✅ Implement SHAP for global and local explanations
- ✅ Interpret SHAP visualizations and values
- ✅ Identify potential biases using explainability tools
- ✅ Discuss ethical implications of AI in sensitive domains
- ✅ Communicate model behavior to non-technical stakeholders

---

## 🏆 Project Milestone Achieved

✅ **Explainability Section Complete**
- Model predictions are now interpretable
- Ethical considerations documented
- Transparency established for stakeholders
- Trust enhanced through clear explanations

---

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review SHAP documentation
3. Verify all dependencies are installed
4. Try the verification script in `requirements.txt`

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Features Analyzed** | 3 |
| **Samples Used** | 50 |
| **Most Important Feature** | GrLivArea |
| **Explainability Method** | SHAP (TreeExplainer) |
| **Visualizations Generated** | 4 |

---

## 👨‍💻 Author

**Course:** AI/ML Course - Week 14  
**Topic:** Ethics & Explainability  
**Method:** SHAP (SHapley Additive exPlanations)  
**Date:** December 2024  

---

## 📝 License

This project is part of an academic assignment and is intended for educational purposes only.

---

## ✅ Checklist

- [x] Install SHAP and dependencies
- [x] Load model and data
- [x] Create SHAP explainer
- [x] Calculate SHAP values
- [x] Generate global importance plots
- [x] Create local explanation visualizations
- [x] Write feature importance analysis
- [x] Discuss ethical implications
- [x] Complete assignment report
- [x] Document findings in README

---

**Status:** ✅ Complete
