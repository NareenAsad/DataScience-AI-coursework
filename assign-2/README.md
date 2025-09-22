# Assignment 2 – Data Cleaning (House Price Prediction)

This assignment focuses on applying **data cleaning techniques** to a House Price
Prediction dataset and preparing a **Before vs After Cleaning** report.

---

## 📂 Dataset

- **Source:** [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- **Format:** CSV (train.csv)

---

## 🎯 Objective
1. Identify and handle **missing values**, **duplicates**, and **incorrect data types**.  
2. Detect and treat **outliers** where appropriate.  
3. Produce a **Before vs After Cleaning** comparison to show improvements.

---

## 🛠️ Cleaning Steps

The following main steps were performed inside  
[`Assignment2_HousePrice_Cleaning.ipynb`](Assignment2_HousePrice_Cleaning.ipynb):

1. **Initial Profiling (Before Cleaning)**  
   - Used `df.shape`, `df.info()`, `df.describe(include='all')`, and `df.isnull().sum()`  
     to capture dataset size, data types, and missing-value counts.

2. **Handling Missing Values**  
   - Numeric columns (e.g. `LotFrontage`) filled with median values:
     ```python
     df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
     ```
   - Categorical columns (e.g. `Alley`) filled with `"None"` or mode:
     ```python
     df['Alley'] = df['Alley'].fillna('None')
     ```

3. **Removing Duplicates**  
   ```python
   df.drop_duplicates(inplace=True)
   ```

4. **Correcting Data Types**

   * Example: `MSSubClass` stored as category instead of integer.

     ```python
     df['MSSubClass'] = df['MSSubClass'].astype(str)
     ```

5. **Outlier Treatment (Optional)**

   * Extreme `GrLivArea` values capped by filtering rows with
     `GrLivArea < 4500`.

6. **Post-Cleaning Profiling (After Cleaning)**

   * Repeated the same profiling commands to verify
     reductions in missing values and duplicates.

---

## 📊 Before vs After Summary

| Feature             | Before Cleaning           | After Cleaning |
| ------------------- | ------------------------- | -------------- |
| **Shape**           | 1460 × 81                 | 1458 × 81      |
| **Missing Columns** | 19 columns                | 0 columns\*    |
| **Duplicates**      | 2 rows                    | 0 rows         |
| **Notable Fix**     | `LotFrontage` median fill | Completed      |

\*Some columns may remain intentionally unfilled if missingness
represents a valid “None” category (e.g., `PoolQC`, `Fence`).

---

## 🔑 Key Insights

* Proper handling of missing values (especially numeric fields) is
  essential for modeling.
* Converting data types improves memory usage and model interpretation.
* Outlier removal prevents skewed training and unrealistic predictions.

---