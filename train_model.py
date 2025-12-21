import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

print("="*70)
print("🏠 HOUSE PRICE PREDICTION MODEL - TRAINING SCRIPT")
print("="*70)

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================

print("\n STEP 1: Loading data...")

# Option A: If you have a CSV file
try:
    # Try to load train_cleaned.csv (I saw this in your files)
    df = pd.read_csv('train_cleaned.csv')
    print(f"Data loaded from train_cleaned.csv")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print(" train_cleaned.csv not found!")
    print("\n Alternative: Using sample data for demonstration...")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4.5, n_samples),
        'sqft_living': np.random.randint(500, 5000, n_samples),
        'sqft_lot': np.random.randint(1000, 20000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(4, 13, n_samples),
        'sqft_above': np.random.randint(500, 4000, n_samples),
        'sqft_basement': np.random.randint(0, 1500, n_samples),
        'yr_built': np.random.randint(1900, 2025, n_samples),
        'yr_renovated': np.random.choice([0] + list(range(1990, 2025)), n_samples),
        'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005], n_samples),
        'lat': np.random.uniform(47.3, 47.8, n_samples),
        'long': np.random.uniform(-122.5, -122.0, n_samples),
    })
    
    # Create target variable (price) based on features
    df['price'] = (
        df['sqft_living'] * 200 +
        df['bedrooms'] * 20000 +
        df['bathrooms'] * 30000 +
        df['grade'] * 15000 +
        df['waterfront'] * 100000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    print(f" Sample data created for demonstration")
    print(f"   Shape: {df.shape}")

# Display basic info
print(f"\n Data Overview:")
print(df.head())
print(f"\n Data Statistics:")
print(df.describe())

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================

print("\n STEP 2: Preparing data...")

# Define features and target
# ADJUST THIS based on your actual data!
feature_columns = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'
]

# Check if price column exists
if 'price' in df.columns:
    target_column = 'price'
elif 'SalePrice' in df.columns:
    target_column = 'SalePrice'
elif 'median_house_value' in df.columns:
    target_column = 'median_house_value'
else:
    print("No price column found!")
    print(f"Available columns: {list(df.columns)}")
    print("\n Please specify your target column name")
    exit()

# Filter features that exist in the dataframe
available_features = [col for col in feature_columns if col in df.columns]
print(f"\nUsing {len(available_features)} features:")
for i, feat in enumerate(available_features, 1):
    print(f"   {i}. {feat}")

# Prepare X and y
X = df[available_features]
y = df[target_column]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle missing values
if X.isnull().sum().sum() > 0:
    print(f"\nFound {X.isnull().sum().sum()} missing values, filling with median...")
    X = X.fillna(X.median())

# ============================================================================
# STEP 3: SPLIT DATA
# ============================================================================

print("\nSTEP 3: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 4: SCALE FEATURES (OPTIONAL BUT RECOMMENDED)
# ============================================================================

print("\nSTEP 4: Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================

print("\n🤖 STEP 5: Training model...")
print("   Using Random Forest Regressor...")
print("   This may take a minute...")

model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum depth of trees
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples in leaf
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

# Train the model
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

# ============================================================================
# STEP 6: EVALUATE MODEL
# ============================================================================

print("\nSTEP 6: Evaluating model...")

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nTraining Set Performance:")
print(f"   R² Score: {train_r2:.4f}")
print(f"   RMSE: ${train_rmse:,.2f}")
print(f"   MAE: ${train_mae:,.2f}")

print(f"\nTest Set Performance:")
print(f"   R² Score: {test_r2:.4f}")
print(f"   RMSE: ${test_rmse:,.2f}")
print(f"   MAE: ${test_mae:,.2f}")

# Feature importance
print(f"\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# STEP 7: SAVE MODEL AND SCALER
# ============================================================================

print("\nSTEP 7: Saving model and scaler...")

# Save model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

model_size = os.path.getsize('house_price_model.pkl')
print(f"Model saved: house_price_model.pkl")
print(f"   Size: {model_size:,} bytes ({model_size/1024:.2f} KB)")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

scaler_size = os.path.getsize('scaler.pkl')
print(f"Scaler saved: scaler.pkl")
print(f"   Size: {scaler_size:,} bytes ({scaler_size/1024:.2f} KB)")

# ============================================================================
# STEP 8: TEST LOADING
# ============================================================================

print("\nSTEP 8: Testing model loading...")

# Load model
with open('house_price_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Make test prediction
test_data = np.array([[3, 2, 2000, 5000, 1, 0, 0, 3, 7, 1500, 500, 1990, 0, 98001, 47.5, -122.2]])
test_data_scaled = loaded_scaler.transform(test_data)
test_prediction = loaded_model.predict(test_data_scaled)[0]

print(f"Model loaded successfully!")
print(f"Scaler loaded successfully!")
print(f"Test prediction: ${test_prediction:,.2f}")
