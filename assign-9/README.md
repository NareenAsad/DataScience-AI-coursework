# Assignment 9 – Artificial Neural Network (ANN)

This assignment implements a feedforward Artificial Neural Network (ANN) using Keras to predict house prices.

## Tasks Completed
- Built an ANN with Dense and Dropout layers.  
- Used ReLU activation and Adam optimizer for training.  
- Trained for 100 epochs with validation monitoring.  
- Evaluated performance using MAE and RMSE, and compared results with previous models (Linear Regression, Random Forest).

## Key Insights
- The ANN successfully modeled nonlinear relationships between input features and SalePrice.  
- However, the model’s MAE (35,933) and RMSE (46,474) were higher than those of Linear Regression and Random Forest, indicating possible underfitting or insufficient tuning.  
- Increasing the number of neurons, optimizing learning rate, and adding regularization may improve results in future iterations.  
- Random Forest remains the best-performing model so far in terms of accuracy and generalization.

## Files
- `Assignment9_ANN.ipynb`  
- `train_cleaned.csv`

