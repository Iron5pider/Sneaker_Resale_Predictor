# main.py

from src.preprocess import preprocess_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import pandas as pd

# Load and preprocess the training data
data = pd.read_csv('datasets/stockx.csv')
X_processed, y = preprocess_data(data)

# Split the data into training and testing sets (assuming this hasn't been done in preprocess_data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Save the trained model for future predictions
dump(rf_regressor, 'random_forest_model.joblib')

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Calculate performance metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse}")
print(f"Test R^2: {r2}")

# If you also want to save the actual predictions
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv('model_predictions.csv', index=False)
