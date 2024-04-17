from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from src.preprocess import preprocess_data  # Make sure to import your preprocessing function
from joblib import dump

filepath = '../datasets/stockx.csv'

# preprocess_data returns the preprocessed train and test sets

X_train, X_test, y_train, y_test = preprocess_data(filepath)

# Initialize the Random Forest Regressor with 100 trees
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1)

# Train the regressor on the training data
rf_regressor.fit(X_train, y_train)

# Predict on the training and test sets
y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)

# Evaluate the performance of the Random Forest Regressor
train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Random Forest training RMSE: {train_rmse}')
print(f'Random Forest test RMSE: {test_rmse}')
print(f'Random Forest training R^2: {train_r2}')
print(f'Random Forest test R^2: {test_r2}')

# can also use the model to make predictions on new data
# new_data = ... # Load or define new data here
# predictions = rf_regressor.predict(new_data)

# or, save the model using joblib for later
dump(rf_regressor, 'random_forest_model.joblib')
