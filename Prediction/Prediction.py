import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import preprocess
import numpy as np

def run_random_forest(X_train, y_train, X_test, y_test, X_scaled, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_absolute_error')
    print(f'Average MAE for Random Forest (10-fold CV): {np.abs(np.mean(rf_scores)):.2f}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)

def run_linear_regression(X_train, y_train, X_test, y_test, X_scaled, y):
    model = LinearRegression()
    linear_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_absolute_error')
    print(f'Average MAE for Linear Regression (10-fold CV): {np.abs(np.mean(linear_scores)):.2f}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)

def run_knn(X_train, y_train, X_test, y_test, X_scaled, y):
    model = KNeighborsRegressor(n_neighbors=5)
    scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_absolute_error')
    print(f'Average MAE from 10-fold Cross-Validation: {np.abs(np.mean(scores)):.2f}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R^2': r2}

def main():
    file_path = "Melbourne_housing_FULL.csv"

    # Choose preprocessing based on the model
    choice = input("Which model would you like to run? (random_forest/linear_regression/knn): ")

    if choice == "random_forest":
        # No outlier handling or normalization for Random Forest
        X, y = preprocess.preprocess_data(file_path, handle_outliers=False, normalize=False)
    else:
        # Handle outliers and normalize for Linear Regression and KNN
        X, y = preprocess.preprocess_data(file_path, handle_outliers=True, normalize=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Run the selected model
    if choice == "random_forest":
        metrics = run_random_forest(X_train, y_train, X_test, y_test, X_scaled, y)
    elif choice == "linear_regression":
        metrics = run_linear_regression(X_train, y_train, X_test, y_test, X_scaled, y)
    elif choice == "knn":
        metrics = run_knn(X_train, y_train, X_test, y_test, X_scaled, y)
    else:
        print("Invalid choice")
        exit()

    # Print the results
    print(f"Model: {choice}")
    print(f"MAE: {metrics['MAE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"R^2: {metrics['R^2']}")

# This ensures the script can still be run directly
if __name__ == "__main__":
    main()
