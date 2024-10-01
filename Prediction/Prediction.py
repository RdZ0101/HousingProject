import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import preprocess
from Visualization import Visualization

def run_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)

def run_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)

def run_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R^2': r2}

if __name__ == "__main__":
    file_path = "C:\Study\Semester_2_2024\COS30049\Draft\Melbourne_housing_FULL.csv"

    # Choose preprocessing based on the model
    choice = input("Which model would you like to run? (random_forest/linear_regression/knn): ")

    if choice == "random_forest":
        # No outlier handling or normalization for Random Forest
        X, y = preprocess.preprocess_data(file_path, handle_outliers=False, normalize=False)
    else:
        # Handle outliers and normalize for Linear Regression and KNN
        X, y = preprocess.preprocess_data(file_path, handle_outliers=True, normalize=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Run the selected model
    if choice == "random_forest":
        metrics = run_random_forest(X_train, y_train, X_test, y_test)
    elif choice == "linear_regression":
        metrics = run_linear_regression(X_train, y_train, X_test, y_test)
    elif choice == "knn":
        metrics = run_knn(X_train, y_train, X_test, y_test)
    else:
        print("Invalid choice")
        exit()

    # Print the results
    print(f"Model: {choice}")
    print(f"MAE: {metrics['MAE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"R^2: {metrics['R^2']}")