import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR  # Import SVR from sklearn
import Preprocessor
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os



# Functions to run the time series rental models 

def LinearRegression_Rent_Model(df, postcode, months_ahead):
    postcode = int(postcode)

    # Extract time series data for a given postcode
    time_series_data = df[df['Postcode'] == postcode].iloc[:, 1:]

    if time_series_data.empty:
        print(f"No data found for postcode {postcode}")
        return None

    # Extract the last date in the dataset
    last_month_col = time_series_data.columns[-1]
    last_month_date = datetime.strptime(last_month_col, '%m-%Y')

    # Calculate the current date
    current_date = datetime.now()

    # Find how many months ahead we are from the last recorded data
    months_since_last_record = (current_date.year - last_month_date.year) * 12 + current_date.month - last_month_date.month
    
    # Adjust future month prediction based on current date
    future_month = len(time_series_data.columns) + months_ahead + months_since_last_record
    
    # Prepare X and y for the regression model
    X = np.array(range(len(time_series_data.columns))).reshape(-1, 1)
    y = time_series_data.values.flatten()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the rent price for the future month
    predicted_price = model.predict([[future_month]])
    print(f'Predicted Rent Price for {postcode} in {months_ahead} months: {predicted_price[0]} using Linear Regression')
    
    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Linear regression MSE: {mse}, R²: {r2}, MAE: {mae}, RMSE: {rmse}')
    X = np.array(range(len(time_series_data.columns))).reshape(-1, 1)

    y = time_series_data.values.flatten().astype(float)  # Ensure rent prices are floats

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Ensure y_test and y_pred are floats
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)

    # Calculate and apply the bias adjustment
    bias = np.mean(y_test - y_pred) + (mse + mae)/10  # Calculate the average prediction bias
    adjusted_price = predicted_price + bias  # Apply the bias correction
    adjusted_price = adjusted_price[0]  # Convert to a scalar value
    print(f'Adjusted Rent Price: {adjusted_price}')

    return adjusted_price

def RandomForest_Rent_Model(df, postcode, months_ahead):
    postcode = int(postcode)

    time_series_data = df[df['Postcode'] == postcode].iloc[:, 1:]

    print(f"\nTime series data for postcode {postcode}:\n{time_series_data.head()}\n")

    if time_series_data.empty:
        print(f"No data found for postcode {postcode}")
        return None

    X = np.array(range(len(time_series_data.columns))).reshape(-1, 1)
    y = time_series_data.values.flatten()

    if len(X) != len(y):
        print(f"Inconsistent number of samples: {len(X)} months, but {len(y)} rent prices")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(n_estimators=500, max_depth=50, min_samples_split=2, random_state=42)

    # Perform Grid Search with Cross Validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    
    best_rf = grid_search.best_estimator_

    # Calculate months since the last recorded date
    last_month_col = time_series_data.columns[-1]
    last_month_date = datetime.strptime(last_month_col, '%m-%Y')
    current_date = datetime.now()
    months_since_last_record = (current_date.year - last_month_date.year) * 12 + current_date.month - last_month_date.month

    # Adjust future month prediction
    future_month = len(time_series_data.columns) + months_ahead + months_since_last_record
    predicted_price = best_rf.predict([[future_month]])

    print(f'Predicted Rent Price for {postcode} in {months_ahead} months: {predicted_price[0]} using Random Forest')

    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Random Forest MSE: {mse}, R²: {r2}, MAE: {mae}, RMSE: {rmse}')
    
    bias =   (mse + mae) + (mse + mse)%10 # Adjust by the average underprediction
    adjusted_price = predicted_price + bias
    adjusted_price = adjusted_price[0]  # Convert to a scalar value
    print(f'Adjusted Rent Price: {adjusted_price}')

    return adjusted_price


def SVR_Rent_Model(df, postcode, months_ahead):
    postcode = int(postcode)

    # Extract time series data for the given postcode
    time_series_data = df[df['Postcode'] == postcode].iloc[:, 1:]

    print(f"\nTime series data for postcode {postcode}:\n{time_series_data.head()}\n")

    if time_series_data.empty:
        print(f"No data found for postcode {postcode}")
        return None

    # Create X (months) and y (rental prices)
    X = np.array(range(len(time_series_data.columns))).reshape(-1, 1)
    y = time_series_data.values.flatten()

    # Check for data consistency
    if len(X) != len(y):
        print(f"Inconsistent number of samples: {len(X)} months, but {len(y)} rent prices")
        return None

    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Perform 10-fold cross-validation
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_mse_scores = cross_val_score(model, X_scaled, y_scaled, cv=kf, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(model, X_scaled, y_scaled, cv=kf, scoring='r2')

    # Convert negative MSE to positive for interpretation
    cv_mse_scores = -cv_mse_scores
    
    print(f"\n10-Fold Cross-Validation Results:")
    print(f"Average MSE: {np.mean(cv_mse_scores)}")
    print(f"Average R²: {np.mean(cv_r2_scores)}")

    # Train the SVR model on the entire training set after cross-validation
    model.fit(X_scaled, y_scaled)

    # Calculate months since the last recorded date
    last_month_col = time_series_data.columns[-1]
    last_month_date = datetime.strptime(last_month_col, '%m-%Y')
    current_date = datetime.now()
    months_since_last_record = (current_date.year - last_month_date.year) * 12 + current_date.month - last_month_date.month

    # Adjust future month prediction and scale it
    future_month = np.array([[len(time_series_data.columns) + months_ahead + months_since_last_record]])
    future_month_scaled = scaler_X.transform(future_month)

    # Predict future rent price
    predicted_price_scaled = model.predict(future_month_scaled)
    predicted_price = scaler_y.inverse_transform([[predicted_price_scaled[0]]])[0][0]

    print(f'Predicted Rent Price for {postcode} in {months_ahead} months: {predicted_price} using SVR')


    # Evaluate the model on the test data
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    print(f'SVR MSE: {mse}, R²: {r2}, MAE: {mae}, RMSE: {rmse}')


    bias =   (mse + mae)/10 # Adjust by the average underprediction
    # Calculate and apply the bias adjustment
    adjusted_price = predicted_price + bias

    # Inverse transform the bias-adjusted price
    adjusted_price = scaler_y.inverse_transform([[adjusted_price]])[0][0]
    adjusted_price = predicted_price + bias  # Apply the bias correction
    print(f'Adjusted Rent Price: {adjusted_price}')


    return adjusted_price

def Get_Rental_Prediction(postcode, bedrooms, property_type, months_ahead):
    # Initialize Prediction variable to None
    Prediction = None


    # Convert the variables to appropriate types
    postcode = int(postcode)
    rooms = int(bedrooms)
    months_ahead = int(months_ahead)
    housing_type = str(property_type)

    if rooms == 1 and housing_type == 'F':
        dataset_path = 'Dataset\\processed_Rent_1BF_Final.csv'
        df = Preprocessor.TimeSeriesPreprocessor(dataset_path)
        Prediction = RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 2 and housing_type == 'F':
        dataset_path = 'Dataset\\processed_Rent_2BF_Final.csv'
        df = Preprocessor.TimeSeriesPreprocessor(dataset_path)
        Prediction = RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 3 and housing_type == 'F':
        dataset_path = 'Dataset\\processed_Rent_2BH_Final.csv'
        df = Preprocessor.TimeSeriesPreprocessor(dataset_path)
        Prediction = RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 2 and housing_type == 'H':
        dataset_path = 'Dataset\\processed_Rent_3BF_Final.csv'
        df = Preprocessor.TimeSeriesPreprocessor(dataset_path)
        print(df.columns)
        print(df.head())
        Prediction = RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 3 and housing_type == 'H':
        dataset_path = 'Dataset\\processed_Rent_3BH_Final.csv'
        df = Preprocessor.TimeSeriesPreprocessor(dataset_path)
        Prediction = SVR_Rent_Model(df, postcode, months_ahead)
    elif rooms == 4 and housing_type == 'H':
        dataset_path = 'Dataset\\processed_Rent_4BH_Final.csv'
        df = Preprocessor.TimeSeriesPreprocessor(dataset_path)
        Prediction = LinearRegression_Rent_Model(df, postcode, months_ahead)

    # Check if Prediction was generated, otherwise return None
    if Prediction is None:
        print("No matching model found for the provided input.")
    
    return Prediction

