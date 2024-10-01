import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR  # Import SVR from sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Data Preprocessing Function (Data Cleaning)
def clean_data(inputFileName):
    df = pd.read_csv(inputFileName)
    
    # Drop duplicates if 'Suburb' and 'Postcode' are both duplicated
    df.drop_duplicates(subset=['Suburb', 'Postcode'], keep='first', inplace=True)

    # Handle missing values in 'Postcode'
    df['Postcode'] = df['Postcode'].fillna(0)  # Fill missing postcodes with 0

    # Convert 'Postcode' to integer for consistency
    df['Postcode'] = df['Postcode'].astype(int)

    # Remove any unwanted columns
    non_date_cols = ['Suburb', 'Postcode']
    
    # Extract columns that represent rent prices (assumed to be the columns after 'Postcode')
    date_cols = df.columns.difference(non_date_cols)

    # Convert the date column names to 'mm-yyyy' format
    new_date_cols = [pd.to_datetime(col, format='%b %Y').strftime('%m-%Y') for col in date_cols]

    # Create a mapping of old to new column names and rename
    column_mapping = dict(zip(date_cols, new_date_cols))
    df.rename(columns=column_mapping, inplace=True)

    # Clean up the DataFrame by removing unnecessary columns
    df_clean = df.drop(columns=['Suburb', 'SellerG', 'Address', 'Regionname', 'Propertycount', 'CouncilArea'], errors='ignore')

    print(df_clean.head())
    
    # Return the cleaned dataframe
    return df_clean

# Function for Linear Regression model
def LinearRegression_Rent_Model(df, postcode, months_ahead):
    postcode = int(postcode)

    # Extract time series data for a given postcode
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

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_month = len(time_series_data.columns) + months_ahead
    predicted_price = model.predict([[future_month]])

    print(f'Predicted Rent Price for {postcode} in {months_ahead} months: {predicted_price[0]}')

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Linear Regression MSE: {mse}, R²: {r2}')
    
    return model

# Function for Random Forest model with hyperparameter tuning
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

    rf = RandomForestRegressor(random_state=42)

    # Perform Grid Search with Cross Validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")

    best_rf = grid_search.best_estimator_

    future_month = len(time_series_data.columns) + months_ahead
    predicted_price = best_rf.predict([[future_month]])

    print(f'Predicted Rent Price for {postcode} in {months_ahead} months: {predicted_price[0]}')

    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Random Forest MSE: {mse}, R²: {r2}')
    
    return best_rf

# Function for SVR model
def SVR_Rent_Model(df, postcode, months_ahead):
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

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    future_month = len(time_series_data.columns) + months_ahead
    predicted_price = model.predict([[future_month]])

    print(f'Predicted Rent Price for {postcode} in {months_ahead} months: {predicted_price[0]}')

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'SVR MSE: {mse}, R²: {r2}')
    
    return model

# Main function to run the selected rental model
def run_rental_models(df, model_choice, postcode, months_ahead):
    if model_choice == '1':
        model = LinearRegression_Rent_Model(df, postcode, months_ahead)
    elif model_choice == '2':
        model = RandomForest_Rent_Model(df, postcode, months_ahead)
    elif model_choice == '3':
        model = SVR_Rent_Model(df, postcode, months_ahead)
    else:
        print("Invalid model choice.")
        return

    print(f'Predictions completed for postcode {postcode} for {months_ahead} months ahead.')

# Main execution
if __name__ == "__main__":
    postcode = '3067'  # Example postcode
    months_ahead = 2  # Example: Predict rent 2 months ahead

    # Preprocess the dataset (data cleaning)
    dataset_path = 'Rent_1BF_Final.csv'
    df_cleaned = clean_data(dataset_path)

    # Ask user for model choice
    model_choice = input("Enter 1 for Linear Regression, 2 for Random Forest, 3 for SVR: ").strip()

    # Run the selected model
    run_rental_models(df_cleaned, model_choice, postcode, months_ahead)
