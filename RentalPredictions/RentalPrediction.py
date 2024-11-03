import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from .Preprocessor import TimeSeriesPreprocessor
from datetime import datetime

# Functions to run the time series rental models

def LinearRegression_Rent_Model(df, postcode, months_ahead):
    postcode = int(postcode)
    time_series_data = df[df['Postcode'] == postcode].iloc[:, 1:]

    if time_series_data.empty:
        print(f"No data found for postcode {postcode}")
        return None

    # Prepare X and y for the regression model
    X = np.array(range(len(time_series_data.columns))).reshape(-1, 1)
    y = time_series_data.values.flatten()
    
    # Calculate future prediction index
    future_month = len(time_series_data.columns) + months_ahead
    
    model = LinearRegression().fit(X, y)
    predicted_price = model.predict([[future_month]])[0]
    
    return round(predicted_price, 2)

def RandomForest_Rent_Model(df, postcode, months_ahead):
    postcode = int(postcode)
    time_series_data = df[df['Postcode'] == postcode].iloc[:, 1:]

    if time_series_data.empty:
        print(f"No data found for postcode {postcode}")
        return None

    # Prepare X and y for the regression model
    X = np.array(range(len(time_series_data.columns))).reshape(-1, 1)
    y = time_series_data.values.flatten()

    future_month = len(time_series_data.columns) + months_ahead
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    predicted_price = rf.predict([[future_month]])[0]
    
    return round(predicted_price, 2)

def SVR_Rent_Model(df, postcode, months_ahead):
    postcode = int(postcode)
    time_series_data = df[df['Postcode'] == postcode].iloc[:, 1:]

    if time_series_data.empty:
        print(f"No data found for postcode {postcode}")
        return None

    X = np.array(range(len(time_series_data.columns))).reshape(-1, 1)
    y = time_series_data.values.flatten()

    # Scale data for SVR
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    future_month_scaled = scaler_X.transform([[len(time_series_data.columns) + months_ahead]])
    
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_scaled, y_scaled)
    predicted_price_scaled = model.predict(future_month_scaled)
    predicted_price = scaler_y.inverse_transform([[predicted_price_scaled[0]]])[0][0]

    return round(predicted_price, 2)

def Get_Postcode(suburb):
    # Set a default value for PostCode
    PostCode = None
    suburb = suburb.upper()
    # List of files that contain suburb to postcode mapping
    DataFiles = [
        'Dataset\\Rent_1BF_Final.csv', 'Dataset\\Rent_2BF_Final.csv', 'Dataset\\Rent_3BF_Final.csv', 
        'Dataset\\Rent_2BH_Final.csv', 'Dataset\\Rent_3BH_Final.csv', 'Dataset\\Rent_4BH_Final.csv'
    ]
    
    for file in DataFiles:
        df = pd.read_csv(file)
        if suburb in df['Suburb'].values:
            PostCode = df[df['Suburb'] == suburb]['Postcode'].values[0]
            break  # Exit the loop once the suburb is found

    return PostCode

        

def Get_Rental_Prediction(suburb, bedrooms, property_type, months_ahead):
    # Convert suburb to postcode
    postcode = Get_Postcode(suburb)
    if postcode is None:
        print(f"Suburb '{suburb}' not found.")
        return None

    rooms = int(bedrooms)
    months_ahead = int(months_ahead)
    property_type = property_type.upper()
    if property_type not in ['F', 'H']:
        if property_type == 'FLAT':
            property_type = 'F'
        elif property_type == 'HOUSE':
            property_type = 'H'

    # Determine dataset path and model based on input criteria
    dataset_mapping = {
        ('F', 1): 'Dataset\\Rent_1BF_Final.csv',
        ('F', 2): 'Dataset\\Rent_2BF_Final.csv',
        ('F', 3): 'Dataset\\Rent_3BF_Final.csv',
        ('H', 2): 'Dataset\\Rent_2BH_Final.csv',
        ('H', 3): 'Dataset\\Rent_3BH_Final.csv',
        ('H', 4): 'Dataset\\Rent_4BH_Final.csv',
    }

    dataset_path = dataset_mapping.get((property_type, rooms))
    if not dataset_path:
        print("Invalid room and housing type combination.")
        return None

    # Process dataset and select model
    df = TimeSeriesPreprocessor(dataset_path)
    if rooms == 1 and property_type == 'F':
        return RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 2 and property_type == 'F':
        return RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 3 and property_type == 'F':
        return RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 2 and property_type == 'H':
        return RandomForest_Rent_Model(df, postcode, months_ahead)
    elif rooms == 3 and property_type == 'H':
        return SVR_Rent_Model(df, postcode, months_ahead)
    elif rooms == 4 and property_type == 'H':
        return LinearRegression_Rent_Model(df, postcode, months_ahead)
    
    return None

def Get_Historical_Rent_Prices(suburb, bedrooms, property_type, months_back):
    # Convert suburb to postcode
    postcode = Get_Postcode(suburb)
    if postcode is None:
        print(f"Suburb '{suburb}' not found.")
        return None

    rooms = int(bedrooms)
    property_type = property_type.upper()
    if property_type not in ['F', 'H']:
        if property_type == 'FLAT':
            property_type = 'F'
        elif property_type == 'HOUSE':
            property_type = 'H'

    # Determine dataset path based on input criteria
    dataset_mapping = {
        ('F', 1): 'Dataset\\Rent_1BF_Final.csv',
        ('F', 2): 'Dataset\\Rent_2BF_Final.csv',
        ('F', 3): 'Dataset\\Rent_3BF_Final.csv',
        ('H', 2): 'Dataset\\Rent_2BH_Final.csv',
        ('H', 3): 'Dataset\\Rent_3BH_Final.csv',
        ('H', 4): 'Dataset\\Rent_4BH_Final.csv',
    }

    dataset_path = dataset_mapping.get((property_type, rooms))
    if not dataset_path:
        print("Invalid room and housing type combination.")
        return None

    # Process the dataset
    df = TimeSeriesPreprocessor(dataset_path)
    time_series_data = df[df['Postcode'] == postcode].iloc[:, 1:]

    if time_series_data.empty:
        print(f"No data found for postcode {postcode}")
        return None

    # Get the last `months_back` months of data
    recent_data = time_series_data.iloc[:, -months_back:]
    
    # Prepare data as a list of date and price dictionaries
    historical_data = [
        {"date": col, "price": recent_data[col].values[0]} 
        for col in recent_data.columns
    ]

    return historical_data