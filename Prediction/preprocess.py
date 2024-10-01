import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

from Visualization import Visualization

# Function to preprocess data
def preprocess_data(file_path, handle_outliers=True, normalize=True):
    # Load the dataset
    data = pd.read_csv(file_path)
    data2 = pd.read_csv(file_path)

    # Drop rows where 'Price' (target variable) is missing
    data_cleaned = data.dropna(subset=['Price'])
    data2_cleaned = data2.dropna(subset=['Price'])

    # Fill missing values for other features with 0
    data_cleaned['Landsize'] = data_cleaned['Landsize'].fillna(0)
    data_cleaned['Rooms'] = data_cleaned['Rooms'].fillna(0)
    data_cleaned['Postcode'] = data_cleaned['Postcode'].fillna(0)

    data2_cleaned['Landsize'] = data2_cleaned['Landsize'].fillna(0)
    data2_cleaned['Rooms'] = data2_cleaned['Rooms'].fillna(0)
    data2_cleaned['Postcode'] = data2_cleaned['Postcode'].fillna(0)

    # Label Encode the 'Type' feature (categorical to numerical)
    label_encoder = LabelEncoder()
    data_cleaned['Type'] = label_encoder.fit_transform(data_cleaned['Type'])
    data2_cleaned['Type'] = label_encoder.fit_transform(data2_cleaned['Type'])

    # Create new derived features
    #data_cleaned['Price_per_Room'] = data_cleaned['Price'] / data_cleaned['Rooms']
    #data_cleaned['Price_per_Landsize'] = data_cleaned['Price'] / data_cleaned['Landsize'].replace(0, 1)
    
    # Fill any NaN values in the new features with 0
    #data_cleaned['Price_per_Room'] = data_cleaned['Price_per_Room'].fillna(0)
    #data_cleaned['Price_per_Landsize'] = data_cleaned['Price_per_Landsize'].fillna(0)

    selected_features = ['Postcode', 'Rooms', 'Landsize', 'Type']#, 'Price_per_Room', 'Price_per_Landsize']

    # Handle outliers if specified
    if handle_outliers:
        def cap_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

        outlier_features = ['Landsize']#, 'Price_per_Room', 'Price_per_Landsize']
        outlier_Price = ['Price']

        # Applying the new function for both features
        for feature in outlier_features:
            cap_outliers(data_cleaned, feature)

        for feature in outlier_Price:
            cap_outliers(data2_cleaned, feature)

    # Normalize data if specified
    if normalize:
        scaler = MinMaxScaler()
        scaled_features = ['Landsize']#, 'Price_per_Room', 'Price_per_Landsize']
        scale_price = ['Price']
        data_cleaned[scaled_features] = scaler.fit_transform(data_cleaned[scaled_features])

    data_cleaned[selected_features] = data_cleaned[selected_features].fillna(0)
    data2_cleaned[selected_features] = data2_cleaned[selected_features].fillna(0)
    def log_transform(df, column):
        df[column + '_log'] = np.log1p(df[column])
        return df

# Add the percentile capping function
    def percentile_capping(df, column, lower_percentile=0.05, upper_percentile=0.95):
        lower_bound = df[column].quantile(lower_percentile)
        upper_bound = df[column].quantile(upper_percentile)

        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        return df
    data2_cleaned = log_transform(data2_cleaned, 'Price')
    data2_cleaned = percentile_capping(data2_cleaned, 'Price')

    #data2_cleaned.to_csv('test.csv')

    #Visualization(data2_cleaned)

    return data_cleaned[selected_features], data_cleaned['Price']

