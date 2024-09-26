import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import Preprocessor as preprocessor

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

# Function to build the LSTM model
def build_lstm_model(units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Predicting price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the LSTM model with cross-validation
def LSTM_TimeSeries_Model(df, seq_length=12):
    data = df[['Price']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = create_sequences(data_scaled, seq_length)
    
    global X_train, X_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    test_mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    print(f"Test Mean Squared Error: {test_mse}")
    
    return model, scaler

# Function to train and return a Linear Regression model
def LinearRegressionModel(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Linear Regression Mean Squared Error: {mse}')
    
    return model

# Main prediction function
def predict_price_for_postcode(df, postcode, months, model, scaler=None, model_type='lstm'):
    postcode_data = df[df['Postcode'] == postcode][['Price']].values
    if len(postcode_data) < months:
        print("Not enough data for this postcode.")
        return None

    if model_type == 'lstm':
        postcode_data_scaled = scaler.transform(postcode_data)
        X_postcode, _ = create_sequences(postcode_data_scaled, months)
        predicted_price_scaled = model.predict(X_postcode[-1].reshape(1, months, 1))
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        return predicted_price[0][0]

    elif model_type == 'linear':
        X_postcode = df[df['Postcode'] == postcode].drop(['Price'], axis=1)
        predicted_price = model.predict(X_postcode)
        return predicted_price

# Load the dataset
dataset = input("Enter the name of the dataset file (CSV format): ")
df = pd.read_csv(dataset)

# Preprocess the dataset using Preprocessor.py functions
df = preprocessor.MixedDataPreprocessing(dataset)

# Ask the user for input
postcode = int(input("Enter the postcode for prediction: "))
method = input("Press 1 to get rent prediction, 2 to get Sale price prediction: ")
months = int(input("Enter the time period for prediction (in months): "))
model_choice = input("Select a model to train: \n1. Linear Regression\n2. LSTM\n")

if model_choice == '1':
    model = LinearRegressionModel(df)
    model_type = 'linear'
elif model_choice == '2':
    model, scaler = LSTM_TimeSeries_Model(df, seq_length=months)
    model_type = 'lstm'

predicted_price = predict_price_for_postcode(df, postcode, months, model, scaler if model_type == 'lstm' else None, model_type=model_type)

if predicted_price:
    print(f"Predicted price for postcode {postcode} after {months} months: ${predicted_price:.2f}")
else:
    print("Prediction could not be made.")
