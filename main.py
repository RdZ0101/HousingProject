import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from scikeras.wrappers import KerasRegressor  # Use SciKeras instead of tensorflow.keras.wrappers
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

# Function to train the LSTM model with cross-validation and grid search
def LSTM_TimeSeries_Model(df, seq_length=12, n_splits=10):
    # Extract the price and other necessary features
    data = df[['Price']].values
    
    # Normalize the price data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    X, y = create_sequences(data_scaled, seq_length)
    
    # Train-test split (20% test)
    global X_train, X_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model using KerasRegressor for hyperparameter tuning
    model = KerasRegressor(build_fn=build_lstm_model, verbose=0)
    
    # Define hyperparameters grid for tuning
    param_grid = {
        'units': [50, 100],
        'dropout_rate': [0.2, 0.3],
        'batch_size': [16, 32],
        'epochs': [50, 100]
    }
    
    # 10-fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Grid search for hyperparameter tuning
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train, y_train)
    
    # Print the best parameters and the corresponding score
    print(f"Best Params: {grid_result.best_params_}")
    print(f"Best MSE Score: {-grid_result.best_score_}")

    # Train the best model on the entire training set
    best_model = grid_result.best_estimator_.model
    best_model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'], verbose=1)
    
    # Predict on the test set
    y_pred = best_model.predict(X_test)
    
    # Inverse scale the predictions
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate test MSE
    test_mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    print(f"Test Mean Squared Error: {test_mse}")
    
    return best_model, grid_result.best_params_, scaler

# Function to train and return a Linear Regression model
def LinearRegressionModel(df):
    """
    This function trains a linear regression model on the given dataset and returns the trained model.
    """
    # Define the features and target variable
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Linear Regression Mean Squared Error: {mse}')
    
    return model

# Function to make predictions for a specific suburb (for both models)
def predict_price_for_suburb(df, suburb_name, model, scaler=None, seq_length=12, model_type='lstm'):
    suburb_data = df[df['Suburb'] == suburb_name][['Price']].values
    
    if len(suburb_data) < seq_length and model_type == 'lstm':
        print("Not enough data for this suburb.")
        return None
    
    if model_type == 'lstm':
        # Normalize the price data for LSTM
        suburb_data_scaled = scaler.transform(suburb_data)
        X_suburb, _ = create_sequences(suburb_data_scaled, seq_length)
        predicted_price_scaled = model.predict(X_suburb[-1].reshape(1, seq_length, 1))
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        return predicted_price[0][0]
    
    elif model_type == 'linear':
        suburb_data_encoded = df.drop(['Price'], axis=1)
        suburb_price_prediction = model.predict(suburb_data_encoded)
        return suburb_price_prediction

# Load the dataset
dataset = input("Enter the name of the dataset file (CSV format): ")
df = pd.read_csv(dataset)

# Preprocess the dataset using Preprocessor.py functions
print("Dataset types 1. all 2. single time series")
type = int(input("Enter the type of dataset: "))
if type == 1:
    df = preprocessor.MixedDataPreprocessing(dataset)
elif type == 2:
    df = preprocessor.TimeSeriesPreprocessor(dataset)

# Ask the user to choose a model
mChoice = input("Select a model to train: \n1. Linear Regression\n2. LSTM\n")

if mChoice == '1':
    model = LinearRegressionModel(df)
    model_type = 'linear'
elif mChoice == '2':
    model, best_params, scaler = LSTM_TimeSeries_Model(df)
    model_type = 'lstm'

# Predict price for a specific suburb
while True:
    suburb_name = input("Enter the suburb name for prediction (or 'exit' to stop): ").strip()
    if suburb_name.lower() == 'exit':
        break
    
    predicted_price = predict_price_for_suburb(df, suburb_name, model, scaler if model_type == 'lstm' else None, model_type=model_type)
    
    if predicted_price:
        print(f"Predicted price for next month in {suburb_name}: ${predicted_price:.2f}")
    else:
        print(f"Could not predict price for suburb: {suburb_name}")
