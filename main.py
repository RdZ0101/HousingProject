from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import Preprocessor

# Function for Linear Regression model
def LinearRegressionModel(df, input_data):
    # Include 4 features: 'Postcode', 'Rooms', 'Type', 'Months'
    X = df[['Postcode', 'Rooms', 'Type', 'Months']]  
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 70-30 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    
    # 10-Fold Cross-Validation
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

    print(f"Linear Regression Cross-Validation MSE: {-cv_results.mean()}")

    # Train and evaluate the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Linear Regression MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}')

    # Predicting the price for the user's input data (now includes 4 features: Postcode, Rooms, Type, Months)
    input_data_scaled = scaler.transform([input_data])
    predicted_price = model.predict(input_data_scaled)
    print(f"Predicted Price: {predicted_price[0]}")
    return model


# Function to perform hyperparameter tuning with cross-validation for RandomForest
def RandomForestTunedModel(df, input_data):
    X = df[['Postcode', 'Rooms', 'Type', 'Months']]  # Include 4 features
    y = df['Price']

    # Perform a 70-30 train-test split without scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Hyperparameters for RandomForest (simplified for testing)
    param_grid = {
        'n_estimators': [50, 100],  
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')  
    grid_search.fit(X_train, y_train)

    # Best parameters from the grid search
    print("Best parameters:", grid_search.best_params_)

    # Train the best model
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    
    y_pred = best_rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Random Forest (Tuned) MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}')

    # Predicting the price for the user's input data (now includes 4 features: Postcode, Rooms, Type, Months)
    predicted_price = best_rf.predict([input_data])
    print(f"Predicted Price: {predicted_price[0]}")
    return best_rf


# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)


# Function to perform LSTM tuning and prediction
def LSTM_TimeSeries_Model_Tuned(df, input_data, seq_length=12):
    data = df[['Price']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, seq_length)
    
    # 70-30 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build and train LSTM model with hyperparameter tuning
    def build_lstm_model(units=50, dropout_rate=0.2):
        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))  # Predicting price
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    print(f'LSTM MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}')
    
    # Predict the price for the user's input data
    historical_prices = df[df['Postcode'] == input_data[0]]['Price'].values[-seq_length:]
    
    if len(historical_prices) < seq_length:
        print(f"Not enough historical data for postcode {input_data[0]} to create a sequence.")
        return model

    input_sequence_scaled = scaler.transform(historical_prices.reshape(-1, 1))
    lstm_pred = model.predict(input_sequence_scaled.reshape(1, seq_length, 1))
    predicted_price = scaler.inverse_transform(lstm_pred)
    print(f"Predicted Price (LSTM): {predicted_price[0][0]}")
    
    return model


# Main function to run the selected model based on user input
def run_models(df, model_choice, postcode, rooms, housing_type, months):
    # input_data should have 4 features: postcode, rooms, housing_type, months
    input_data = [postcode, rooms, housing_type, months]

    # Ensure the 'Months' column is in the dataframe before training
    df['Months'] = months

    if model_choice == '1':
        model = LinearRegressionModel(df, input_data)
    elif model_choice == '2':
        model = RandomForestTunedModel(df, input_data)
    elif model_choice == '3':
        model = LSTM_TimeSeries_Model_Tuned(df, input_data, seq_length=months)
    else:
        print("Invalid model choice.")
        return
    
    print(f'Predictions completed for postcode {postcode}, {rooms} rooms, type {housing_type}, for {months} months.')


# Main execution starts here

# Ask for user input
user_input = input("\nEnter postcode, number of rooms, Type of housing(h=1,u=2,t=3,f=4), prediction period (months) separated by commas: ").strip()

# Split the user input correctly into four variables
postcode, rooms, housing_type, months = user_input.split(',')

# Convert the variables to appropriate types
postcode = str(postcode.strip())   # Postcode as a string
rooms = int(rooms.strip())         # Rooms as an integer
housing_type = str(housing_type.strip())  # Housing type as a string
months = int(months.strip())       # Months as an integer

# Ask for Sale or Rental
sale_or_rental = input("\nPlease enter 1 for Sale, 2 for Rental: ").strip()

# Choose dataset based on Sale or Rental choice and preprocess
if sale_or_rental == '1':
    dataset_path = 'Melbourne_housing_FULL.csv'
    df = Preprocessor.MixedDataPreprocessing(dataset_path)
elif sale_or_rental == '2':
    if rooms == 1 and housing_type == '4':
        dataset_path = 'Rent_1BF_Final.csv'
    elif rooms == 2 and housing_type == '4':
        dataset_path = 'Rent_2BF_Final.csv'
    elif rooms == 3 and housing_type == '4':
        dataset_path = 'Rent_3BF_Final.csv'
    elif rooms == 2 and housing_type == '1':
        dataset_path = 'Rent_2BH_Final.csv'
    elif rooms == 3 and housing_type == '1':
        dataset_path = 'Rent_3BH_Final.csv'
    df = Preprocessor.TimeSeriesPreprocessor(dataset_path)

# Add the 'Months' column to the dataframe to match the expected input for all models
df['Months'] = months

# Ask the user to choose the model
print("\nChoose the model for prediction:")
model_choice = input("Enter 1 for Linear Regression, 2 for Random Forest, 3 for LSTM: ").strip()

# Run the model based on the user's choice
run_models(df, model_choice, postcode, rooms, housing_type, months)
