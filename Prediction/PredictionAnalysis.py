import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load the dataset
file_path = "C:\Study\Semester_2_2024\COS30049\Draft\Melbourne_housing_FULL.csv"
data = pd.read_csv(file_path)

# 1. Drop rows where 'Price' (target variable) is missing
data_cleaned = data.dropna(subset=['Price'])

# 2. Fill missing values for other features with 0
data_cleaned['Landsize'] = data_cleaned['Landsize'].fillna(0)
data_cleaned['Rooms'] = data_cleaned['Rooms'].fillna(0)
data_cleaned['Postcode'] = data_cleaned['Postcode'].fillna(0)

# 3. Label Encode the 'Type' feature (categorical to numerical)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data_cleaned['Type'] = label_encoder.fit_transform(data_cleaned['Type'])

# 4. Create the new derived features
#data_cleaned['Price_per_Room'] = data_cleaned['Price'] / data_cleaned['Rooms']
#data_cleaned['Price_per_Landsize'] = data_cleaned['Price'] / data_cleaned['Landsize'].replace(0, 1)

# Fill any NaN values in the new features with 0
#data_cleaned['Price_per_Room'] = data_cleaned['Price_per_Room'].fillna(0)
#data_cleaned['Price_per_Landsize'] = data_cleaned['Price_per_Landsize'].fillna(0)

# Define the selected features
selected_features = ['Postcode', 'Rooms', 'Landsize', 'Type']#, 'Price_per_Room', 'Price_per_Landsize']
X_rf = data_cleaned[selected_features].fillna(0)
y_rf = data_cleaned['Price']

# Split the data for Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Train Random Forest (no outlier handling or normalization)
rf_model_final = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_final.fit(X_train_rf, y_train_rf)
y_pred_rf_final = rf_model_final.predict(X_test_rf)
mae_rf_final = mean_absolute_error(y_test_rf, y_pred_rf_final)
mse_rf_final = mean_squared_error(y_test_rf, y_pred_rf_final)
rmse_rf_final = mse_rf_final ** 0.5
r2_rf_final = r2_score(y_test_rf, y_pred_rf_final)

# Preprocess for Linear Regression and KNN (handle outliers and normalize)
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

outlier_features = ['Landsize']#, 'Price_per_Room', 'Price_per_Landsize']

# Copy the dataset to preprocess for Linear Regression and KNN
data_preprocessed = data_cleaned.copy()

# Handle outliers for the selected features
for feature in outlier_features:
    cap_outliers(data_preprocessed, feature)

# Normalize the features that are sensitive to scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = ['Landsize']#, 'Price_per_Room', 'Price_per_Landsize']
data_preprocessed[scaled_features] = scaler.fit_transform(data_preprocessed[scaled_features])

# Prepare the data for Linear Regression and KNN
X_preprocessed = data_preprocessed[selected_features]
y_preprocessed = data_preprocessed['Price']

# Ensure there are no NaN values by filling them with 0 for all models
X_rf = X_rf.fillna(0)
X_preprocessed = X_preprocessed.fillna(0)

# Split the data for Linear Regression and KNN
X_train_preprocessed, X_test_preprocessed, y_train_preprocessed, y_test_preprocessed = train_test_split(X_preprocessed, y_preprocessed, test_size=0.2, random_state=42)

# Train Linear Regression (with outlier handling and normalization)
linear_model_final = LinearRegression()
linear_model_final.fit(X_train_preprocessed, y_train_preprocessed)
y_pred_linear_final = linear_model_final.predict(X_test_preprocessed)
mae_linear_final = mean_absolute_error(y_test_preprocessed, y_pred_linear_final)
mse_linear_final = mean_squared_error(y_test_preprocessed, y_pred_linear_final)
rmse_linear_final = mse_linear_final ** 0.5
r2_linear_final = r2_score(y_test_preprocessed, y_pred_linear_final)

# Train KNN (with outlier handling and normalization)
knn_model_final = KNeighborsRegressor(n_neighbors=5)
knn_model_final.fit(X_train_preprocessed, y_train_preprocessed)
y_pred_knn_final = knn_model_final.predict(X_test_preprocessed)
mae_knn_final = mean_absolute_error(y_test_preprocessed, y_pred_knn_final)
mse_knn_final = mean_squared_error(y_test_preprocessed, y_pred_knn_final)
rmse_knn_final = mse_knn_final ** 0.5
r2_knn_final = r2_score(y_test_preprocessed, y_pred_knn_final)

# Prepare data for visualization
models = ['Random Forest', 'Linear Regression', 'KNN Regression']
mae_values = [mae_rf_final, mae_linear_final, mae_knn_final]
rmse_values = [rmse_rf_final, rmse_linear_final, rmse_knn_final]
r2_values = [r2_rf_final, r2_linear_final, r2_knn_final]

# Create a bar chart for MAE comparison
plt.figure(figsize=(10, 6))
plt.bar(models, mae_values, color=['green', 'blue', 'orange'])
plt.title('MAE (Mean Absolute Error) Comparison of Models')
plt.ylabel('MAE')
plt.show()

# Create a bar chart for RMSE comparison
plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['green', 'blue', 'orange'])
plt.title('RMSE (Root Mean Squared Error) Comparison of Models')
plt.ylabel('RMSE')
plt.show()

# Create a bar chart for R² comparison
plt.figure(figsize=(10, 6))
plt.bar(models, r2_values, color=['green', 'blue', 'orange'])
plt.title('R^2 Comparison of Models')
plt.ylabel('R^2 Score')
plt.ylim(0, 1)  # Set y-axis to a fixed range from 0 to 1 for better R^2 visualization
plt.show()




joblib.dump(rf_model_final, 'rf.pkl')
joblib.dump(knn_model_final, 'knn.pkl')
joblib.dump(linear_model_final, 'linear.pkl')