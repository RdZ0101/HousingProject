import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
df = pd.read_csv('Melbourne_housing_FULL.csv')

# Define the columns that should be in the final dataset, based on the first dataset
first_dataset_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Date', 'Postcode',
                         'Regionname', 'Propertycount', 'Distance', 'CouncilArea', 'Year', 'Month']

# Step 1: Convert 'Date' column to datetime and extract useful features like year and month
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Step 2: Ensure only the columns from the first dataset are kept
df = df[first_dataset_columns]

# Step 3: Remove records with missing values
df_clean = df.dropna()

# plot the outliers in a boxplot
plt.figure(figsize=(10, 6))
df_clean.boxplot(column=['Price', 'Rooms', 'Distance', 'Propertycount'])
plt.title('Boxplot of Price, Rooms, Distance, and Propertycount before handling outliers')
plt.show()

# Step 4: Handle outliers using the IQR method
def handle_outliers_IQR(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR

    # Replace outliers with whisker values
    df[column] = np.where(df[column] < lower_whisker, lower_whisker, df[column])
    df[column] = np.where(df[column] > upper_whisker, upper_whisker, df[column])

# Apply IQR method to numeric columns
numeric_cols = ['Price', 'Rooms', 'Distance', 'Propertycount']
for col in numeric_cols:
    handle_outliers_IQR(df_clean, col)

# plot the outliers in a boxplot
plt.figure(figsize=(10, 6))
df_clean.boxplot(column=['Price', 'Rooms', 'Distance', 'Propertycount'])
plt.title('Boxplot of Price, Rooms, Distance, and Propertycount after handling outliers')
plt.show()

# Step 5: Normalize numeric features using Min-Max Scaling
scaler = MinMaxScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

# Save the processed data to a new CSV file
output_file = 'processed_housing_data2.csv'
if not os.path.exists(output_file):
    df_clean.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
