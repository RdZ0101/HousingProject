import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


Months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# This is a general purpose data preprocessing function that can be used for any Most datasets
def MixedDataPreprocessing(inputFileName):
    """
    This function processes mixed datasets (e.g., sale price data) by handling missing values, 
    outliers, normalizing numeric features, and creating new features such as price per room, 
    price per square meter, and year-over-year price changes.
    
    Args:
    inputFileName (str): The name of the dataset to be processed.
    
    Returns:
    pd.DataFrame: The processed dataset with new features.
    """
    
    # Load the dataset
    df = pd.read_csv(inputFileName)

    # Define the columns that should be in the final dataset
    first_dataset_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Date', 'Postcode',
                             'Regionname', 'Propertycount', 'Distance', 'CouncilArea', 'Year', 'Month', 'Landsize']

    # Step 1: Convert 'Date' column to datetime and extract useful features like year and month
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Step 2: Ensure only the columns from the first dataset are kept
    df = df[first_dataset_columns]

    # Step 3: Remove records with missing values
    df_clean = df.dropna()

    # Categorize the 'Type' column 
    """ 
    h - House
    u - Unit
    t - Townhouse
    f - Flat
    a - Apartment
    o - Other
    """
    for row in df_clean['Type']:
        if row == 'h':
            df_clean['Type'] = 1
        elif row == 'u':
            df_clean['Type'] = 2
        elif row == 't':
            df_clean['Type'] = 3
        elif row == 'f':
            df_clean['Type'] = 4
        elif row == 'a':
            df_clean['Type'] = 5
        elif row == 'o':
            df_clean['Type'] = 6

    # Categorize the 'Method' column
    """
    Method:
    S - property sold;
    SP - property sold prior;
    PI - property passed in;
    PN - sold prior not disclosed;
    SN - sold not disclosed;
    NB - no bid;
    VB - vendor bid;
    W - withdrawn prior to auction;
    SA - sold after auction;
    SS - sold after auction price not disclosed.
    N/A - price or highest bid not available. 
    """
    for row in df_clean['Method']:
        if row == 'S':
            df_clean['Method'] = 1
        elif row == 'SP':
            df_clean['Method'] = 2
        elif row == 'PI':
            df_clean['Method'] = 3
        elif row == 'PN':
            df_clean['Method'] = 4
        elif row == 'SN':
            df_clean['Method'] = 5
        elif row == 'NB':
            df_clean['Method'] = 6
        elif row == 'VB':
            df_clean['Method'] = 7
        elif row == 'W':
            df_clean['Method'] = 8
        elif row == 'SA':
            df_clean['Method'] = 9
        elif row == 'SS':
            df_clean['Method'] = 10
        elif row == 'N/A':
            df_clean['Method'] = 11

    # Step 4: Create new features
    # Price per Room
    df_clean['Price_per_Room'] = df_clean['Price'] / df_clean['Rooms']

    # Price per Square Meter (if land size is available)
    if 'Landsize' in df_clean.columns and not df_clean['Landsize'].isnull().all():
        df_clean['Price_per_Square_Meter'] = df_clean['Price'] / df_clean['Landsize']
    else:
        df_clean['Price_per_Square_Meter'] = np.nan  # If no Landsize data available

    # Year-over-Year Price Change (group by Suburb and Year)
    df_clean['YoY_Price_Change'] = df_clean.groupby('Suburb')['Price'].pct_change()

    # Step 5: Handle outliers using the IQR method
    def handle_outliers_IQR(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR

        # Replace outliers with whisker values
        df[column] = np.where(df[column] < lower_whisker, lower_whisker, df[column])
        df[column] = np.where(df[column] > upper_whisker, upper_whisker, df[column])

    # Apply IQR method to numeric columns (including newly created ones)
    numeric_cols = ['Price', 'Rooms', 'Distance', 'Propertycount', 'Price_per_Room', 'Price_per_Square_Meter']
    for col in numeric_cols:
        handle_outliers_IQR(df_clean, col)

    # Step 6: Normalize numeric features using Min-Max Scaling
    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    # Step 7: Save the processed data to a new CSV file
    outputFilename = 'processed_' + inputFileName
    if os.path.exists(outputFilename):
        os.remove(outputFilename)
    df_clean.to_csv(outputFilename, index=False)
    print(f'Processed data saved to {outputFilename}')

    #plot the housing price distribution per postcode
    df_clean.groupby('Postcode')['Price'].mean().plot(kind='bar')
    plt.title('Average Price per Postcode')
    plt.show()

    #plot the housing price distribution per suburb
    df_clean.groupby('Suburb')['Price'].mean().plot(kind='bar')
    plt.title('Average Price per Suburb')
    plt.show()

    #plot the housing price distribution per type
    df_clean.groupby('Type')['Price'].mean().plot(kind='bar')
    plt.title('Average Price per Type')
    plt.show()

    #plot a heatmap of the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    
    #plot the distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(df_clean['Price'], kde=True)
    plt.title('Price Distribution')
    plt.show()

    #plot the heatmapt of the most sold properties with regards to distance
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_clean.groupby(['Suburb', 'Type'])['Distance'].count().unstack(), cmap='coolwarm', annot=True, fmt='d')
    plt.title('Most Sold Properties by Type and Distance')
    plt.show()

    # Step 8 : Drop features that will not directly impact the model
    df_clean.drop(['Suburb', 'Address','SellerG', 'Date', 'Regionname', 'Propertycount',  'CouncilArea'], axis=1, inplace=True)
    return df_clean

    



# This works with the datasets that go with the Rent_nBF/BH_Final.csv format
def TimeSeriesPreprocessor(inputFileName):
    """
    This function processes time-series datasets, typically rental price datasets.
    
    Args:
    inputFileName (str): The name of the dataset to be processed.
    
    Returns:
    pd.DataFrame: The processed dataset.
    """
    df = pd.read_csv(inputFileName)
    print(df.head())
    print(f'Original dataset shape: {df.shape}')

    # get rid of null values
    df.dropna()

    #drop duplicates if suburb and postcode both are duplicated
    df.drop_duplicates(subset=['Suburb', 'Postcode'], keep='first', inplace=True)

    # Convert all columns with Month-Year format to datetime with only month and year. the given format is like May 2021 it needs to be 05-2021
    for col in df.columns:
        if col in Months:
            df[col] = pd.to_datetime(df[col], format='%b %Y').dt.strftime('%m-%Y')


    # Save the processed data to a new CSV file. delete the exising file if it exists
    outputFilename = 'processed_' + inputFileName
    if os.path.exists(outputFilename):
        os.remove(outputFilename)
    df.to_csv(outputFilename, index=False)
    print(f'Processed data saved to {outputFilename}')

    
    df_clean = df
    #plot the housing price distribution per postcode
    df_clean.groupby('Postcode')['Price'].mean().plot(kind='bar')
    plt.title('Average Price per Postcode')
    plt.show()
    df_clean.drop(['Suburb', 'Address','SellerG', 'Date', 'Regionname', 'Propertycount',  'CouncilArea'], axis=1, inplace=True)
    return df_clean
