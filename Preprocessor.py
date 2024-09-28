import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def VisualizeData(dataframe):
    """
    Visualizes the data using a pairplot, a heatmap of the correlation matrix, 
    and various bar charts for price analysis with increased font size and adjusted plot sizes.
    """
    # Set global font size for all plots
    plt.rcParams.update({'font.size': 14})

    # Convert categorical columns to numeric if necessary or drop them for the correlation matrix
    numeric_df = dataframe.select_dtypes(include=['float64', 'int64'])

    # Pairplot of the dataset
    sns.pairplot(dataframe, height=3)  # Increase plot size by adjusting height
    plt.show()

    # Correlation heatmap (use only numeric columns)
    plt.figure(figsize=(14, 10))  # Increase size for better fit
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Average price per postcode
    plt.figure(figsize=(16, 8))  # Increase width and height for better readability
    dataframe.groupby('Postcode')['Price'].mean().plot(kind='bar')
    plt.title('Average Price per Postcode')
    plt.xlabel('Postcode')
    plt.ylabel('Average Price')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.show()

    # Average price per method (if applicable)
    if 'Method' in dataframe.columns:
        plt.figure(figsize=(12, 6))  # Increase size
        dataframe.groupby('Method')['Price'].mean().plot(kind='bar')
        plt.title('Average Price per Method')
        plt.xlabel('Method')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)  # Rotate x-axis labels slightly
        plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=dataframe['Rooms'], y=dataframe['Price'], hue=dataframe['Postcode'], palette='coolwarm')
    plt.title('Price vs Number of Rooms, Colored by Postcode')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price')
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Postcode', y='Price', data=dataframe)
    plt.title('Price Distribution by Postcode')
    plt.xlabel('Postcode')
    plt.ylabel('Price')
    plt.xticks(rotation=90)  # Rotate for readability
    plt.show()

    sns.pairplot(dataframe[['Rooms', 'Postcode', 'Price']], diag_kind='kde', height=3)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Rooms', y='Price', data=dataframe)
    plt.title('Price Distribution by Number of Rooms')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price')
    plt.show()


    plt.figure(figsize=(16, 8))
    sns.barplot(x='Postcode', y='Price', hue='Rooms', data=dataframe)
    plt.title('Average Price by Postcode and Number of Rooms')
    plt.xlabel('Postcode')
    plt.ylabel('Average Price')
    plt.xticks(rotation=90)
    plt.show()




def MixedDataPreprocessing(inputFileName):
    """
    This function processes mixed datasets (e.g., sale price data) by handling missing values, 
    outliers, normalizing numeric features (except Price), and creating new features such as 
    price per room, price per square meter, and year-over-year price changes.
    
    Args:
    inputFileName (str): The name of the dataset to be processed.
    
    Returns:
    pd.DataFrame: The processed dataset with new features.
    """
    
    # Load the dataset
    df = pd.read_csv(inputFileName)
    #fix this to pass if the column is not present
    # Define the columns that should be in the final dataset
    first_dataset_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Date', 'Postcode',
                             'Regionname', 'Propertycount', 'Distance', 'CouncilArea', 'Year', 'Month', 'Landsize']
    
    # Handle missing values
    df['Postcode'] = df['Postcode'].fillna(0)  # or drop missing rows: df.dropna(subset=['Postcode'], inplace=True)

    # Convert to integer
    df['Postcode'] = df['Postcode'].astype(int)

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
    numeric_cols = ['Rooms', 'Distance', 'Propertycount', 'Price_per_Room', 'Price_per_Square_Meter']
    for col in numeric_cols:
        handle_outliers_IQR(df_clean, col)

    print(df_clean.head())

    # Step 6: Normalize numeric features using Min-Max Scaling (exclude Price)
    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    # Step 7: Save the processed data to a new CSV file
    outputFilename = 'processed_' + inputFileName
    if os.path.exists(outputFilename):
        os.remove(outputFilename)
    df_clean.to_csv(outputFilename, index=False)
    print(f'Processed data saved to {outputFilename}')
    
    VisualizeData(df_clean)
    return df_clean

    

# This works with the datasets that go with the Rent_nBF/BH_Final.csv format these datasets are having quarterly prices
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
    # Handle missing values
    df['Postcode'] = df['Postcode'].fillna(0)  # or drop missing rows: df.dropna(subset=['Postcode'], inplace=True)

    # Convert to integer
    df['Postcode'] = df['Postcode'].astype(int)

    non_date_cols = ['Suburb', 'Postcode']
    # Extract the columns that need to be converted to date format
    date_cols = df.columns.difference(non_date_cols)

    # Convert the date column names to 'mm-yyyy' format
    new_date_cols = [pd.to_datetime(col, format='%b %Y').strftime('%m-%Y') for col in date_cols]

    # Create a mapping of old to new column names
    column_mapping = dict(zip(date_cols, new_date_cols))

    # Rename the columns in the DataFrame
    df.rename(columns=column_mapping, inplace=True)

    # Display the updated DataFrame
    df.head()

    # Save the processed data to a new CSV file. delete the exising file if it exists
    outputFilename = 'processed_' + inputFileName
    if os.path.exists(outputFilename):
        os.remove(outputFilename)
    df.to_csv(outputFilename, index=False)
    print(f'Processed data saved to {outputFilename}')

    
    df_clean = df
    
    if 'Suburb' in df_clean.columns:
        df_clean.drop(['Suburb'], axis=1, inplace=True)
    if 'Address' in df_clean.columns:
        df_clean.drop(['Address'], axis=1, inplace=True)
    if 'SellerG' in df_clean.columns:
        df_clean.drop(['SellerG'], axis=1, inplace=True)
    if 'Regionname' in df_clean.columns:
        df_clean.drop(['Regionname'], axis=1, inplace=True)
    if 'Propertycount' in df_clean.columns:
        df_clean.drop(['Propertycount'], axis=1, inplace=True)
    if 'CouncilArea' in df_clean.columns:
        df_clean.drop(['CouncilArea'], axis=1, inplace=True)
    
    print(df_clean.head())
    VisualizeData(df_clean)
    return df_clean
