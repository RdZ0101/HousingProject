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

    # Correlation heatmap (use only numeric columns)
    
    #function to load and process individual CSVs
    def load_and_process_rent_data(file_path, rooms, housing_type):
        with open(file_path, 'r') as file:
            df = pd.read_csv(file)
    
            # Assign rooms and type (1 for House, 2 for Flat)
            df['Rooms'] = rooms
            df['Type'] = housing_type  # 1 for House, 2 for Flat
        
            # Extract columns that represent rental prices (assuming MM-YYYY format)
            date_columns = df.columns.difference(['Suburb', 'Postcode', 'Rooms', 'Type'])
        
            # Convert the date columns to numeric (coerce errors to NaN)
            df[date_columns] = df[date_columns].apply(pd.to_numeric, errors='coerce')
        
            # Calculate the average rent over the years, ignoring NaN values
            df['Average_Rent'] = df[date_columns].mean(axis=1)
        
            # Return relevant columns for correlation analysis
            return df[['Rooms', 'Type', 'Average_Rent']]

            # Load and process all datasets
    df_1bf = load_and_process_rent_data('processed_Rent_1BF_Final.csv', 1, 2)  # 1 Bedroom Flat
    df_2bf = load_and_process_rent_data('processed_Rent_2BF_Final.csv', 2, 2)  # 2 Bedroom Flat
    df_3bf = load_and_process_rent_data('processed_Rent_3BF_Final.csv', 3, 2)  # 3 Bedroom Flat
    df_2bh = load_and_process_rent_data('processed_Rent_2BH_Final.csv', 2, 1)  # 2 Bedroom House
    df_3bh = load_and_process_rent_data('processed_Rent_3BH_Final.csv', 3, 1)  # 3 Bedroom House

    # Concatenate all datasets into one dataframe
    combined_df = pd.concat([df_1bf, df_2bf, df_3bf, df_2bh, df_3bh], ignore_index=True)

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(combined_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix with Rooms, Type, and Average Rent')
    plt.show()

    #Average price per postcode (Top 20)
    # Ensure the rent columns are the date columns representing prices
    date_columns = dataframe.columns.difference(['Suburb', 'Postcode'])

    # Convert non-numeric values to NaN for the date columns to handle them gracefully
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_numeric, errors='coerce')

    # Calculate the average price over time for each postcode, ignoring NaN values
    dataframe['Average_Price'] = dataframe[date_columns].mean(axis=1)

    # Grouping by 'Postcode' and calculating the average price
    plt.figure(figsize=(16, 8))  
    top_postcodes = dataframe.groupby('Postcode')['Average_Price'].mean().nlargest(20)  # Top 20 postcodes by average price
    top_postcodes.plot(kind='bar')
    plt.title('Average Price for Top 20 Postcodes')
    plt.xlabel('Postcode')
    plt.ylabel('Average Price')
    plt.xticks(rotation=90)
    plt.show()

    
    #top 40 postcodes based on the number of entries
    top_postcodes = dataframe['Postcode'].value_counts().nlargest(40).index

    # Filter the dataframe for these top postcodes
    filtered_dataframe = dataframe[dataframe['Postcode'].isin(top_postcodes)]

    # Boxplot for Average Price Distribution by Top 40 Postcodes
    # Top 40 postcodes based on the number of entries
    top_postcodes = dataframe['Postcode'].value_counts().nlargest(40).index

    # Filter the dataframe for these top postcodes
    filtered_dataframe = dataframe[dataframe['Postcode'].isin(top_postcodes)]
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Postcode', y='Average_Price', data=filtered_dataframe)
    plt.title('Average Price Distribution by Top 40 Postcodes')
    plt.xlabel('Postcode')
    plt.ylabel('Average Price')
    plt.xticks(rotation=90) 
    plt.show()




# This works with the datasets that go with the Rent_nBF/BH_Final.csv format these datasets are having quarterly prices
def TimeSeriesPreprocessor(inputFileName):
    """
    This function processes time-series datasets, typically rental price datasets.
    
    Args:
    inputFileName (str): The name of the dataset to be processed.
    
    Returns:
    pd.DataFrame: The processed dataset.
    """
    # Load the dataset
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
    
    
    outputFilename = 'processed_' + inputFileName
    if os.path.exists(outputFilename):
        os.remove(outputFilename)
    df_clean.to_csv(outputFilename, index=False)
    print(f'Processed data saved to {outputFilename}')
    #VisualizeData(df_clean)
    return df_clean
