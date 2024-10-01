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
    
    if numeric_df.empty:
        print("No numeric data available for correlation matrix.")
    else:
        plt.figure(figsize=(14, 10))  # Increase size for better fit
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    #Average price per postcode (Top 20)
    plt.figure(figsize=(16, 8))  
    top_postcodes = dataframe.groupby('Postcode')['Price'].mean().nlargest(20)  # Top 20 postcodes by average price
    top_postcodes.plot(kind='bar')
    plt.title('Average Price for Top 20 Postcodes')
    plt.xlabel('Postcode')
    plt.ylabel('Average Price')
    plt.xticks(rotation=90)
    plt.show()

    # Average price per method (if applicable)
    if 'Method' in dataframe.columns:
        plt.figure(figsize=(12, 6))  # Increase size
        method_avg_price = dataframe.groupby('Method')['Price'].mean()

    # mapping back to the original names
    method_labels = {
        1: 'Property Sold',
        2: 'Property Sold Prior',
        3: 'Property Passed In',
        4: 'Sold Prior Not Disclosed',
        5: 'Sold Not Disclosed',
        6: 'No Bid',
        7: 'Vendor Bid',
        8: 'Withdrawn Prior to Auction',
        9: 'Sold After Auction',
        10: 'Sold After Auction Price Not Disclosed',
        11: 'Price Not Available'
    }
    
    # Ensure the mapping from the numerical codes to descriptive labels is applied
    dataframe['Method'] = dataframe['Method'].map(method_labels)

    # Ensure all method types are included, even if some have no data in the dataset
    all_methods = pd.Series(list(method_labels.values()))

    # Group by the 'Method' column to calculate the average price for each method
    method_avg_price = dataframe.groupby('Method')['Price'].mean()
    
    # Reindex the grouped data so that all methods appear (fill missing with NaN or 0)
    method_avg_price = method_avg_price.reindex(all_methods, fill_value=0)

    # Plot the data
    plt.bar(method_avg_price.index, method_avg_price.values)

    # Set x-axis labels, rotating them to avoid overlap
    plt.xticks(rotation=45, ha='right')

    # Add labels and title
    plt.xlabel('Sale Method')
    plt.ylabel('Average Price')
    plt.title('Average Price per Sale Method')

    # Ensure layout fits well
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
    #top 40 postcodes based on the number of entries
    top_postcodes = dataframe['Postcode'].value_counts().nlargest(40).index

    # Filter the dataframe for these top postcodes
    filtered_dataframe = dataframe[dataframe['Postcode'].isin(top_postcodes)]

    # Boxplot for Price Distribution by Top 40 Postcodes
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Postcode', y='Price', data=filtered_dataframe)
    plt.title('Price Distribution by Top 40 Postcodes')
    plt.xlabel('Postcode')
    plt.ylabel('Price')
    plt.xticks(rotation=90)  # Rotate for readability
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
    
    #VisualizeData(df_clean)
    return df_clean
