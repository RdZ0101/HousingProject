import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import preprocess

# 1. Average Price vs. Number of Rooms
def Visualization(file):
    data_cleaned= file
    avg_price_vs_rooms = data_cleaned.groupby('Rooms')['Price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Rooms', y='Price', data=avg_price_vs_rooms)
    plt.title('Average Price vs Number of Rooms')
    plt.ylabel('Average Price')
    plt.xlabel('Number of Rooms')
    plt.show()

    # 2. Price Distribution Across Top Property Types
    top_types = data_cleaned['Type'].value_counts().nlargest(3).index  # Get the top 3 property types
    price_distribution_top_types = data_cleaned[data_cleaned['Type'].isin(top_types)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Price_log', data=price_distribution_top_types)
    plt.title('Price Distribution Across Top Property Types')
    plt.ylabel('Price_log')
    plt.xlabel('Property Type')
    plt.show()

    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data_cleaned[['Price', 'Rooms', 'Postcode', 'Landsize', 'Type']].corr()#, 'Price_per_Room', 'Price_per_Landsize']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Features')
    plt.show()
