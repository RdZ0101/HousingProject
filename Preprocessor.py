import pandas as pd
import numpy as np

# create a function to do the below for each dataset in the folder


with open("Melbourne_housing_FULL.csv", "r") as f:
    data = pd.read_csv(f)

# Print the first few rows to verify the data
print(data.head())

# Display basic information about the dataset
print(data.describe())
print(data.info())

# Select the following columns and save it to another Excel if not exists 
# ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 'Regionname', 'Propertycount']
data = data[['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 'Regionname', 'Propertycount']]
data.to_csv("Melbourne_housing_FULL.csv", index=False)
