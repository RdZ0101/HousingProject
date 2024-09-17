import pandas as pd
import numpy as np



with open("data.csv", "r") as f:
    data = pd.read_csv(f)

# Print the first few rows to verify the data
print(data.head())

# Display basic information about the dataset
print(data.describe())
print(data.info())