import pandas as pd
import json

# Load the rent CSV file (use raw string for path or replace backslashes with forward slashes)
rent_df = pd.read_csv(r'HousingProject/Rent_allprop.csv')

# Extract columns for suburb and rent values over the years
rent_columns = [col for col in rent_df.columns if "Count" not in col and "Suburb" not in col]
rent_data = rent_df[["Suburb"] + rent_columns]

# Convert rent values to numeric, handle non-numeric values by setting them as NaN
for col in rent_columns:
    rent_data[col] = pd.to_numeric(rent_data[col].replace('[\$,]', '', regex=True).replace('-', 'NaN'), errors='coerce')

# Calculate the highest average rent for each suburb
rent_data['average_rent'] = rent_data[rent_columns].mean(axis=1, skipna=True)
highest_rent_suburbs = rent_data[['Suburb', 'average_rent']]

# Save to JSON
high_rent_json = highest_rent_suburbs.to_dict(orient='records')
with open('avg_rent_summary.json', 'w') as f:
    json.dump(high_rent_json, f)

print("avg-rent suburbs data saved as avg_rent_summary.json")
