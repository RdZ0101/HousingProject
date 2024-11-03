import pandas as pd
import json
from geopy.geocoders import Nominatim
from time import sleep

# Load the dataset
file_path = 'HousingProject/Melbourne_housing_FULL.csv'
df = pd.read_csv(file_path)

# Filter rows with valid SalePrice and group by suburb
df = df.dropna(subset=['Price'])  # Drop rows where Price is NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # Ensure prices are numeric
average_prices = df.groupby('Suburb')['Price'].mean()  # Calculate average price per suburb

# Initialize geolocator
geolocator = Nominatim(user_agent="suburb_locator")
suburb_data = []

# Loop through each suburb and fetch coordinates
for suburb, price in average_prices.items():
    try:
        location = geolocator.geocode(f"{suburb}, Melbourne, Australia")
        if location:
            suburb_data.append({
                "suburb": suburb,
                "average_price": round(price, 2),  # Round the average price to 2 decimal places
                "coordinates": {"lat": location.latitude, "lng": location.longitude}
            })
            print(f"Added {suburb} with coordinates: ({location.latitude}, {location.longitude})")
    except Exception as e:
        print(f"Error fetching data for {suburb}: {e}")
    
    sleep(1)  # To respect the geolocation API rate limit

# Save to JSON
output_path = 'avgSalesSuburbData.json'
with open(output_path, 'w') as f:
    json.dump(suburb_data, f, indent=4)

print(f"Data saved to {output_path}")
