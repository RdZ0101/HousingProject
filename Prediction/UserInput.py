import joblib
import numpy as np

# Load the trained model
model = joblib.load('rf.pkl')
model2 = joblib.load('knn.pkl')
model3 = joblib.load('linear.pkl')

def encode_property_type(property_type):
    type_mapping = {
        'house': 0,
        'unit': 1,
        'townhouse': 2
    }
    return type_mapping.get(property_type.lower(), -1)  # Return -1 if the input is invalid

# Function to predict price based on user inputs
def predict_price(postcode, rooms, property_type, modelt, landsize=None):#, price_per_room=None, price_per_landsize=None):
    # Encode the property type
    encoded_property_type = encode_property_type(property_type)
    if encoded_property_type == -1:
        return "Invalid property type. Please enter 'House', 'Unit', or 'Townhouse'."
    if modelt>3 or modelt<1:
        return "Wrong input for model type"
    
    # Use provided values or fallback to estimated/default values
    landsize = landsize if landsize is not None else 300  # Default/estimated landsize
    #price_per_room = price_per_room if price_per_room is not None else 150000 / rooms if rooms > 0 else 150000
    #price_per_landsize = price_per_landsize if price_per_landsize is not None else 750  # Default estimate for price per landsize
    
    # Create input array with provided and estimated/default values
    user_input = [
        postcode,
        rooms,
        landsize,
        encoded_property_type,
        #price_per_room,
        #price_per_landsize
    ]
    
    # Reshape the input and predict
    input_data = np.array(user_input).reshape(1, -1)
    if modelt ==1:
        predicted_price = model3.predict(input_data)
    elif modelt==2:
        predicted_price = model.predict(input_data)
    else:
        predicted_price = model2.predict(input_data)

    return predicted_price[0]

# Collect user inputs
def StartSP():
    postcode = int(input("Enter postcode: "))
    rooms = int(input("Enter number of rooms: "))
    property_type = input("Enter property type (House, Unit, Townhouse): ")

    # Optionally collect flexible inputs
    landsize = input("Enter landsize (or press Enter to use default): ")
    landsize = float(landsize) if landsize else None
    modelt = int(input("Which model do you want to use? (1 for Linear Regression, 2 for Random Forest, 3 for KNN): "))

    #price_per_room = input("Enter price per room (or press Enter to use default): ")
    #price_per_room = float(price_per_room) if price_per_room else None

    #price_per_landsize = input("Enter price per landsize (or press Enter to use default): ")
    #price_per_landsize = float(price_per_landsize) if price_per_landsize else None

    # Predict the price
    predicted_price = predict_price(postcode, rooms, property_type, modelt, landsize)#, price_per_room, price_per_landsize)
    print(f"The predicted house price is: ${predicted_price}")
