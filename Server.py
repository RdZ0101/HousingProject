from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from RentalPredictions.RentalPrediction import Get_Rental_Prediction

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model for the POST request
class RentPredictionRequest(BaseModel):
    suburb: str
    numRooms: int
    houseType: str
    rentalPeriod: int

# FastAPI endpoint for rent prediction
@app.post("/predict_rent")
async def predict_rent(request: RentPredictionRequest):
    try:
        suburb = request.suburb
        bedrooms = request.numRooms
        property_type = request.houseType
        months_ahead = request.rentalPeriod

        # Call the Get_Rental_Prediction function with provided data
        predicted_price = Get_Rental_Prediction(
            suburb=suburb,
            bedrooms=bedrooms,
            property_type=property_type,
            months_ahead=months_ahead
        )
        
        if predicted_price is None:
            raise HTTPException(status_code=404, detail="Prediction could not be made with the given data.")

        # Return the predicted rent price as a JSON response, rounded to 2 decimal places
        return {"predicted_price": round(predicted_price, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
