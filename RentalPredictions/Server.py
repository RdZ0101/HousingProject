from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from RentalPrediction import Get_Rental_Prediction


app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class RentPredictionRequest(BaseModel):
    postcode: int
    bedrooms: int
    property_type: str
    months_ahead: int

# FastAPI endpoint for rent prediction
@app.post("/predict_rent")
async def predict_rent(request: RentPredictionRequest):
    try:
        # Call the Get_Rental_Prediction function with provided data
        predicted_price = Get_Rental_Prediction(
            postcode=request.postcode,
            bedrooms=request.bedrooms,
            property_type=request.property_type,
            months_ahead=request.months_ahead
        )
        
        if predicted_price is None:
            raise HTTPException(status_code=404, detail="Prediction could not be made with the given data.")

        # Return the predicted rent price as a JSON response
        return {"predicted_price": predicted_price}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
