from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from RentalPredictions.RentalPrediction import Get_Rental_Prediction, Get_Historical_Rent_Prices, Get_Type_Predictions

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class RentPredictionRequest(BaseModel):
    suburb: str
    numRooms: int
    houseType: str
    rentalPeriod: int

class HistoricalRentPricesRequest(BaseModel):
    suburb: str
    numRooms: int
    houseType: str
    monthsBack: int

class HouseTypeComparisonRequest(BaseModel):
    suburb: str
    numRooms: int
    monthsBack: int
    monthsAhead: int

@app.post("/predict_rent")
async def predict_rent(request: RentPredictionRequest):
    try:
        suburb = request.suburb
        bedrooms = request.numRooms
        property_type = request.houseType
        months_ahead = request.rentalPeriod

        predicted_price = Get_Rental_Prediction(
            suburb=suburb,
            bedrooms=bedrooms,
            property_type=property_type,
            months_ahead=months_ahead
        )
        
        if predicted_price is None:
            raise HTTPException(status_code=404, detail="Prediction could not be made with the given data.")

        return {"predicted_price": round(predicted_price, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/getHistoricalRentPrices")
async def get_historical_rent_prices(request: HistoricalRentPricesRequest):
    try:
        suburb = request.suburb
        numRooms = request.numRooms
        houseType = request.houseType
        monthsBack = request.monthsBack

        historical_data = Get_Historical_Rent_Prices(suburb, numRooms, houseType, monthsBack)
        if not historical_data:
            raise HTTPException(status_code=404, detail="Historical data not found")
        
        return {"historical_data": historical_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for comparing property types
@app.post("/comparePropertyTypes")
async def compare_property_types(request: HouseTypeComparisonRequest):
    try:
        suburb = request.suburb
        bedrooms = request.numRooms
        months_back = request.monthsBack
        months_ahead = request.monthsAhead

        # Call Get_Type_Predictions function
        comparison_data = Get_Type_Predictions(
            suburb=suburb,
            bedrooms=bedrooms,
            months_back=months_back,
            months_ahead=months_ahead
        )
        
        if comparison_data is None:
            raise HTTPException(status_code=404, detail="Comparison data could not be generated with the given input.")
        
        return comparison_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
