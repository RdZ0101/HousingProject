from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from typing import List


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model = joblib.load("rf.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None

class PredictionInput(BaseModel):
    postcode: str
    rooms: int
    house_type: int 


@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        features = [input_data.postcode, input_data.rooms, 300, input_data.house_type]
        prediction = model.predict([features])[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

class RoomPrediction(BaseModel):
    postcode: str
    house_type: int
    room_counts: List[int]

@app.post("/predict_room_prices")
async def predict_room_prices(request: RoomPrediction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    predictions = {}
    try:
        for rooms in request.room_counts:
            features = [request.postcode, rooms, 300, request.house_type]
            prediction = model.predict([features])[0]
            predictions[rooms] = prediction
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

class PostPrediction(BaseModel):
    postcode: List[str]
    house_type: int
    room_counts: int

@app.post("/predict_postcode_prices")
async def predict_postcode_prices(request: PostPrediction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    predictions = {}
    try:
        for postcode in request.postcode:
            features = [postcode, request.room_counts, 300, request.house_type]
            prediction = model.predict([features])[0]
            predictions[postcode] = prediction
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")