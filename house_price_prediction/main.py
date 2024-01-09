from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Configure CORS settings
origins = [
    "http://localhost",
    "http://localhost:5500",  # Add the URL where your HTML file is served
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use "*" for any origin during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your linear regression model from file
model = joblib.load('mymlmodel.pkl')

class Item(BaseModel):
    features: list

@app.post("/predict", response_model=dict)
async def predict(item: Item):
    try:
        # Ensure features are converted to float before prediction
        features = [float(feature) for feature in item.features]
        
        # Make prediction using the loaded model
        prediction = model.predict([features])
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optionally, include an endpoint for health checks
@app.get("/health")
async def health():
    return {"status": "ok"}
