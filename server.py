from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Allow CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML models
knn_model = joblib.load('knn_model.joblib')
linear_model = joblib.load('linear_model.joblib')

@app.post('/predict', response_class=JSONResponse)
def predict(date_str: str = Form(...)):
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    year, month, day = date.year, date.month, date.day
    features = np.array([[year, month, day]])
    cutoff_date = pd.to_datetime('2024-04-01')

    predicted_value = (
        knn_model.predict(features)[0]
        if date < cutoff_date
        else linear_model.predict(features)[0]
    )
    return {'predicted': predicted_value}

# Serve HTML file if needed
@app.get('/')
def root():
    return {"message": "FastAPI server is running!"}
