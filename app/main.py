import os
import sys
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Add project root and src directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict import ChurnPredictor
from app.schemas import (
    CustomerData,
    BatchCustomerData,
    PredictionResult,
    BatchPredictionResult,
    ModelMetadataResponse
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-End ML Prediction Service for Bank Customer Churn Analysis powered by FastAPI & Joblib",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = ChurnPredictor(models_dir=MODELS_DIR)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"ML Model artifact loading error: {str(e)}")
    return predictor

@app.on_event("startup")
def startup_event():
    try:
        get_predictor()
        print("ML Artifacts loaded successfully into API runtime.")
    except Exception as e:
        print(f"Warning: Could not load ML artifacts on startup ({e}). Ensure model is trained.")

@app.get("/health", tags=["Health"])
def health_check():
    p = None
    try:
        p = get_predictor()
    except Exception:
        pass
    return {
        "status": "healthy",
        "service": "Customer Churn Prediction API",
        "model_loaded": p is not None and p.model is not None
    }

@app.get("/api/v1/metadata", response_model=ModelMetadataResponse, tags=["Model Info"])
def get_model_metadata():
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Model metadata not found. Train model first.")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata

@app.post("/api/v1/predict", response_model=PredictionResult, tags=["Predictions"])
def predict_churn(customer: CustomerData):
    pred_instance = get_predictor()
    try:
        customer_dict = customer.model_dump()
        result = pred_instance.predict_single(customer_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/api/v1/predict-batch", response_model=BatchPredictionResult, tags=["Predictions"])
def predict_churn_batch(batch: BatchCustomerData):
    pred_instance = get_predictor()
    try:
        input_list = [c.model_dump() for c in batch.customers]
        df = pd.DataFrame(input_list)
        results = pred_instance.predict_dataframe(df)

        high_risk = sum(1 for r in results if r['risk_level'] == 'High')
        med_risk = sum(1 for r in results if r['risk_level'] == 'Medium')
        low_risk = sum(1 for r in results if r['risk_level'] == 'Low')

        return {
            "total_customers": len(results),
            "high_risk_count": high_risk,
            "medium_risk_count": med_risk,
            "low_risk_count": low_risk,
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
def serve_ui():
    html_file = os.path.join(BASE_DIR, "app", "templates", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Customer Churn Prediction API</h1><p>UI template not found. Visit <a href='/docs'>/docs</a> for API documentation.</p>"
