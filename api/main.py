import os
import datetime
import sqlite3
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
import pandas as pd

from src.predict import HeartDiseasePredictor
from src import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST API to predict heart disease risk using clinical features.",
    version="1.0.0"
)

# Initialize Predictor
try:
    predictor = HeartDiseasePredictor()
except Exception as e:
    logger.warning(f"Predictor could not be initialized (likely no trained model yet): {e}")
    predictor = None

# SQLite Database Setup
DB_PATH = os.path.join(config.DATA_DIR, "predictions.db")

def init_db():
    """Initialize SQLite database for storing prediction history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age REAL,
            sex REAL,
            cp REAL,
            trestbps REAL,
            chol REAL,
            fbs REAL,
            restecg REAL,
            thalach REAL,
            exang REAL,
            oldpeak REAL,
            slope REAL,
            ca REAL,
            thal REAL,
            predicted_class INTEGER,
            probability REAL,
            risk_status TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize DB on start
init_db()

def log_prediction_to_db(data: dict, prediction_result: dict):
    """Log prediction parameters and outcomes to SQLite DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                exang, oldpeak, slope, ca, thal, predicted_class, probability, risk_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
            data["fbs"], data["restecg"], data["thalach"], data["exang"],
            data["oldpeak"], data["slope"], data["ca"], data["thal"],
            prediction_result["prediction"], prediction_result["probability"],
            prediction_result["risk_status"]
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving prediction to database: {e}")

# Pydantic Input Validator
class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., description="Gender (1 = male; 0 = female)")
    cp: float = Field(..., description="Chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure in mm Hg")
    chol: float = Field(..., ge=50, le=600, description="Serum cholesterol in mg/dl")
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: float = Field(..., description="Resting electrocardiographic results (0 = normal; 1 = ST-T wave abnormality; 2 = left ventricular hypertrophy)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: float = Field(..., description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: float = Field(..., description="The slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)")
    ca: float = Field(..., description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: float = Field(..., description="Thalassemia type (3 = normal; 6 = fixed defect; 7 = reversible defect)")

    @field_validator('sex', 'fbs', 'exang')
    @classmethod
    def validate_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError("Value must be 0 or 1")
        return v

    @field_validator('cp')
    @classmethod
    def validate_cp(cls, v):
        if v not in [1, 2, 3, 4]:
            raise ValueError("Chest pain type must be 1, 2, 3, or 4")
        return v

    @field_validator('restecg')
    @classmethod
    def validate_restecg(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("Resting ECG must be 0, 1, or 2")
        return v

    @field_validator('slope')
    @classmethod
    def validate_slope(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("Slope must be 1, 2, or 3")
        return v

    @field_validator('ca')
    @classmethod
    def validate_ca(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("Number of major vessels (ca) must be 0, 1, 2, or 3")
        return v

    @field_validator('thal')
    @classmethod
    def validate_thal(cls, v):
        if v not in [3, 6, 7]:
            raise ValueError("Thalassemia (thal) must be 3, 6, or 7")
        return v

class BatchPredictionRequest(BaseModel):
    patients: List[PatientFeatures]

# API Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint."""
    status = "healthy"
    model_loaded = predictor is not None
    return {
        "status": status,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_loaded": model_loaded
    }

@app.get("/model-info")
def model_info():
    """Return info about the trained model."""
    global predictor
    # Try re-initializing predictor if it is None (e.g. if trained after API start)
    if predictor is None:
        try:
            predictor = HeartDiseasePredictor()
        except Exception:
            pass
            
    if predictor is None:
        return {
            "model_status": "No model loaded. Please run the training script first.",
        }
    
    return {
        "model_status": "Loaded",
        "model_type": type(predictor.model).__name__,
        "features_expected": predictor.model.n_features_in_ if hasattr(predictor.model, "n_features_in_") else len(config.COLUMNS) - 1,
    }

@app.post("/predict")
def predict(features: PatientFeatures, background_tasks: BackgroundTasks):
    """Predict heart disease risk for a single patient."""
    global predictor
    if predictor is None:
        try:
            predictor = HeartDiseasePredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model is not loaded or trained: {e}")
            
    input_data = features.model_dump()
    try:
        prediction_res = predictor.predict_single(input_data)
        # Log to DB asynchronously in background
        background_tasks.add_task(log_prediction_to_db, input_data, prediction_res)
        return {
            "success": True,
            "result": prediction_res,
            "disclaimer": "This prediction is generated by a machine learning model and should not be considered medical advice."
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Predict heart disease risk for multiple patients."""
    global predictor
    if predictor is None:
        try:
            predictor = HeartDiseasePredictor()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model is not loaded or trained: {e}")
            
    patient_dicts = [p.model_dump() for p in request.patients]
    try:
        results = []
        for p in patient_dicts:
            res = predictor.predict_single(p)
            # Log each prediction to DB
            background_tasks.add_task(log_prediction_to_db, p, res)
            results.append(res)
            
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {e}")
