import os
import joblib
import pandas as pd
import numpy as np
from src import config
from src.feature_engineering import engineer_features

# Feature order expected by the model
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "hr_max_age_ratio", "bp_chol_interaction", "double_product", "age_group"
]

class HeartDiseasePredictor:
    def __init__(self):
        # Load artifacts
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError("Model file not found. Please train the model first.")
        
        self.model = joblib.load(config.MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.imputer = joblib.load(config.IMPUTER_PATH)

    def preprocess_input(self, raw_input: dict) -> pd.DataFrame:
        """Preprocess a single dictionary of input parameters."""
        df = pd.DataFrame([raw_input])
        
        # Ensure the 13 raw feature columns are present (target is excluded)
        raw_cols = [c for c in config.COLUMNS if c != config.TARGET_COL]
        for col in raw_cols:
            if col not in df.columns:
                raise ValueError(f"Missing input column: {col}")
        
        # Order raw columns correctly
        df = df[raw_cols]
        
        # Impute missing values for ca and thal if any NaN exists
        df[['ca', 'thal']] = self.imputer.transform(df[['ca', 'thal']])
        
        # Engineer features
        df = engineer_features(df)
        
        # Scale numerical features
        df[config.NUMERICAL_COLS] = self.scaler.transform(df[config.NUMERICAL_COLS])
        
        # Reorder to the exact structure the model expects
        df = df[FEATURE_ORDER]
        return df

    def predict_single(self, raw_input: dict) -> dict:
        """Predict for a single input dictionary."""
        processed_df = self.preprocess_input(raw_input)
        
        # Predict class and probability
        prediction = int(self.model.predict(processed_df)[0])
        
        probability = 0.0
        if hasattr(self.model, "predict_proba"):
            # Probability of target = 1
            probability = float(self.model.predict_proba(processed_df)[0][1])
        
        # Compute confidence score
        confidence = probability if prediction == 1 else (1.0 - probability)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "risk_status": "High Risk" if prediction == 1 else "Low Risk"
        }

    def predict_batch(self, raw_inputs: list) -> list:
        """Predict for a list/batch of raw inputs."""
        results = []
        for inp in raw_inputs:
            try:
                res = self.predict_single(inp)
                results.append(res)
            except Exception as e:
                results.append({"error": str(e)})
        return results
