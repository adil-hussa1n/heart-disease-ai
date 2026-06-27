import pytest
import os
from src.predict import HeartDiseasePredictor
from src import config

def test_predictor_initialization():
    # If the model has not been trained yet, it should raise FileNotFoundError
    # Otherwise, it should initialize
    if not os.path.exists(config.MODEL_PATH):
        with pytest.raises(FileNotFoundError):
            predictor = HeartDiseasePredictor()
    else:
        predictor = HeartDiseasePredictor()
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.imputer is not None
