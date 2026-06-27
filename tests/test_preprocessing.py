import pytest
import pandas as pd
import numpy as np
import os

from src import config
from src.preprocessing import clean_data, detect_outliers_iqr

def test_clean_data():
    # Construct a dummy DataFrame similar to raw data
    dummy_data = pd.DataFrame({
        "age": [50.0, 60.0],
        "sex": [1.0, 0.0],
        "cp": [4.0, 3.0],
        "trestbps": [140.0, 120.0],
        "chol": [240.0, 200.0],
        "fbs": [0.0, 1.0],
        "restecg": [2.0, 0.0],
        "thalach": [150.0, 160.0],
        "exang": [0.0, 1.0],
        "oldpeak": [1.5, 0.0],
        "slope": [2.0, 1.0],
        "ca": ["0.0", "?"], # includes missing marker
        "thal": ["3.0", "6.0"],
        "target": [2, 0] # multi-class targets
    })
    
    cleaned = clean_data(dummy_data)
    
    # 1. '?' should be imputed
    assert not cleaned.isnull().values.any()
    
    # 2. ca and thal should be float
    assert cleaned['ca'].dtype == float
    assert cleaned['thal'].dtype == float
    
    # 3. target should be binary [0, 1]
    assert set(cleaned['target'].unique()).issubset({0, 1})
    assert cleaned.loc[0, 'target'] == 1 # 2 mapped to 1
    assert cleaned.loc[1, 'target'] == 0 # 0 mapped to 0

def test_detect_outliers_iqr():
    df = pd.DataFrame({"val": [10.0, 12.0, 11.0, 13.0, 100.0]}) # 100.0 is outlier
    df_cleaned = detect_outliers_iqr(df, ["val"], factor=1.5)
    
    # Value at index 4 (100.0) should be capped to upper bound
    assert df_cleaned.loc[4, "val"] < 100.0
