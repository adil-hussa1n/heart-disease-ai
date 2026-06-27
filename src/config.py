import os

# Project Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)

# File Paths
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "cleveland.csv")
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.pkl")

# Columns Configuration
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "hr_max_age_ratio", "bp_chol_interaction", "double_product"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal", "age_group"]
BINARY_COLS = ["sex", "fbs", "exang"]
TARGET_COL = "target"
