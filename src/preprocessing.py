import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset and assign column names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}")
    
    logger.info(f"Loading raw data from {filepath}")
    # Raw cleveland.csv has no header
    df = pd.read_csv(filepath, header=None, names=config.COLUMNS)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and target binary mapping."""
    df = df.copy()
    
    # 1. Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Convert ca and thal columns to float (since they might have '?' which made them object)
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    
    # 2. Impute missing values using median (for numerical/discrete columns)
    imputer = SimpleImputer(strategy='median')
    # Impute only ca and thal
    df[['ca', 'thal']] = imputer.fit_transform(df[['ca', 'thal']])
    
    # Save the imputer for prediction pipeline
    joblib.dump(imputer, config.IMPUTER_PATH)
    logger.info(f"Saved imputer to {config.IMPUTER_PATH}")
    
    # 3. Remove duplicates
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        logger.info(f"Removing {duplicates_count} duplicate rows.")
        df = df.drop_duplicates()
        
    # 4. Map target to binary classification: 0 (no disease), 1 (disease)
    # The original dataset has target values 0, 1, 2, 3, 4. 0 is normal, 1-4 are heart disease.
    df[config.TARGET_COL] = df[config.TARGET_COL].apply(lambda x: 1 if x > 0 else 0)
    
    logger.info(f"Data cleaned. Shape: {df.shape}")
    return df

def detect_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Optionally flag or cap outliers based on IQR."""
    df_cleaned = df.copy()
    for col in columns:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Capping outliers to bounds instead of dropping, to preserve data size
        outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
        outliers_count = outliers.sum()
        if outliers_count > 0:
            logger.info(f"Capping {outliers_count} outliers in column '{col}' to range [{lower_bound:.1f}, {upper_bound:.1f}]")
            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
            
    return df_cleaned

def preprocess_and_save_data():
    """Main pipeline execution for preprocessing."""
    from src.feature_engineering import engineer_features
    
    df = load_raw_data(config.RAW_DATA_PATH)
    df_cleaned = clean_data(df)
    
    # Handle outliers in initial numerical columns
    initial_numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df_cleaned = detect_outliers_iqr(df_cleaned, initial_numeric)
    
    # Apply feature engineering
    df_cleaned = engineer_features(df_cleaned)
    logger.info("Feature engineering completed.")
    
    # Train-test split before scaling to avoid data leakage
    train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42, stratify=df_cleaned[config.TARGET_COL])
    
    # Scale numerical features
    scaler = StandardScaler()
    
    # Fit scaler on train set, transform both
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[config.NUMERICAL_COLS] = scaler.fit_transform(train_df[config.NUMERICAL_COLS])
    test_scaled[config.NUMERICAL_COLS] = scaler.transform(test_df[config.NUMERICAL_COLS])
    
    # Save the scaler for inference
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"Saved scaler to {config.SCALER_PATH}")
    
    # Save train and test data
    train_scaled.to_csv(config.PROCESSED_TRAIN_PATH, index=False)
    test_scaled.to_csv(config.PROCESSED_TEST_PATH, index=False)
    logger.info(f"Processed train data saved to {config.PROCESSED_TRAIN_PATH}")
    logger.info(f"Processed test data saved to {config.PROCESSED_TEST_PATH}")

if __name__ == "__main__":
    preprocess_and_save_data()
