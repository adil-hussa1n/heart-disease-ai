import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for heart disease classification."""
    df = df.copy()
    
    # 1. Max Heart Rate Ratio to Age-Predicted Max Heart Rate (220 - age)
    # Heart rate ratio can indicate cardiovascular health
    df['hr_max_age_ratio'] = df['thalach'] / (220 - df['age'])
    
    # 2. Blood Pressure - Cholesterol interaction
    df['bp_chol_interaction'] = df['trestbps'] * df['chol']
    
    # 3. Double Product (trestbps * thalach) - indicator of myocardial oxygen demand
    df['double_product'] = df['trestbps'] * df['thalach']
    
    # 4. Age groups (categorical)
    # Binned age: Youth/Early-Adulthood (<45), Middle-Age (45-60), Senior (>60)
    # We will map it to integers/discrete values to make it easy for all algorithms
    conditions = [
        (df['age'] < 45),
        (df['age'] >= 45) & (df['age'] <= 60),
        (df['age'] > 60)
    ]
    choices = [0, 1, 2] # 0: Young, 1: Middle-aged, 2: Senior
    df['age_group'] = np.select(conditions, choices, default=1)
    
    return df
