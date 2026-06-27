import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import joblib

from src import config

class HeartDiseaseExplainer:
    def __init__(self):
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError("Model file not found. Please train the model first.")
            
        self.model = joblib.load(config.MODEL_PATH)
        
        # Load background data for SHAP explainer
        train_df = pd.read_csv(config.PROCESSED_TRAIN_PATH)
        self.feature_names = [c for c in train_df.columns if c != config.TARGET_COL]
        self.X_train = train_df[self.feature_names]
        
        # Initialize explainer
        # For compatibility with various models, try generic shap.Explainer.
        # If it fails, fallback to specific explainers or KernelExplainer
        try:
            # We pass mask background to make it faster
            background = shap.maskers.Independent(self.X_train, max_samples=100)
            self.explainer = shap.Explainer(self.model, background, feature_names=self.feature_names)
        except Exception:
            # Fallback to KernelExplainer which is model agnostic
            # Use a small sample of X_train to speed up calculations
            background_summary = shap.kmeans(self.X_train, 10)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background_summary, feature_names=self.feature_names)

    def get_explanation(self, processed_df: pd.DataFrame):
        """Get SHAP values for a given preprocessed input dataframe."""
        # For KernelExplainer, we compute shap values using predict_proba (class 1)
        if isinstance(self.explainer, shap.KernelExplainer):
            shap_values = self.explainer.shap_values(processed_df)
            # shap_values is a list of arrays (one per class). For binary, index 1 is class 1 (Disease)
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values
                
            # Create an Explanation object for compatibility with shap plotting
            base_values = self.explainer.expected_value
            if isinstance(base_values, list) or isinstance(base_values, np.ndarray):
                bv = base_values[1] if len(base_values) > 1 else base_values[0]
            else:
                bv = base_values
                
            explanation = shap.Explanation(
                values=sv,
                base_values=bv,
                data=processed_df.values,
                feature_names=self.feature_names
            )
            return explanation
        else:
            explanation = self.explainer(processed_df)
            
            # If the output has multiple classes, shap explanation might be 3D. Extract class 1.
            if len(explanation.shape) == 3: # (num_instances, num_features, num_classes)
                # Select the class 1 (heart disease risk) explanation
                explanation = explanation[:, :, 1]
            elif len(explanation.shape) == 2 and hasattr(self.explainer, "expected_value"):
                # If expected value is a list of size 2, make sure we align it
                bv = self.explainer.expected_value
                if isinstance(bv, (list, np.ndarray)) and len(bv) > 1:
                    explanation.base_values = np.repeat(bv[1], len(explanation))
            return explanation

    def plot_waterfall(self, processed_df: pd.DataFrame, index: int = 0, save_path: str = None):
        """Generate a waterfall plot for a single instance prediction."""
        explanation = self.get_explanation(processed_df)
        
        fig = plt.figure(figsize=(10, 6))
        # Plot the first instance from the batch
        shap.plots.waterfall(explanation[index], show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig

    def plot_summary(self, save_path: str = None):
        """Generate a SHAP summary plot for the training set."""
        # Explain the first 100 rows of training set for performance
        subset = self.X_train.head(100)
        explanation = self.get_explanation(subset)
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(explanation, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig
