import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src import config
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_roc_curve, generate_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data():
    """Load train and test processed datasets."""
    train_df = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(config.PROCESSED_TEST_PATH)
    
    # Feature columns (everything except target)
    features = [c for c in train_df.columns if c != config.TARGET_COL]
    
    X_train = train_df[features]
    y_train = train_df[config.TARGET_COL]
    X_test = test_df[features]
    y_test = test_df[config.TARGET_COL]
    
    return X_train, y_train, X_test, y_test

def train_and_compare():
    """Train multiple models, compare results, and select the best model class."""
    X_train, y_train, X_test, y_test = load_data()
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    comparison_results = []
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
            
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        comparison_results.append(metrics)
        
    comparison_df = pd.DataFrame(comparison_results)
    # Order columns
    comparison_df = comparison_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    logger.info("\nModel Comparison Table:\n" + comparison_df.to_string(index=False))
    
    # Save comparison to reports
    comparison_df.to_csv(os.path.join(config.REPORTS_DIR, "model_comparison.csv"), index=False)
    
    # Select best model based on F1 score
    best_row = comparison_df.sort_values(by="f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    logger.info(f"Best model based on F1-Score: {best_model_name} (F1: {best_row['f1']:.4f})")
    
    return best_model_name, models[best_model_name]

def tune_and_save_best(best_model_name, initial_model):
    """Perform hyperparameter tuning for the selected best model class and save it."""
    X_train, y_train, X_test, y_test = load_data()
    
    logger.info(f"Tuning hyperparameters for {best_model_name}...")
    
    # Define grid search parameter grids based on model type
    if best_model_name == "Logistic Regression":
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
        
    elif best_model_name == "Decision Tree":
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
        
    elif best_model_name == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 8, None],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
        
    elif best_model_name == "Support Vector Machine":
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
        
    elif best_model_name == "KNN":
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1', n_jobs=-1)
        
    elif best_model_name == "Gradient Boosting":
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        }
        grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
        
    elif best_model_name == "XGBoost":
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2]
        }
        grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
        
    else:
        logger.warning(f"No grid search config defined for {best_model_name}. Saving base model.")
        joblib.dump(initial_model, config.MODEL_PATH)
        return
        
    grid_search.fit(X_train, y_train)
    best_tuned_model = grid_search.best_estimator_
    
    logger.info(f"Hyperparameter tuning completed. Best Params: {grid_search.best_params_}")
    
    # Evaluate tuned model
    y_pred = best_tuned_model.predict(X_test)
    y_prob = best_tuned_model.predict_proba(X_test)[:, 1] if hasattr(best_tuned_model, "predict_proba") else None
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    logger.info(f"Tuned Model Metrics - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1']:.4f}")
    
    # Save the model
    joblib.dump(best_tuned_model, config.MODEL_PATH)
    logger.info(f"Best tuned model saved to {config.MODEL_PATH}")
    
    # Generate final evaluation plots & reports
    plot_confusion_matrix(y_test, y_pred, best_model_name, config.REPORTS_DIR)
    plot_roc_curve(y_test, y_prob, best_model_name, config.REPORTS_DIR)
    generate_reports(y_test, y_pred, y_prob, best_model_name, config.REPORTS_DIR)
    
    # Save feature importance if available
    save_feature_importance(best_tuned_model, X_train.columns)

def save_feature_importance(model, feature_names):
    """Extract and save feature importance report if model supports it."""
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        
    if importance is not None:
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False)
        
        fi_path = os.path.join(config.REPORTS_DIR, "feature_importance.csv")
        fi_df.to_csv(fi_path, index=False)
        logger.info(f"Feature importance saved to {fi_path}")

def main():
    best_name, best_model = train_and_compare()
    tune_and_save_best(best_name, best_model)

if __name__ == "__main__":
    main()
