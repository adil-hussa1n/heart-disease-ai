import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from src import config

def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Calculate standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name: str, save_dir: str):
    """Plot and save ROC curve."""
    if y_prob is None or np.isnan(y_prob).any():
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.savefig(filepath, dpi=300)
    plt.close()

def generate_reports(y_true, y_pred, y_prob, model_name: str, save_dir: str):
    """Generate and save text reports."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Classification report
    report_dict = classification_report(y_true, y_pred, target_names=["No Disease", "Disease"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_classification_report.csv")
    report_df.to_csv(report_path)
    
    # 2. Evaluation report text file
    eval_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_evaluation_report.txt")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    with open(eval_path, "w") as f:
        f.write(f"Evaluation Report for {model_name}\n")
        f.write("="*40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k.capitalize()}: {v:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=["No Disease", "Disease"]))
