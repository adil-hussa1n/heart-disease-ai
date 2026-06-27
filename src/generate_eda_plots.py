import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
from src.preprocessing import load_raw_data, clean_data

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.size': 10
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_eda_reports():
    """Generate all requested EDA charts and save them in the reports folder."""
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
    # Load and clean dataset (without target splitting)
    df_raw = load_raw_data(config.RAW_DATA_PATH)
    df = clean_data(df_raw)
    
    # 1. Disease Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='target', data=df, palette=['#3b82f6', '#ef4444'])
    plt.title('Disease Distribution', fontsize=14, pad=15)
    plt.xlabel('Heart Disease (0 = Normal, 1 = Disease)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    ax.set_xticklabels(['No Disease', 'Disease'])
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "disease_distribution.png"), dpi=300)
    plt.close()

    # 2. Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='target', kde=True, multiple='stack', palette=['#3b82f6', '#ef4444'])
    plt.title('Age Distribution by Target Status', fontsize=14, pad=15)
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "age_distribution.png"), dpi=300)
    plt.close()

    # 3. Gender Comparison
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='sex', hue='target', data=df, palette=['#3b82f6', '#ef4444'])
    plt.title('Heart Disease Prevalence by Gender', fontsize=14, pad=15)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    ax.set_xticklabels(['Female', 'Male'])
    plt.legend(title='Status', labels=['No Disease', 'Disease'])
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "gender_comparison.png"), dpi=300)
    plt.close()

    # 4. Cholesterol Analysis
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='target', y='chol', data=df, palette=['#3b82f6', '#ef4444'])
    plt.title('Serum Cholesterol Levels by Target Status', fontsize=14, pad=15)
    plt.xlabel('Status', fontsize=12)
    plt.ylabel('Cholesterol (mg/dl)', fontsize=12)
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "cholesterol_analysis.png"), dpi=300)
    plt.close()

    # 5. Blood Pressure Analysis
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='target', y='trestbps', data=df, palette=['#3b82f6', '#ef4444'])
    plt.title('Resting Blood Pressure by Target Status', fontsize=14, pad=15)
    plt.xlabel('Status', fontsize=12)
    plt.ylabel('Resting Blood Pressure (mm Hg)', fontsize=12)
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "blood_pressure_analysis.png"), dpi=300)
    plt.close()

    # 6. Histograms for all Numerical Features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    numeric_cols_raw = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for i, col in enumerate(numeric_cols_raw):
        sns.histplot(df[col], ax=axes[i], kde=True, color='#3b82f6')
        axes[i].set_title(f'Histogram of {col.capitalize()}')
    axes[-1].axis('off') # Turn off the extra axis
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "numerical_histograms.png"), dpi=300)
    plt.close()

    # 7. Boxplots for Outlier Analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols_raw):
        sns.boxplot(y=df[col], ax=axes[i], color='#93c5fd')
        axes[i].set_title(f'Boxplot of {col.capitalize()}')
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "numerical_boxplots.png"), dpi=300)
    plt.close()

    # 8. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Filter only numerical columns for heatmaps or use full clean df
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap", fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # 9. Pairplot
    logger.info("Generating pairplot (this might take a few seconds)...")
    pair_cols = ["age", "trestbps", "chol", "thalach", "target"]
    pp = sns.pairplot(df[pair_cols], hue="target", palette=['#3b82f6', '#ef4444'], diag_kind="kde")
    pp.fig.suptitle("Pairplot of Key Clinical Variables", y=1.02, fontsize=16)
    plt.tight_layout()
    pp.savefig(os.path.join(config.REPORTS_DIR, "pairplots.png"), dpi=300)
    plt.close()
    
    # 10. Generate EDA text summary report
    eda_summary_path = os.path.join(config.REPORTS_DIR, "eda_summary.txt")
    with open(eda_summary_path, "w") as f:
        f.write("Exploratory Data Analysis (EDA) Summary Report\n")
        f.write("="*46 + "\n")
        f.write(f"Total Patient Records: {len(df)}\n")
        f.write(f"Normal (No Disease): {len(df[df['target'] == 0])} ({len(df[df['target'] == 0])/len(df)*100:.1f}%)\n")
        f.write(f"Heart Disease: {len(df[df['target'] == 1])} ({len(df[df['target'] == 1])/len(df)*100:.1f}%)\n\n")
        f.write("Statistical Description of Numerical Features:\n")
        f.write(df[numeric_cols_raw].describe().to_string())
        
    logger.info("All EDA reports and figures generated successfully.")

if __name__ == "__main__":
    generate_eda_reports()
