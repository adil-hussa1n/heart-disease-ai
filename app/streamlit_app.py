import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
import datetime

# Setup pathing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.predict import HeartDiseasePredictor
from src.explain import HeartDiseaseExplainer

# Page Config
st.set_page_config(
    page_title="CardioAI | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Blue and White Theme)
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 8px 24px;
        border: none;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border-left: 5px solid #2563eb;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .disclaimer {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        color: #991b1b;
        font-size: 13px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to get database connection
def get_db_connection():
    db_path = os.path.join(config.DATA_DIR, "predictions.db")
    conn = sqlite3.connect(db_path)
    return conn

# Try loading prediction assets
@st.cache_resource
def load_ml_assets():
    try:
        predictor = HeartDiseasePredictor()
        explainer = HeartDiseaseExplainer()
        return predictor, explainer, None
    except Exception as e:
        return None, None, str(e)

predictor, explainer, load_error = load_ml_assets()

# Sidebar Navigation
st.sidebar.markdown("<div style='text-align: center; padding-bottom: 10px;'><h2 style='color:#2563eb; margin:0;'>CardioAI</h2><p style='color:#64748b; font-size:12px; margin:0;'>Heart Disease Diagnostic Assistant</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard", "Predict Disease", "Dataset Explorer", "Model Performance", "Explain Prediction", "About"]
)

# Render disclaimer in sidebar on all pages
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='background-color:#eff6ff; padding:12px; border-radius:8px; border-left:4px solid #3b82f6; font-size:11px; color:#1e40af;'>
        <strong>Disclaimer:</strong> This system uses a machine learning model for prediction. It is designed for educational/information purposes only and <strong>does not</strong> constitute medical advice or formal diagnosis.
    </div>
""", unsafe_allow_html=True)

# Load datasets
@st.cache_data
def get_clean_dataset():
    # If processed test data exists, load it, otherwise construct it
    train_path = config.PROCESSED_TRAIN_PATH
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        # Unscale numerical columns for human viewing in the explorer
        # We can read the raw dataset if possible
        if os.path.exists(config.RAW_DATA_PATH):
            from src.preprocessing import load_raw_data, clean_data
            return clean_data(load_raw_data(config.RAW_DATA_PATH))
        return train_df
    return None

# ==========================================
# PAGE: DASHBOARD
# ==========================================
if page == "Dashboard":
    st.markdown("<h1>Clinical Dashboard Overview</h1>", unsafe_allow_html=True)
    st.write("Real-time telemetry and overview of the heart disease prediction system.")
    
    # Check if database has prediction records
    try:
        conn = get_db_connection()
        history_df = pd.read_sql("SELECT * FROM predictions", conn)
        conn.close()
    except Exception:
        history_df = pd.DataFrame()
        
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
            <div class="card">
                <div class="metric-label">Total Predictions Logged</div>
                <div class="metric-value">{len(history_df) if not history_df.empty else 0}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        high_risk = len(history_df[history_df['predicted_class'] == 1]) if not history_df.empty else 0
        st.markdown(f"""
            <div class="card" style="border-left-color: #ef4444;">
                <div class="metric-label">High Risk Cases Logged</div>
                <div class="metric-value">{high_risk}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        # Load main dataset to show stats
        dataset = get_clean_dataset()
        dataset_len = len(dataset) if dataset is not None else 303
        st.markdown(f"""
            <div class="card" style="border-left-color: #10b981;">
                <div class="metric-label">Training Dataset Size</div>
                <div class="metric-value">{dataset_len} Patients</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
            <div class="card" style="border-left-color: #f59e0b;">
                <div class="metric-label">System Status</div>
                <div class="metric-value" style="color: #10b981;">Online</div>
            </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predictive Analytics Model")
        st.write("CardioAI is currently running an optimized classifier. The model operates by ingestion of clinical attributes and uses cross-validated hyperparameter spaces to render prediction indices.")
        if predictor is not None:
            st.info(f"Active Model: **{type(predictor.model).__name__}**")
        else:
            st.warning("No model currently loaded. Run training script first.")
            
        st.subheader("Quick Cardiovascular Wellness Tips")
        st.markdown("""
        * **Dietary Intake:** Focus on leafy greens, whole grains, and omega-3 rich fish. Limit processed sugar and sodium.
        * **Physical Activity:** Aim for at least 150 minutes of moderate aerobic exercise weekly.
        * **Monitoring:** Maintain routine evaluations of blood pressure, blood glucose, and cholesterol index.
        """)
        
    with col2:
        st.subheader("Historical Risk Distribution (Logged)")
        if not history_df.empty:
            fig = px.pie(history_df, names="risk_status", color="risk_status",
                         color_discrete_map={"High Risk": "#ef4444", "Low Risk": "#3b82f6"},
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions have been logged in history database yet. Run some predictions to see live metrics!")

# ==========================================
# PAGE: PREDICT DISEASE
# ==========================================
elif page == "Predict Disease":
    st.markdown("<h1>Cardiac Risk Analysis</h1>", unsafe_allow_html=True)
    
    if load_error:
        st.error(f"Error loading system assets: {load_error}. Please ensure data is preprocessed and model is trained by running preprocessing/training scripts.")
    else:
        st.write("Input patient physiological parameters below to evaluate cardiac disease probability indices.")
        
        tab_single, tab_batch = st.tabs(["Single Patient Prediction", "Batch Prediction (CSV Upload)"])
        
        # --- SINGLE PREDICTION ---
        with tab_single:
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.slider("Age (Years)", 1, 100, 50)
                    sex = st.selectbox("Gender", ["Female", "Male"], index=1)
                    cp = st.selectbox("Chest Pain Type", [
                        "1: Typical Angina",
                        "2: Atypical Angina",
                        "3: Non-Anginal Pain",
                        "4: Asymptomatic"
                    ], index=3)
                    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
                    
                with c2:
                    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 500, 200)
                    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"], index=0)
                    restecg = st.selectbox("Resting Electrocardiographic Results", [
                        "0: Normal",
                        "1: ST-T Wave Abnormality",
                        "2: Left Ventricular Hypertrophy"
                    ], index=0)
                    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
                    
                with c3:
                    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0)
                    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
                    slope = st.selectbox("Peak Exercise ST Segment Slope", [
                        "1: Upsloping",
                        "2: Flat",
                        "3: Downsloping"
                    ], index=1)
                    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3], index=0)
                    thal = st.selectbox("Thalassemia Type", [
                        "3: Normal",
                        "6: Fixed Defect",
                        "7: Reversible Defect"
                    ], index=0)
                    
                submit = st.form_submit_button("Perform Diagnostic Inference")
                
            if submit:
                # Build raw input dict
                sex_val = 1.0 if sex == "Male" else 0.0
                fbs_val = 1.0 if fbs == "True" else 0.0
                exang_val = 1.0 if exang == "Yes" else 0.0
                cp_val = float(cp.split(":")[0])
                restecg_val = float(restecg.split(":")[0])
                slope_val = float(slope.split(":")[0])
                thal_val = float(thal.split(":")[0])
                
                raw_input = {
                    "age": float(age),
                    "sex": sex_val,
                    "cp": cp_val,
                    "trestbps": float(trestbps),
                    "chol": float(chol),
                    "fbs": fbs_val,
                    "restecg": restecg_val,
                    "thalach": float(thalach),
                    "exang": exang_val,
                    "oldpeak": oldpeak,
                    "slope": slope_val,
                    "ca": float(ca),
                    "thal": thal_val
                }
                
                with st.spinner("Analyzing patient biometrics..."):
                    res = predictor.predict_single(raw_input)
                    processed_df = predictor.preprocess_input(raw_input)
                    
                    st.success("Analysis Complete!")
                    
                    # Layout Results
                    col_res, col_chart = st.columns([1, 1])
                    with col_res:
                        pred_class = res["prediction"]
                        prob = res["probability"]
                        confidence = res["confidence"]
                        
                        if pred_class == 1:
                            st.markdown(f"""
                                <div style="background-color: #fef2f2; border-left: 6px solid #ef4444; padding: 20px; border-radius: 8px;">
                                    <h3 style="color:#b91c1c; margin:0;">HIGH RISK INDEX DETECTED</h3>
                                    <p style="font-size:18px; margin: 5px 0 0 0;">Probability: <strong>{prob*100:.1f}%</strong></p>
                                    <p style="font-size:14px; color:#7f1d1d;">Confidence Level: {confidence*100:.1f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style="background-color: #ecfdf5; border-left: 6px solid #10b981; padding: 20px; border-radius: 8px;">
                                    <h3 style="color:#047857; margin:0;">LOW RISK INDEX DETECTED</h3>
                                    <p style="font-size:18px; margin: 5px 0 0 0;">Probability: <strong>{prob*100:.1f}%</strong></p>
                                    <p style="font-size:14px; color:#064e3b;">Confidence Level: {confidence*100:.1f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        # Lifestyle recommendations
                        st.subheader("Tailored Recommendations")
                        if pred_class == 1:
                            st.write("- **Schedule Clinical Assessment:** Consult a cardiologist immediately for further diagnostic checks (e.g., stress test, angiogram).")
                            st.write("- **Regular BP Monitoring:** Track blood pressure daily and review medication adherence.")
                            st.write("- **Heart-Healthy Lifestyle:** Adopt the DASH diet, reduce sodium intake, and avoid strenuous exercises until cleared by a physician.")
                        else:
                            st.write("- **Preventive Care:** Continue routine physical examinations and maintain active exercise regimes.")
                            st.write("- **Diet & Nutrition:** Keep cholesterol levels low by limiting saturated fat intake.")
                            st.write("- **Activity:** Sustain 30 mins/day of cardiovascular workouts.")

                        # Generate PDF Report function
                        def generate_pdf(raw, res):
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Helvetica", "B", 18)
                            pdf.set_text_color(30, 58, 138)
                            pdf.cell(0, 10, "CardioAI Clinical Evaluation Report", new_x="LMARGIN", new_y="NEXT", align="C")
                            pdf.ln(5)
                            
                            pdf.set_font("Helvetica", "", 10)
                            pdf.set_text_color(100, 116, 139)
                            pdf.cell(0, 5, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align="C")
                            pdf.ln(10)
                            
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.set_text_color(37, 99, 235)
                            pdf.cell(0, 8, "Patient Metrics Summary", new_x="LMARGIN", new_y="NEXT")
                            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                            pdf.ln(4)
                            
                            pdf.set_font("Helvetica", "", 11)
                            pdf.set_text_color(50, 50, 50)
                            for k, v in raw.items():
                                pdf.cell(90, 6, f"{k.upper()}: {v}", border=0)
                                if pdf.get_x() > 100:
                                    pdf.ln(6)
                            if pdf.get_x() <= 100:
                                pdf.ln(6)
                            pdf.ln(6)
                            
                            pdf.set_font("Helvetica", "B", 14)
                            pdf.set_text_color(37, 99, 235)
                            pdf.cell(0, 8, "Inference Outcome", new_x="LMARGIN", new_y="NEXT")
                            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                            pdf.ln(4)
                            
                            pdf.set_font("Helvetica", "B", 12)
                            pdf.cell(0, 8, f"Status: {res['risk_status']}", new_x="LMARGIN", new_y="NEXT")
                            pdf.set_font("Helvetica", "", 11)
                            pdf.cell(0, 6, f"Risk Probability Score: {res['probability']*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
                            pdf.cell(0, 6, f"Prediction Confidence: {res['confidence']*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
                            pdf.ln(10)
                            
                            pdf.set_font("Helvetica", "I", 9)
                            pdf.set_text_color(150, 50, 50)
                            pdf.multi_cell(0, 5, "Disclaimer: This prediction is generated by a machine learning model and should not be considered medical advice.")
                            
                            return pdf.output()

                        pdf_bytes = generate_pdf(raw_input, res)
                        st.download_button(
                            label="Export Evaluation Report as PDF",
                            data=bytes(pdf_bytes),
                            file_name="cardio_ai_evaluation_report.pdf",
                            mime="application/pdf"
                        )
                            
                    with col_chart:
                        # Probability gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Disease Risk Probability (%)", 'font': {'size': 16, 'color': '#1e3a8a'}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1e3a8a"},
                                'bar': {'color': "#ef4444" if pred_class == 1 else "#3b82f6"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "#cbd5e1",
                                'steps': [
                                    {'range': [0, 40], 'color': '#d1fae5'},
                                    {'range': [40, 70], 'color': '#fef3c7'},
                                    {'range': [70, 100], 'color': '#fee2e2'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # SHAP waterfall plot
                    st.subheader("Explaining Prediction (SHAP Waterfall Plot)")
                    try:
                        fig = explainer.plot_waterfall(processed_df, index=0)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not render SHAP waterfall plot: {e}. Ensure model can be explained by SHAP explainer.")

        # --- BATCH PREDICTION ---
        with tab_batch:
            st.write("Upload a CSV file containing patient parameters. The CSV must contain the following columns:")
            st.code(", ".join([c for c in config.COLUMNS if c != config.TARGET_COL]))
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    st.write("Uploaded Data Preview:")
                    st.dataframe(input_df.head())
                    
                    if st.button("Perform Batch Diagnostic Inference"):
                        patients_list = input_df.to_dict(orient="records")
                        with st.spinner("Processing batch inputs..."):
                            batch_results = predictor.predict_batch(patients_list)
                            
                            # Build output dataframe
                            res_df = input_df.copy()
                            res_df["prediction"] = [r.get("prediction", np.nan) for r in batch_results]
                            res_df["probability"] = [r.get("probability", np.nan) for r in batch_results]
                            res_df["risk_status"] = [r.get("risk_status", "N/A") for r in batch_results]
                            
                            st.success("Batch Prediction Complete!")
                            st.dataframe(res_df)
                            
                            # Provide download link for output
                            csv_buffer = io.StringIO()
                            res_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv_buffer.getvalue(),
                                file_name="batch_predictions_results.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")

# ==========================================
# PAGE: DATASET EXPLORER
# ==========================================
elif page == "Dataset Explorer":
    st.markdown("<h1>UCI Cleveland Heart Disease Dataset Explorer</h1>", unsafe_allow_html=True)
    st.write("Browse, filter, and sort the clinical study dataset.")
    
    df = get_clean_dataset()
    if df is not None:
        # Filtering tools
        st.subheader("Filter and Search Patient Directory")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_filter = st.slider("Filter by Maximum Age", int(df['age'].min()), int(df['age'].max()), int(df['age'].max()))
        with c2:
            gender_filter = st.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
        with c3:
            disease_filter = st.multiselect("Disease Status", ["No Disease", "Disease"], default=["No Disease", "Disease"])
            
        # Map values to match filters
        gender_map = {0: "Female", 1: "Male"}
        disease_map = {0: "No Disease", 1: "Disease"}
        
        filtered_df = df.copy()
        filtered_df['Gender_Name'] = filtered_df['sex'].map(gender_map)
        filtered_df['Status_Name'] = filtered_df['target'].map(disease_map)
        
        # Apply filters
        filtered_df = filtered_df[
            (filtered_df['age'] <= age_filter) &
            (filtered_df['Gender_Name'].isin(gender_filter)) &
            (filtered_df['Status_Name'].isin(disease_filter))
        ]
        
        # Drop temporary columns for final view
        view_df = filtered_df.drop(columns=['Gender_Name', 'Status_Name'])
        
        st.write(f"Displaying {len(view_df)} matching records:")
        st.dataframe(view_df, use_container_width=True)
        
        # Export/Download CSV
        csv_buffer = io.StringIO()
        view_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Filtered Dataset as CSV",
            data=csv_buffer.getvalue(),
            file_name="filtered_heart_disease_dataset.csv",
            mime="text/csv"
        )
    else:
        st.warning("Processed or raw datasets could not be loaded. Please ensure dataset is fetched and preprocessed.")

# ==========================================
# PAGE: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.markdown("<h1>System Performance Metrics</h1>", unsafe_allow_html=True)
    st.write("Review training and evaluation details for the heart disease prediction model.")
    
    # Check if reports exist
    comparison_path = os.path.join(config.REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        st.subheader("Model Evaluation Comparison Table")
        st.dataframe(comparison_df.style.highlight_max(subset=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], color='#d1fae5'))
        
        # Render the bar chart comparison
        fig = px.bar(comparison_df, x="model", y=["accuracy", "f1", "roc_auc"],
                     barmode="group", title="Model Performance Metric Comparison",
                     labels={"value": "Score", "variable": "Metric"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Performance reports not found. Run training script to generate comparison reports.")
        
    # Render static curves if generated
    st.subheader("Evaluation Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Confusion Matrix**")
        # Try loading confusion matrix
        if predictor is not None:
            model_name_lower = type(predictor.model).__name__.lower().replace(" ", "_")
            cm_path = os.path.join(config.REPORTS_DIR, f"{model_name_lower}_confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("No saved confusion matrix plot found.")
    with col2:
        st.write("**ROC Curve**")
        if predictor is not None:
            model_name_lower = type(predictor.model).__name__.lower().replace(" ", "_")
            roc_path = os.path.join(config.REPORTS_DIR, f"{model_name_lower}_roc_curve.png")
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
            else:
                st.info("No saved ROC curve plot found.")

# ==========================================
# PAGE: EXPLAIN PREDICTION
# ==========================================
elif page == "Explain Prediction":
    st.markdown("<h1>Global Model Explainability</h1>", unsafe_allow_html=True)
    st.write("Understand which features drive the model predictions globally using SHAP (SHapley Additive exPlanations).")
    
    if explainer is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Global Feature Importance (SHAP Summary Plot)")
            st.write("This plot shows the distribution of SHAP values for each feature in the training set. Features are ranked by importance. Red points represent high values of the feature, and blue points represent low values.")
            try:
                fig = explainer.plot_summary()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not render beeswarm plot: {e}.")
        with col2:
            st.subheader("Calculated Feature Importance Summary")
            fi_path = os.path.join(config.REPORTS_DIR, "feature_importance.csv")
            if os.path.exists(fi_path):
                fi_df = pd.read_csv(fi_path)
                st.dataframe(fi_df, use_container_width=True)
                
                # Interactive bar plot
                fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                             title="Feature Importance Ranking",
                             color="importance", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance ranking available.")
    else:
        st.warning("Explainer assets could not be loaded. Please ensure the model is trained.")

# ==========================================
# PAGE: ABOUT
# ==========================================
elif page == "About":
    st.markdown("<h1>About CardioAI</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    CardioAI is a clinical decision support application powered by Machine Learning.
    By analyzing patient physiological attributes, it evaluates the statistical likelihood of coronary artery disease.
    
    ### Tech Stack
    - **Backend:** FastAPI, SQLite3
    - **Frontend:** Streamlit Dashboard, Plotly
    - **Machine Learning:** Scikit-learn, XGBoost, SHAP Explainer
    - **Language:** Python 3.12
    
    ### Attribute Glossary
    1. **Age:** Age of the patient in years.
    2. **Sex:** Sex (1 = Male, 0 = Female).
    3. **Chest Pain (CP):** Chest pain type (1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic).
    4. **Trestbps:** Resting blood pressure on admission to the hospital (mm Hg).
    5. **Chol:** Serum cholesterol level in mg/dl.
    6. **FBS:** Fasting blood sugar > 120 mg/dl (1 = True, 0 = False).
    7. **RestECG:** Resting ECG results (0 = Normal, 1 = ST-T wave anomaly, 2 = Left ventricular hypertrophy).
    8. **Thalach:** Maximum heart rate achieved during exercise stress test.
    9. **Exang:** Exercise induced angina (1 = Yes, 0 = No).
    10. **Oldpeak:** ST depression induced by exercise relative to rest.
    11. **Slope:** Slope of the peak exercise ST segment.
    12. **CA:** Number of major vessels colored by fluoroscopy (0-3).
    13. **Thal:** Thalassemia type (3 = Normal, 6 = Fixed defect, 7 = Reversible defect).
    
    ---
    ### Educational Disclaimer
    <div class="disclaimer">
        <strong>IMPORTANT LEGAL NOTICE:</strong> CardioAI is developed strictly for educational, demonstration, and research purposes.
        Predictions generated by this machine learning model must NOT be treated as a professional medical diagnosis, advice, or treatment recommendation.
        Always consult a licensed clinical practitioner or cardiologist for medical concerns.
    </div>
    """, unsafe_allow_html=True)
