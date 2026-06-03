import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image # We need this new library to load the image!

# 1. Configure the Web Page
st.set_page_config(
    page_title="Alzheimer's Diagnostic AI", 
    page_icon="🧠", 
    layout="wide"
)

# Initialize Session State Memory
if 'X_ready' not in st.session_state:
    st.session_state['X_ready'] = None

# 2. Build the Header
st.title("🧠 Alzheimer's Disease Diagnostic AI")
st.markdown("***Web Application For Disease Diagnosis Using Biomarker Panel***")
st.markdown("---")

# 3. Define Intelligent Loading Functions
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'XGBoost_Strict_Panel.pkl') 
    return joblib.load(model_path)

@st.cache_data
def load_biology_data():
    data_path = os.path.join('biomarker_data', 'final_biological_interpretation_table.csv')
    return pd.read_csv(data_path)

# 4. Execute the Loaders
try:
    with st.spinner("Waking up the Machine Learning Engine..."):
        model = load_model()
        bio_df = load_biology_data()
except Exception as e:
    st.error(f"Critical Error Loading Assets: {e}")
    st.stop() 

# ---------------------------------------------------------
# STEP 4: HYBRID USER INPUT SYSTEM
# ---------------------------------------------------------
st.sidebar.header("🔬 Patient Data Input")
input_method = st.sidebar.radio("Choose Input Method:", ["📁 File Upload", "✍️ Manual Entry"])
patient_data = None

if input_method == "📁 File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Microbiome Data (.tsv / .csv)", type=["csv", "tsv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            patient_data = pd.read_csv(uploaded_file, index_col=0)
        else:
            patient_data = pd.read_csv(uploaded_file, sep='\t', index_col=0)
        st.sidebar.success("✅ File uploaded successfully!")

else:
    st.sidebar.subheader("Enter Biomarker Abundance")
    with st.sidebar.form("manual_input_form"):
        manual_inputs = {}
        for index, row in bio_df.iterrows():
            clean_name = row['Clean_Biomarker']
            long_name = row['Biomarker'] 
            val = st.number_input(f"{clean_name}:", min_value=-5.0, value=0.0, step=0.01)
            manual_inputs[long_name] = val
            
        if st.form_submit_button("Save Patient Profile"):
            patient_data = pd.DataFrame([manual_inputs], index=['Manual_Patient_1'])
            st.sidebar.success("✅ Manual profile saved!")
            st.session_state['X_ready'] = None 

if patient_data is None:
     st.session_state['X_ready'] = None

# ---------------------------------------------------------
# STEP 5: PREPROCESSING PIPELINE INTEGRATION
# ---------------------------------------------------------
if patient_data is not None:
    st.subheader("📋 Raw Patient Profile Preview")
    st.dataframe(patient_data.head())
    
    st.markdown("---")
    st.subheader("⚙️ Preprocessing Pipeline")
    
    if st.button("🔄 Run Data Preprocessing", type="secondary"):
        with st.spinner("Sanitizing features and aligning data tensors..."):
            try:
                expected_features = model.feature_names_in_
                X_ready = patient_data[expected_features]
                st.session_state['X_ready'] = X_ready
                st.success("✅ Preprocessing Complete! Data is perfectly aligned.")
            except KeyError as e:
                st.error("⚠️ Preprocessing Failed: Data Mismatch Error")
                st.code(f"Missing Features: {e}")
                st.session_state['X_ready'] = None

# ---------------------------------------------------------
# STEP 6 & 7: PREDICTION & BIOLOGICAL INTERPRETATION
# ---------------------------------------------------------
if st.session_state['X_ready'] is not None:
    st.markdown("---")
    st.subheader("🔮 AI Diagnostic Engine")
    
    if st.button("🚀 Run Clinical Diagnostics", type="primary", use_container_width=True):
        with st.spinner("Analyzing microbiome signatures..."):
            
            # --- STEP 6: AI PREDICTION ---
            X_predict = st.session_state['X_ready']
            predictions = model.predict(X_predict)
            probabilities = model.predict_proba(X_predict)
            
            st.success("✅ Analysis Complete!")
            st.subheader("📊 Diagnostic Results")
            
            if len(X_predict) > 1:
                results_df = patient_data.copy()
                results_df['Diagnosis'] = ["Alzheimer's" if p == 1 else "Healthy" for p in predictions]
                results_df['Confidence (%)'] = [round(prob[p] * 100, 2) for p, prob in zip(predictions, probabilities)]
                st.dataframe(results_df[['Diagnosis', 'Confidence (%)']])
                total_ad = sum(predictions)
                st.warning(f"**Summary:** Evaluated {len(X_predict)} patient profiles. Identified **{total_ad}** high-risk signatures.")
            else:
                pred_class = predictions[0]
                confidence = probabilities[0][pred_class] * 100
                col1, col2 = st.columns(2)
                with col1:
                    if pred_class == 1:
                        st.error("### 🚨 High Risk: Alzheimer's Profile Detected")
                    else:
                        st.success("### 🟢 Low Risk: Healthy Profile Detected")
                with col2:
                    st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
                    st.progress(int(confidence) / 100)

            # --- STEP 7: BIOLOGICAL INTERPRETATION ---
            st.markdown("---")
            st.subheader("🧬 Biological Interpretation")
            
            if len(X_predict) == 1:
                st.write("**Personalized Biomarker Profile Breakdown:**")
                patient_vals = X_predict.iloc[0].reset_index()
                patient_vals.columns = ['Biomarker', 'Patient_Abundance']
                merged_bio = pd.merge(bio_df, patient_vals, on='Biomarker', how='inner')
                
                display_cols = ['Clean_Biomarker', 'Patient_Abundance', 'Disease_Trend']
                if 'Biological_Function' in merged_bio.columns:
                    display_cols.append('Biological_Function')
                st.dataframe(merged_bio[display_cols])
            else:
                with st.expander("🔬 View Biomarker Knowledge Base (Reference Table)"):
                    st.dataframe(bio_df)

# ---------------------------------------------------------
# STEP 8: EXPLAINABLE AI VISUALIZATION
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💡 Explainable AI (XAI) Model Transparency")
st.info("Understand how the XGBoost algorithm evaluates biomarker abundance to reach a clinical diagnosis.")

with st.expander("📊 View AI Decision Matrix (SHAP Analysis)"):
    try:
        # Load the clean SHAP plot from the figures folder
        shap_image_path = os.path.join('figures', 'shap_summary_beeswarm_clean.png')
        img = Image.open(shap_image_path)
        
        # Display the image on the web app
        st.image(img, caption="SHAP Summary: Global Feature Importance", use_container_width=True)
        
        # Add a quick clinical guide on how to read the chart
        st.markdown("""
        **Clinical Interpretation Guide:**
        * 🔴 **Red dots:** Represent a **high abundance** of that specific bacteria in a patient.
        * 🔵 **Blue dots:** Represent a **low abundance** of that specific bacteria.
        * 👉 **Shift to the Right:** Pushes the model toward an **Alzheimer's** diagnosis.
        * 👈 **Shift to the Left:** Pushes the model toward a **Healthy** diagnosis.
        """)
    except FileNotFoundError:
        st.error("⚠️ SHAP visualization image not found. Please ensure 'shap_summary_beeswarm_clean.png' is in the 'figures/' folder.")
