
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load assets
@st.cache_resource
def load_assets():
    preprocessor = joblib.load('models\preprocessor.joblib')
    model = joblib.load('models\random_forest.joblib')
    explainer = shap.TreeExplainer(model)
    return preprocessor, model, explainer

preprocessor, model, explainer = load_assets()

# Feature descriptions
feature_descriptions = {
    'age': 'Age of the applicant in years',
    'sex': 'Gender of the applicant',
    'job': 'Job qualification (0 - unskilled, 1 - skilled, 2 - highly skilled)',
    'housing': 'Type of housing (own, rent, or free)',
    'saving_accounts': 'Amount in savings account',
    'checking_account': 'Amount in checking account',
    'credit_amount': 'Loan amount in DM',
    'duration': 'Loan duration in months',
    'purpose': 'Purpose of the loan'
}

# App layout
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
This tool predicts credit risk using machine learning. 
Input applicant details and get instant risk assessment 
with explainable AI insights.
""")

# Main interface
st.title("🏦 Intelligent Credit Risk Assessment")
st.subheader("Applicant Information")

# Input form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 100, 30)
        job = st.selectbox("Job Level", [0, 1, 2], 
                         format_func=lambda x: ["Unskilled", "Skilled", "Highly Skilled"][x])
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        
    with col2:
        saving_accounts = st.selectbox("Savings Account", ["little", "moderate", "quite rich", "rich"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        purpose = st.selectbox("Loan Purpose", ["car", "furniture/equipment", "radio/TV", 
                                              "domestic appliances", "repairs", "education", 
                                              "business", "vacation"])
        
    with col3:
        credit_amount = st.number_input("Loan Amount (DM)", 250, 20000, 2000)
        duration = st.slider("Loan Duration (months)", 6, 72, 12)
        sex = st.radio("Gender", ["male", "female"])
    
    submitted = st.form_submit_button("Assess Credit Risk")

# Prediction and results
if submitted:
    # Create input dataframe
    input_data = pd.DataFrame([[age, sex, job, housing, saving_accounts, 
                              checking_account, credit_amount, duration, purpose]],
                            columns=feature_descriptions.keys())
    
    # Preprocess
    processed_data = preprocessor.transform(input_data)
    
    # Predict
    prediction = model.predict(processed_data)[0]
    proba = model.predict_proba(processed_data)[0][1]
    
    # Display results
    st.subheader("Risk Assessment Results")
    result_col, proba_col, info_col = st.columns(3)
    
    with result_col:
        st.metric("Predicted Risk", 
                value="High Risk ⚠️" if prediction == 1 else "Low Risk ✅",
                delta=f"Probability: {proba:.2%}")
        
    with proba_col:
        st.write("**Probability Distribution**")
        fig, ax = plt.subplots()
        ax.barh(['Low Risk', 'High Risk'], [1-proba, proba], color=['#2ecc71', '#e74c3c'])
        ax.set_xlim(0, 1)
        st.pyplot(fig)
        
    with info_col:
        st.write("**Key Influencing Factors**")
        shap_values = explainer.shap_values(processed_data)
        feature_names = preprocessor.get_feature_names_out()
        
        # Get top 3 influential features
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'impact': shap_values[0]
        }).sort_values('impact', ascending=False).head(3)
        
        for idx, row in shap_df.iterrows():
            st.write(f"- {row['feature'].split('_')[-1]} ({row['impact']:.2f} impact)")
    
    # SHAP explanation
    st.subheader("AI Explanation")
    explain_col, strategy_col = st.columns([2, 1])
    
    with explain_col:
        st.write("**Feature Impact Visualization**")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, processed_data, feature_names=feature_names, 
                         plot_type="bar", max_display=10, show=False)
        st.pyplot(fig)
        
    with strategy_col:
        st.write("**Risk Mitigation Suggestions**")
        if prediction == 1:
            st.markdown("""
            - Consider reducing loan amount
            - Shorten repayment duration
            - Request additional collateral
            - Review applicant's employment history
            """)
        else:
            st.markdown("""
            - Favorable credit terms available
            - Consider upselling financial products
            - Fast-track approval process
            - Offer loyalty benefits
            """)
    
    # Raw data preview
    with st.expander("View Processed Input Data"):
        st.dataframe(pd.DataFrame(processed_data, columns=feature_names))

# Add footer
st.markdown("---")
st.markdown("""
*Credit Risk Analyzer v1.0*  
*Powered by Machine Learning - For demonstration purposes only*
""")