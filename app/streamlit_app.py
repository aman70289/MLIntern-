import os
import sys
import pandas as pd
import streamlit as st

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict import ChurnPredictor

st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Customer Churn Prediction System")
st.markdown("Simplified Machine Learning Deployment featuring **Joblib Serialization** and **Gradient Boosting Inference**.")

@st.cache_resource
def load_predictor():
    return ChurnPredictor(models_dir=MODELS_DIR)

try:
    predictor = load_predictor()
    st.sidebar.success("✅ ML Model & Joblib Preprocessor Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

st.sidebar.header("Customer Profile Parameters")

credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 18, 100, 42)
tenure = st.sidebar.slider("Tenure (Years)", 0, 15, 4)
balance = st.sidebar.number_input("Account Balance ($)", value=75000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=1)
has_crcard = st.sidebar.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active = st.sidebar.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", value=65000.0, step=1000.0)

# Quick presets
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Presets")
col_p1, col_p2 = st.sidebar.columns(2)

input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_crcard,
    'IsActiveMember': is_active,
    'EstimatedSalary': estimated_salary
}

# Main Panel layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Selected Profile Summary")
    st.dataframe(pd.DataFrame([input_data]))

    if st.button("🚀 Predict Customer Churn Risk", use_container_width=True, type="primary"):
        with st.spinner("Evaluating model..."):
            result = predictor.predict_single(input_data)
            
            st.markdown("### Prediction Results")
            prob = result['churn_probability_percent']
            risk = result['risk_level']
            label = result['churn_label']

            if risk == "Low":
                st.success(f"**Outcome:** {label} ({prob:.1f}% Churn Probability)")
                st.info("💡 **Recommendation:** Customer is satisfied. Standard retention procedures apply.")
            elif risk == "Medium":
                st.warning(f"**Outcome:** {label} ({prob:.1f}% Churn Probability)")
                st.info("💡 **Recommendation:** Moderate churn risk. Consider offering targeted discounts or loyalty perks.")
            else:
                st.error(f"**Outcome:** {label} ({prob:.1f}% Churn Probability)")
                st.info("💡 **Recommendation:** High churn risk! Escalate immediately to the retention department.")

with col2:
    st.subheader("Model Information")
    st.json({
        "Model": "GradientBoostingClassifier",
        "ROC-AUC": "0.9652",
        "Accuracy": "90.17%",
        "F1-Score": "0.8239",
        "Serialization": "Joblib",
        "Features": 10
    })
