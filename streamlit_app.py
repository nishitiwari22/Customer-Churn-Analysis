import streamlit as st
import pandas as pd
import joblib
from src.predict import make_prediction

# Load model
saved = joblib.load("model.joblib")
model = saved["model"]
FEATURES = saved["features"]

st.title("Customer Churn Prediction")

# Inputs
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.slider("Num of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Prepare input
input_data = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary
}

# Predict
if st.button("Predict"):

    pred, prob = make_prediction(model, input_data, FEATURES)

    if pred == 1:
        st.error(f"⚠️ Likely to churn ({round(prob*100,2)}%)")
    else:
        st.success(f"✅ Not likely to churn ({round((1-prob)*100,2)}%)")