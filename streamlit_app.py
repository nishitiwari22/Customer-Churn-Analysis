import streamlit as st
import joblib
import pandas as pd
import os


model = joblib.load("model.joblib")
print(os.getcwd())  # optional debug


st.title("Customer Churn Prediction")

tenure = st.slider("Tenure", 0, 72)
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):
    data = pd.DataFrame([[tenure, monthly_charges]], columns=["tenure", "MonthlyCharges"])
    prediction = model.predict(data)

    st.write("Churn" if prediction[0] == 1 else "No Churn")