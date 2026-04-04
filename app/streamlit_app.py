import streamlit as st
import pickle
import os

# safer path handling
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Predictor")

tenure = st.slider("Tenure", 1, 72)
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):
    prediction = model.predict([[tenure, monthly_charges]])
    st.write("Churn Risk:", prediction)
