import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Customer Churn Predictor")

tenure = st.slider("Tenure", 1, 72)
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):
    prediction = model.predict([[tenure, monthly_charges]])
    st.write("Churn Risk:", prediction)
