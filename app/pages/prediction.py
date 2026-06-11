import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Customer Churn Prediction")

# Try to retrieve input dataframe from session state or globals
input_df = None
if "input_df" in st.session_state:
    input_df = st.session_state["input_df"]
elif "input_df" in globals():
    input_df = globals()["input_df"]

if input_df is None:
    st.error("No input data available to make a prediction.")
else:
    # Ensure input is a DataFrame and scaled if scaler exists
    if not isinstance(input_df, pd.DataFrame):
        try:
            input_df = pd.DataFrame([input_df])
        except Exception:
            st.error("Invalid input data format for prediction.")
            st.stop()

    try:
        if 'scaler' in globals() and scaler is not None:
            X = scaler.transform(input_df)
        else:
            X = input_df.values
        probability = model.predict_proba(X)[0][1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    if probability < 0.3:
        st.success("🟢 Low Risk Customer")

    elif probability < 0.7:
        st.warning("🟡 Medium Risk Customer")

    else:
        st.error("🔴 High Risk Customer")

    if probability > 0.7:
        st.markdown("""
        ### Recommended Actions

        - Contact customer
        - Offer retention benefits
        - Personalized marketing campaign
        - Loyalty rewards
        """)


