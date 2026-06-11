import streamlit as st


def display_model_metrics():

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "86%")

    with col2:
        st.metric("Precision", "82%")

    with col3:
        st.metric("ROC-AUC", "0.85")