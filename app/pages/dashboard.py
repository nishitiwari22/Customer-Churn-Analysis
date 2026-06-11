import streamlit as st

st.set_page_config(page_title="Customer Churn Dashboard")

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
This project predicts whether a customer is likely to churn using Machine Learning.

### Models Used
- Logistic Regression
- Decision Tree
- Random Forest

### Dataset
- 10,000+ customer records
- Banking customer data

### Features
- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership
- Estimated Salary
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "10,000")

with col2:
    st.metric("Best Accuracy", "86%")

with col3:
    st.metric("ROC-AUC", "0.85")

st.markdown("---")

st.subheader("Project Workflow")

st.markdown("""
Raw Data → Preprocessing → EDA → Feature Engineering → Model Training → Evaluation → Deployment
""")