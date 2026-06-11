import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================
# LOAD MODELS
# =====================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

models = {
    "Logistic Regression": joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    ),
    "Decision Tree": joblib.load(
        MODEL_DIR / "decision_tree.pkl"
    ),
    "Random Forest (Best Model)": joblib.load(
        MODEL_DIR / "random_forest.pkl"
    )
}

scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)

# =====================
# HEADER
# =====================

st.title("📊 Customer Churn Prediction")

st.markdown("""
Predict whether a customer is likely to churn using Machine Learning.

### Models Available
- Logistic Regression
- Decision Tree
- Random Forest ⭐ Recommended
""")

# =====================
# MODEL SELECTION
# =====================

selected_model = st.selectbox(
    "Choose Prediction Model",
    list(models.keys())
)

model = models[selected_model]

# =====================
# INPUTS
# =====================

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        300,
        900,
        600
    )

    age = st.number_input(
        "Age",
        18,
        100,
        30
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        3
    )

    balance = st.number_input(
        "Balance",
        0.0,
        250000.0,
        50000.0
    )

with col2:

    num_products = st.slider(
        "Number of Products",
        1,
        4,
        1
    )

    has_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    is_active = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    salary = st.number_input(
        "Estimated Salary",
        0.0,
        200000.0,
        50000.0
    )

# =====================
# PREPARE INPUT
# =====================

input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary,

    # Dummy columns from encoding
    "Geography_Germany": 0,
    "Geography_Spain": 0,
    "Gender_Male": 0
}])

input_scaled = scaler.transform(input_df)

# =====================
# PREDICT
# =====================

if st.button("🔮 Predict"):

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ Customer Likely To Churn ({probability*100:.2f}%)"
        )

    else:

        st.success(
            f"✅ Customer Likely To Stay ({(1-probability)*100:.2f}%)"
        )

    # =====================
    # PROBABILITY
    # =====================

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    # =====================
    # RISK LEVEL
    # =====================

    if probability < 0.30:

        st.success("🟢 Low Risk")

    elif probability < 0.70:

        st.warning("🟡 Medium Risk")

    else:

        st.error("🔴 High Risk")

    # =====================
    # RECOMMENDATIONS
    # =====================

    if probability >= 0.70:

        st.markdown("""
        ### Recommended Retention Actions

        - Offer loyalty rewards
        - Personalized discounts
        - Customer support follow-up
        - Exclusive banking benefits
        """)

# =====================
# MODEL COMPARISON
# =====================

st.markdown("---")

st.subheader("📈 Model Comparison")

comparison_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.81,
        0.79,
        0.86
    ],
    "ROC-AUC": [
        0.84,
        0.80,
        0.87
    ]
})

st.dataframe(
    comparison_data,
    use_container_width=True
)

st.success(
    "⭐ Random Forest selected as final deployment model due to highest Accuracy and ROC-AUC."
)


# Three machine learning algorithms were evaluated for customer churn prediction. 
# Random Forest achieved the highest Accuracy (86.65%), Precision (76.25%), F1 Score (57.82%), and ROC-AUC (0.8653), making it the best-performing model. 
# Therefore, Random Forest was selected as the final deployment model for the Streamlit application.