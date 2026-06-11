import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Load processed data
df = pd.read_csv("data/processed/cleaned_churn.csv")

# Features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load scaler
scaler = joblib.load("models/scaler.pkl")

X_test = scaler.transform(X_test)

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
}

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    else:
        roc = None

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc
    })

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved.")