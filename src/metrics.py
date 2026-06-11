import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv(
    "data/processed/cleaned_churn.csv"
)

# ----------------------------
# SPLIT DATA
# ----------------------------

X = df.drop(
    "Exited",
    axis=1
)

y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------
# LOAD SCALER
# ----------------------------

scaler = joblib.load(
    "models/scaler.pkl"
)

X_test = scaler.transform(
    X_test
)

# ----------------------------
# LOAD MODEL
# ----------------------------

model = joblib.load(
    "models/random_forest.pkl"
)

# ----------------------------
# PREDICTIONS
# ----------------------------

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:,1]

# ----------------------------
# METRICS
# ----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

# ----------------------------
# PRINT
# ----------------------------

print("\nMODEL PERFORMANCE\n")

print(
    "Accuracy:",
    round(
        accuracy*100,
        2
    ),
    "%"
)

print(
    "Precision:",
    round(
        precision*100,
        2
    ),
    "%"
)

print(
    "Recall:",
    round(
        recall*100,
        2
    ),
    "%"
)

print(
    "F1 Score:",
    round(
        f1*100,
        2
    ),
    "%"
)

print(
    "ROC-AUC:",
    round(
        roc_auc,
        2
    )
)