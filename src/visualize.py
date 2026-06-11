import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("data/processed/cleaned_churn.csv")

# ----------------------------
# CHURN DISTRIBUTION
# ----------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="Exited",
    data=df
)

plt.title("Customer Churn Distribution")

plt.savefig(
    "reports/figures/churn_distribution.png"
)

plt.close()

# ----------------------------
# CORRELATION HEATMAP
# ----------------------------

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.savefig(
    "reports/figures/heatmap.png"
)

plt.close()

# ----------------------------
# AGE VS CHURN
# ----------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Exited",
    y="Age",
    data=df
)

plt.title("Age vs Customer Churn")

plt.savefig(
    "reports/figures/age_vs_churn.png"
)

plt.close()

# ----------------------------
# MODEL EVALUATION
# ----------------------------

X = df.drop("Exited", axis=1)

y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

scaler = joblib.load("models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)

rf = joblib.load(
    "models/random_forest.pkl"
)

# ----------------------------
# CONFUSION MATRIX
# ----------------------------

y_pred = rf.predict(X_test_scaled)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    "reports/figures/confusion_matrix.png"
)

plt.close()

# ----------------------------
# ROC CURVE
# ----------------------------

y_prob = rf.predict_proba(
    X_test_scaled
)[:,1]

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6,4))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "reports/figures/roc_curve.png"
)

plt.close()

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Important Features"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_importance.png"
)

plt.close()

print("All visualizations generated successfully.")