import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.model import get_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/churn.csv")

# -----------------------------
# 2. Define features (IMPORTANT)
# -----------------------------
FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

TARGET = "Exited"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# 3. Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train model
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", get_model())
])

pipeline.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2))
print("Precision:", round(precision_score(y_test, y_pred)*100, 2))
print("Recall:", round(recall_score(y_test, y_pred)*100, 2))
print("F1 Score:", round(f1_score(y_test, y_pred)*100, 2))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 2))

# -----------------------------
# 6. Save EVERYTHING (KEY FIX)
# -----------------------------
joblib.dump({
    "model": pipeline,
    "features": FEATURES
}, "model.joblib")

print("✅ Model + features saved successfully")