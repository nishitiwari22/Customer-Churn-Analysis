# from sklearn.linear_model import LogisticRegression

# def train_model(X_train, y_train):
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     return model

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("../data/churn.csv")

print(df.columns)

# Example features (adjust if needed)
X = df[["tenure", "MonthlyCharges"]]
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Save
joblib.dump(model, "model.joblib")

print("✅ Model saved successfully")