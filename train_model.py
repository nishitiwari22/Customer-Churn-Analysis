
import os
import joblib
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, preprocess_data
from src.model import train_model

# 1. Load
df = load_data("data/churn.csv")
# print(df.columns)

# 2. Preprocess
df = preprocess_data(df)

# 3. Split
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train
model = train_model(X_train, y_train)

# 5. Save
joblib.dump(model, "model.joblib")

print("✅ Model saved successfully!")