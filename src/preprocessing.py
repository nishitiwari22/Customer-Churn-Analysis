import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()

    # Convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical
    df = pd.get_dummies(df, drop_first=True)

    return df