# import pandas as pd

# def load_data(path):
#     return pd.read_csv(path)

# def preprocess_data(df):
#     df = df.dropna()

#     # Convert target
#     df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

#     # Encode categorical
#     df = pd.get_dummies(df, drop_first=True)

#     return df


# df = preprocess_data(df)

import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/churn.csv")

# Display first 5 rows
print(df.head())

# -----------------------------
# DROP UNNECESSARY COLUMNS
# -----------------------------

columns_to_drop = ["RowNumber", "CustomerId", "Surname"]

df.drop(columns=columns_to_drop, inplace=True)

print("\nColumns after dropping:")
print(df.columns)

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

print("\nMissing values:")
print(df.isnull().sum())

# If missing values exist
df.dropna(inplace=True)

# -----------------------------
# ENCODE CATEGORICAL VARIABLES
# -----------------------------

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding:")
print(df.head())

bool_columns = df.select_dtypes(include='bool').columns
df[bool_columns] = df[bool_columns].astype(int)

# -----------------------------
# SAVE CLEANED DATASET
# -----------------------------

output_path = "data/processed/cleaned_churn.csv"

df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")