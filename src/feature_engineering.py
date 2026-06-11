import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed/cleaned_churn.csv")

print("Original Shape:", df.shape)

# ----------------------------------
# FEATURE 1: AGE GROUP
# ----------------------------------

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]
)

# ----------------------------------
# FEATURE 2: BALANCE CATEGORY
# ----------------------------------

df["BalanceCategory"] = pd.cut(
    df["Balance"],
    bins=[-1, 0, 50000, 100000, 300000],
    labels=[0, 1, 2, 3]
)

# ----------------------------------
# FEATURE 3: TENURE CATEGORY
# ----------------------------------

df["TenureGroup"] = pd.cut(
    df["Tenure"],
    bins=[-1, 2, 5, 10],
    labels=[0, 1, 2]
)

# ----------------------------------
# FEATURE 4: HIGH VALUE CUSTOMER
# ----------------------------------

df["HighValueCustomer"] = (
    (df["Balance"] > 100000)
    & (df["EstimatedSalary"] > 100000)
).astype(int)

# ----------------------------------
# FEATURE 5: ENGAGEMENT SCORE
# ----------------------------------

df["EngagementScore"] = (
    df["NumOfProducts"]
    + df["HasCrCard"]
    + df["IsActiveMember"]
)

# Convert category columns to integer
category_cols = [
    "AgeGroup",
    "BalanceCategory",
    "TenureGroup"
]

for col in category_cols:
    df[col] = df[col].astype(int)

print("New Shape:", df.shape)

print(df.head())

# Save feature engineered dataset
df.to_csv(
    "data/processed/feature_engineered_churn.csv",
    index=False
)

print("\nFeature engineered dataset saved.")