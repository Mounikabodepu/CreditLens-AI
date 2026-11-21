
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("CREDITLENS AI - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# 3. RENAME COLUMNS
# --------------------------------------------------

column_mapping = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": "DEFAULT_NEXT_MONTH"
}

df.rename(columns=column_mapping, inplace=True)

# Confirm required target exists
target = "DEFAULT_NEXT_MONTH"

if target not in df.columns:
    raise ValueError("Target column is missing. Check the dataset.")

# --------------------------------------------------
# 4. DATASET OVERVIEW
# --------------------------------------------------

print("\n1. DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. DATA TYPES")
print(df.info())

print("\n4. FIRST FIVE RECORDS")
print(df.head())

# --------------------------------------------------
# 5. DATA QUALITY CHECKS
# --------------------------------------------------

print("\n5. MISSING VALUES")

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0])

print("\n6. DUPLICATE RECORDS")

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

print("\n7. TARGET VALIDATION")

print("Unique target values:", sorted(df[target].unique()))

if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError("Unexpected target values found.")

# --------------------------------------------------
# 6. STATISTICAL ANALYSIS
# --------------------------------------------------

print("\n8. STATISTICAL SUMMARY")

print(df.describe().T.round(2))

# --------------------------------------------------
# 7. DEFAULT ANALYSIS
# --------------------------------------------------

print("\n9. DEFAULT DISTRIBUTION")

target_counts = df[target].value_counts().sort_index()

print(target_counts.rename(index={
    0: "Non-default",
    1: "Default"
}))

print("\n10. DEFAULT PERCENTAGES")

target_percentages = (
    df[target].value_counts(normalize=True).sort_index() * 100
)

print(target_percentages.rename(index={
    0: "Non-default",
    1: "Default"
}).round(2))

default_rate = df[target].mean() * 100

print(f"\nOverall historical default rate: {default_rate:.2f}%")

# --------------------------------------------------
# 8. BORROWER PROFILE COMPARISON
# --------------------------------------------------

print("\n11. CUSTOMER PROFILE BY DEFAULT STATUS")

comparison_columns = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1"
]

comparison = df.groupby(target)[comparison_columns].mean().round(2)

comparison.index = comparison.index.map({
    0: "Non-default",
    1: "Default"
})

print(comparison)

# --------------------------------------------------
# 9. SAVE RENAMED DATASET
# --------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DIR / "credit_card_default_named.csv"

df.to_csv(output_path, index=False)

print("\n12. OUTPUT")
print("Renamed dataset saved to:", output_path)

print("\nEDA COMPLETED SUCCESSFULLY!")
