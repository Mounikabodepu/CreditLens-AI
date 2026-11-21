
import pandas as pd

# 1. Load our prepared dataset
df = pd.read_csv(
    "data/processed/credit_card_default_named.csv"
)

# 2. Identify repayment, bill and payment columns
repayment_columns = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

bill_columns = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]

payment_columns = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# 3. Count months with delayed repayments
# Positive repayment codes indicate months of delay.
df["DELAY_MONTH_COUNT_6M"] = (
    df[repayment_columns] > 0
).sum(axis=1)

# 4. Calculate the maximum recorded delay
# Negative codes are treated as zero delay for this feature.
df["MAX_DELAY_MONTHS_6M"] = (
    df[repayment_columns].clip(lower=0).max(axis=1)
)

# 5. Capture the latest month's delay
df["LATEST_DELAY_MONTHS"] = (
    df["PAY_0"].clip(lower=0)
)

# 6. Aggregate six months of bill amounts
df["TOTAL_BILL_AMT_6M"] = (
    df[bill_columns].sum(axis=1)
)

# 7. Aggregate six months of payment amounts
df["TOTAL_PAY_AMT_6M"] = (
    df[payment_columns].sum(axis=1)
)

# 8. Display newly created features
new_features = [
    "DELAY_MONTH_COUNT_6M",
    "MAX_DELAY_MONTHS_6M",
    "LATEST_DELAY_MONTHS",
    "TOTAL_BILL_AMT_6M",
    "TOTAL_PAY_AMT_6M"
]

print("\nNEW CREDIT RISK FEATURES")
print(df[new_features].head(10))

print("\nFEATURE SUMMARY")
print(df[new_features].describe())

# 9. Save without overwriting the original dataset
output_path = "data/processed/credit_card_default_features.csv"

df.to_csv(output_path, index=False)

print(f"\nFeature engineering completed.")
print(f"Saved dataset to: {output_path}")
print(f"Final shape: {df.shape}")
