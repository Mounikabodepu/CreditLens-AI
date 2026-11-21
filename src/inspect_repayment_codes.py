
import pandas as pd

df = pd.read_csv(
    "data/processed/credit_card_default_named.csv"
)

repayment_columns = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

print("\nREPAYMENT STATUS VALUE COUNTS")

for column in repayment_columns:
    print(f"\n{column}")
    print(df[column].value_counts().sort_index())

print("\nEDUCATION VALUES")
print(df["EDUCATION"].value_counts().sort_index())

print("\nMARRIAGE VALUES")
print(df["MARRIAGE"].value_counts().sort_index())
