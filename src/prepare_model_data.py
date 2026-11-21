
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the feature-engineered dataset
df = pd.read_csv(
    "data/processed/credit_card_default_features.csv"
)

target_column = "DEFAULT_NEXT_MONTH"

# 2. Check exact duplicate records
duplicate_count = df.duplicated().sum()
print(f"Exact duplicate records found: {duplicate_count}")

# Remove exact duplicates from the modelling copy.
# The original dataset remains unchanged.
df = df.drop_duplicates().copy()

print(f"Records after duplicate handling: {len(df)}")

# 3. Separate input features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# 4. Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Recombine features and target for saving
train_data = X_train.copy()
train_data[target_column] = y_train

test_data = X_test.copy()
test_data[target_column] = y_test

# 6. Save the datasets
train_data.to_csv(
    "data/processed/train_data.csv",
    index=False
)

test_data.to_csv(
    "data/processed/test_data.csv",
    index=False
)

# 7. Display results
print("\nDATA SPLIT SUMMARY")
print("------------------------")
print(f"Training records: {len(train_data)}")
print(f"Testing records: {len(test_data)}")
print(f"Input features: {X.shape[1]}")

print("\nTraining target distribution:")
print(y_train.value_counts())
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting target distribution:")
print(y_test.value_counts())
print((y_test.value_counts(normalize=True) * 100).round(2))

print("\nData preparation completed successfully.")
