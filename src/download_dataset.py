
from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Find the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Create the raw data folder
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Connecting to UCI Machine Learning Repository...")

# Download the Default of Credit Card Clients dataset
dataset = fetch_ucirepo(id=350)

# Separate input features and target
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Combine them into one dataframe
target_column = y.columns[0]

df = X.copy()
df[target_column] = y[target_column].values

# Save original dataset
output_file = RAW_DIR / "credit_card_default.csv"
df.to_csv(output_file, index=False)

print("\nDataset downloaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df[target_column].value_counts())

print("\nFirst five records:")
print(df.head())

print("\nFile saved at:", output_file)
