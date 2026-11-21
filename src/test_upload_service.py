
import pandas as pd

from upload_service import (
    REQUIRED_COLUMNS,
    validate_dataset,
    build_model_features,
    analyse_uploaded_dataset
)


# Use existing test records as a safe demonstration upload.
source = pd.read_csv(
    "data/processed/test_data.csv"
).head(20)

# Simulate a new-borrower upload:
# remove the historical outcome column.
upload = source[REQUIRED_COLUMNS].copy()

print("\n1. VALIDATING UPLOAD")

validation = validate_dataset(upload)
print(validation)

assert validation["valid"] is True

print("\n2. BUILDING FEATURES")

features = build_model_features(upload)

print("Input columns:", len(upload.columns))
print("Model feature columns:", len(features.columns))
print("Engineered columns:")
print(features[[
    "DELAY_MONTH_COUNT_6M",
    "MAX_DELAY_MONTHS_6M",
    "LATEST_DELAY_MONTHS",
    "TOTAL_BILL_AMT_6M",
    "TOTAL_PAY_AMT_6M"
]].head())

assert len(features.columns) == 28

print("\n3. RUNNING MODEL")

analysis = analyse_uploaded_dataset(
    upload,
    "creditlens_demo_upload.csv"
)

print("Run ID:", analysis["run_id"])
print("Records analysed:", analysis["record_count"])
print("Predicted defaults:", analysis["predicted_defaults"])
print(
    "Average predicted probability:",
    f"{analysis['average_probability']:.2%}"
)

print("\n4. PREDICTION PREVIEW")

print(analysis["results"].head())

print("\nUPLOAD SERVICE TEST COMPLETED")
