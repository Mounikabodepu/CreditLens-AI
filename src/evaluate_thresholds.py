
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load saved model and test dataset
model = joblib.load(
    "models/creditlens_logistic_regression.joblib"
)

test_df = pd.read_csv(
    "data/processed/test_data.csv"
)

X_test = test_df.drop(columns=["DEFAULT_NEXT_MONTH"])
y_test = test_df["DEFAULT_NEXT_MONTH"]

# Get predicted probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Compare different classification thresholds
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

print("\nTHRESHOLD COMPARISON")
print("=" * 75)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions
    ).ravel()

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")

print("\nThreshold evaluation completed.")
