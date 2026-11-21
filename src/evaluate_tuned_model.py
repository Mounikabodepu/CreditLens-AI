
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

print("Loading tuned Random Forest model...")

model = joblib.load("models/creditlens_random_forest_tuned.joblib")

print("Loading test dataset...")

test_data = pd.read_csv("data/processed/test_data.csv")

X_test = test_data.drop(columns=["DEFAULT_NEXT_MONTH"])
y_test = test_data["DEFAULT_NEXT_MONTH"]

print("\nGenerating predictions...")

# Probability of next-month default
y_probability = model.predict_proba(X_test)[:, 1]

# Default classification threshold = 0.50
y_predicted = (y_probability >= 0.50).astype(int)

print("\nTUNED RANDOM FOREST - TEST RESULTS")

print(f"Accuracy:  {accuracy_score(y_test, y_predicted):.4f}")
print(f"Precision: {precision_score(y_test, y_predicted):.4f}")
print(f"Recall:    {recall_score(y_test, y_predicted):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_predicted):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_probability):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_probability):.4f}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_predicted))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_predicted))

# Save predictions for later analysis
results = pd.DataFrame({
    "ACTUAL_DEFAULT": y_test,
    "PREDICTED_DEFAULT": y_predicted,
    "PREDICTED_DEFAULT_PROBABILITY": y_probability
})

results.to_csv(
    "reports/tuned_random_forest_test_predictions.csv",
    index=False
)

print("\nPredictions saved:")
print("reports/tuned_random_forest_test_predictions.csv")
