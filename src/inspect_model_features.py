
import joblib
import pandas as pd
from pathlib import Path

print("Loading tuned Random Forest model...")

model = joblib.load("models/creditlens_random_forest_tuned.joblib")

# Extract preprocessing and Random Forest components
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get Random Forest feature importance scores
importance_scores = classifier.feature_importances_

# Create a readable table
importance_df = pd.DataFrame({
    "FEATURE": feature_names,
    "IMPORTANCE": importance_scores
})

importance_df = importance_df.sort_values(
    by="IMPORTANCE",
    ascending=False
)

# Display top 20 features
print("\nTOP 20 FEATURES INFLUENCING MODEL PREDICTIONS")
print(importance_df.head(20).to_string(index=False))

# Save full results
Path("reports").mkdir(exist_ok=True)

importance_df.to_csv(
    "reports/random_forest_feature_importance.csv",
    index=False
)

print("\nFull feature importance report saved:")
print("reports/random_forest_feature_importance.csv")
