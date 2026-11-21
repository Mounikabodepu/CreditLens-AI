
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

print("Loading CreditLens model...")

model = joblib.load("models/creditlens_random_forest_tuned.joblib")

df = pd.read_csv("data/processed/test_data.csv")

X = df.drop(columns=["DEFAULT_NEXT_MONTH"])
y = df["DEFAULT_NEXT_MONTH"]

# Use the same reproducible sample as our global SHAP analysis
X_sample = X.sample(
    n=min(500, len(X)),
    random_state=42
)

y_sample = y.loc[X_sample.index]

# Get predicted probability of next-month default
probabilities = model.predict_proba(X_sample)[:, 1]

# Select the borrower with the highest predicted risk
selected_position = int(np.argmax(probabilities))
selected_index = X_sample.index[selected_position]

selected_probability = probabilities[selected_position]
actual_outcome = y_sample.loc[selected_index]

print("\nSELECTED BORROWER")
print("Dataset row index:", selected_index)
print(f"Predicted default probability: {selected_probability:.2%}")
print("Actual next-month default:", actual_outcome)

# Transform data using the model's existing preprocessing
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X_sample.index
)

# Calculate SHAP values
print("\nCalculating individual SHAP explanation...")

explainer = shap.TreeExplainer(classifier)

raw_shap_values = explainer.shap_values(X_transformed_df)

expected_value = np.asarray(explainer.expected_value)

if isinstance(raw_shap_values, list):
    default_shap_values = raw_shap_values[1]
elif np.asarray(raw_shap_values).ndim == 3:
    default_shap_values = raw_shap_values[:, :, 1]
else:
    default_shap_values = raw_shap_values

if expected_value.ndim > 0:
    base_value = float(expected_value[1])
else:
    base_value = float(expected_value)

# Build explanation for the selected borrower
explanation = shap.Explanation(
    values=default_shap_values[selected_position],
    base_values=base_value,
    data=X_transformed_df.iloc[selected_position].values,
    feature_names=list(feature_names)
)

# Generate waterfall plot
os.makedirs("reports", exist_ok=True)

shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    "reports/individual_borrower_shap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nINDIVIDUAL BORROWER EXPLANATION COMPLETED")
print("Saved: reports/individual_borrower_shap.png")

# --------------------------------------------------
# Generate a readable individual borrower explanation
# --------------------------------------------------

contribution_df = pd.DataFrame({
    "FEATURE": feature_names,
    "SHAP_CONTRIBUTION": default_shap_values[selected_position],
    "TRANSFORMED_FEATURE_VALUE": X_transformed_df.iloc[
        selected_position
    ].values
})

# Positive SHAP = pushes towards default
# Negative SHAP = pushes away from default

risk_increasing = contribution_df[
    contribution_df["SHAP_CONTRIBUTION"] > 0
].sort_values(
    by="SHAP_CONTRIBUTION",
    ascending=False
)

risk_reducing = contribution_df[
    contribution_df["SHAP_CONTRIBUTION"] < 0
].sort_values(
    by="SHAP_CONTRIBUTION",
    ascending=True
)

print("\nTOP 10 FACTORS INCREASING DEFAULT PREDICTION")
print(
    risk_increasing.head(10).to_string(index=False)
)

print("\nTOP 10 FACTORS REDUCING DEFAULT PREDICTION")
print(
    risk_reducing.head(10).to_string(index=False)
)

# Save all individual contributions
contribution_df.to_csv(
    "reports/individual_borrower_contributions.csv",
    index=False
)

# Save the original borrower input values
X_sample.loc[[selected_index]].to_csv(
    "reports/individual_borrower_profile.csv",
    index=False
)

print("\nAdditional reports saved:")
print("reports/individual_borrower_contributions.csv")
print("reports/individual_borrower_profile.csv")
