
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

print("Loading tuned Random Forest model...")

model = joblib.load("models/creditlens_random_forest_tuned.joblib")

print("Loading test dataset...")

df = pd.read_csv("data/processed/test_data.csv")

X = df.drop(columns=["DEFAULT_NEXT_MONTH"])

# Select a sample of 500 borrowers for explanation
X_sample = X.sample(
    n=min(500, len(X)),
    random_state=42
)

print("Transforming borrower data...")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()

# Convert transformed data into a DataFrame
X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X_sample.index
)

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(X_transformed_df)

# Extract SHAP values for the default class (class 1)
if isinstance(shap_values, list):
    default_shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    default_shap_values = shap_values[:, :, 1]
else:
    default_shap_values = shap_values

# Create reports directory if needed
import os
os.makedirs("reports", exist_ok=True)

# 1. Global feature impact: beeswarm plot
print("Creating SHAP summary plot...")

shap.summary_plot(
    default_shap_values,
    X_transformed_df,
    show=False,
    max_display=20
)

plt.tight_layout()
plt.savefig(
    "reports/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 2. Global feature importance: bar plot
print("Creating SHAP feature importance plot...")

shap.summary_plot(
    default_shap_values,
    X_transformed_df,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()
plt.savefig(
    "reports/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP ANALYSIS COMPLETED")
print("Saved:")
print("reports/shap_summary.png")
print("reports/shap_feature_importance.png")
