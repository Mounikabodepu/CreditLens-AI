
from pathlib import Path

import joblib
import pandas as pd
import shap


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "creditlens_random_forest_tuned.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

TARGET_COLUMN = "DEFAULT_NEXT_MONTH"


# --------------------------------------------------
# 2. Load the trained model and evaluation dataset
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 3. Helper: calculate SHAP contributions
# --------------------------------------------------

def get_shap_contributions(borrower_features):

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_data = preprocessor.transform(borrower_features)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(transformed_data)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]

    elif getattr(shap_values, "ndim", 0) == 3:
        class_index = list(classifier.classes_).index(1)
        contributions = shap_values[0, :, class_index]

    else:
        contributions = shap_values[0]

    result = pd.DataFrame({
        "feature": feature_names,
        "shap_contribution": contributions
    })

    # Remove preprocessing prefixes for readability
    result["feature"] = (
        result["feature"]
        .str.replace("numerical__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )

    result = result.sort_values(
        "shap_contribution",
        ascending=False
    )

    return result


# --------------------------------------------------
# 4. Main function: build borrower context
# --------------------------------------------------

def get_borrower_context(row_position):

    if row_position < 0 or row_position >= len(data):
        raise IndexError("Invalid borrower row position.")

    # Select one record
    borrower = data.iloc[[row_position]].copy()

    # Remove actual outcome before model prediction
    borrower_features = borrower.drop(
        columns=[TARGET_COLUMN],
        errors="ignore"
    )

    # Predict probability of next-month default
    probabilities = model.predict_proba(borrower_features)[0]

    default_class_index = list(model.classes_).index(1)

    default_probability = float(
        probabilities[default_class_index]
    )

    # Calculate SHAP contributions
    shap_result = get_shap_contributions(borrower_features)

    top_increasing = shap_result.head(5)
    top_reducing = (
        shap_result[shap_result["shap_contribution"] < 0]
        .tail(5)
        .sort_values("shap_contribution")
    )

    # Build readable borrower profile
    profile_columns = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "DELAY_MONTH_COUNT_6M",
        "MAX_DELAY_MONTHS_6M",
        "LATEST_DELAY_MONTHS",
        "TOTAL_BILL_AMT_6M",
        "TOTAL_PAY_AMT_6M",
    ]

    profile = {}

    for column in profile_columns:
        if column in borrower_features.columns:
            profile[column] = borrower_features.iloc[0][column].item() \
                if hasattr(borrower_features.iloc[0][column], "item") \
                else borrower_features.iloc[0][column]

    # Build context for GenAI
    context = f"""
CREDITLENS AI - BORROWER RISK ASSESSMENT

Borrower reference: DEMO-{row_position + 1}

MODEL INFORMATION:
Model: Tuned Random Forest
Prediction target: Default next month
Predicted default probability: {default_probability:.2%}

BORROWER PROFILE:
{profile}

TOP FEATURES CONTRIBUTING POSITIVELY TO MODEL OUTPUT:
{top_increasing.to_string(index=False)}

FEATURES CONTRIBUTING NEGATIVELY TO MODEL OUTPUT:
{top_reducing.to_string(index=False)}

IMPORTANT LIMITATIONS:
- This is a historical evaluation dataset record, not a live customer.
- The probability is a model prediction, not a validated regulatory PD.
- SHAP contributions explain model output, not causation.
- The model supports analyst review; it does not approve or reject credit.
"""

    return {
        "borrower_reference": f"DEMO-{row_position + 1}",
        "default_probability": default_probability,
        "profile": profile,
        "shap_contributions": shap_result,
        "context": context,
    }
