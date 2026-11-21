
from pathlib import Path
from datetime import datetime, timezone
import uuid
import json

import joblib
import numpy as np
import pandas as pd

from database.repository import (
    initialize_database,
    get_analysis_history,
    get_analysis_run,
    get_user_analysis_run,
    get_borrower_predictions,
    save_analysis_run
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "creditlens_random_forest_tuned.joblib"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
RESULT_DIR = PROJECT_ROOT / "data" / "analysis_results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. EXPECTED INPUT SCHEMA
# ---------------------------------------------------------

REQUIRED_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

TARGET_COLUMN = "DEFAULT_NEXT_MONTH"

REPAYMENT_COLUMNS = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

BILL_COLUMNS = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]

PAYMENT_COLUMNS = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

ENGINEERED_COLUMNS = [
    "DELAY_MONTH_COUNT_6M",
    "MAX_DELAY_MONTHS_6M",
    "LATEST_DELAY_MONTHS",
    "TOTAL_BILL_AMT_6M",
    "TOTAL_PAY_AMT_6M"
]


# ---------------------------------------------------------
# 2. LOAD COMPLETE SAVED ANALYSIS RESULTS
# ---------------------------------------------------------

def load_analysis_results(run_id, user_id):
    """Load saved results only if the analysis belongs to this user."""

    analysis = get_user_analysis_run(run_id, user_id)

    if analysis is None:
        raise ValueError("Analysis run not found.")

    result_path = RESULT_DIR / analysis["result_filename"]

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Saved prediction file is missing: {result_path.name}"
        )

    return pd.read_csv(result_path)


# ---------------------------------------------------------
# 3. READ UPLOADED FILE
# ---------------------------------------------------------

def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            data = pd.read_csv(uploaded_file)

        elif filename.endswith((".xlsx", ".xls")):
            data = pd.read_excel(uploaded_file)

        else:
            raise ValueError(
                "Unsupported file format. Upload CSV or Excel."
            )

    except Exception as error:
        raise ValueError(
            f"Could not read the uploaded file: {error}"
        ) from error

    if data.empty:
        raise ValueError("The uploaded file contains no records.")

    return data


# ---------------------------------------------------------
# 4. VALIDATE DATASET
# ---------------------------------------------------------

def validate_dataset(data):
    errors = []
    warnings = []

    if data.empty:
        errors.append("The uploaded dataset is empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "record_count": len(data)
        }

    if TARGET_COLUMN in data.columns:
        target = pd.to_numeric(
            data[TARGET_COLUMN], errors="coerce"
        )

        if target.isna().any():
            errors.append(
                "DEFAULT_NEXT_MONTH contains missing or non-numeric values."
            )

        elif not set(target.unique()).issubset({0, 1}):
            errors.append(
                "DEFAULT_NEXT_MONTH must contain only 0 or 1."
            )

    numeric_columns = REQUIRED_COLUMNS

    for column in numeric_columns:
        converted = pd.to_numeric(
            data[column], errors="coerce"
        )

        invalid_count = int(converted.isna().sum())

        if invalid_count > 0:
            errors.append(
                f"{column} contains {invalid_count} missing "
                "or non-numeric values."
            )

    if data.duplicated().any():
        warnings.append(
            f"{int(data.duplicated().sum())} exact duplicate rows found."
        )

    extra_columns = [
        column for column in data.columns
        if column not in REQUIRED_COLUMNS + [TARGET_COLUMN]
    ]

    if extra_columns:
        warnings.append(
            "Additional columns will be ignored by the model: "
            + ", ".join(extra_columns[:15])
        )

    if len(data) < 1:
        errors.append("At least one borrower record is required.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "record_count": len(data),
        "missing_columns": missing_columns
    }


# ---------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------

def build_model_features(data):
    """
    Create the same five engineered features used during
    CreditLens model development.
    """

    features = data[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        features[column] = pd.to_numeric(
            features[column], errors="raise"
        )

    # Count months with a positive repayment-delay code.
    features["DELAY_MONTH_COUNT_6M"] = (
        features[REPAYMENT_COLUMNS] > 0
    ).sum(axis=1)

    # Highest positive repayment-delay code.
    features["MAX_DELAY_MONTHS_6M"] = (
        features[REPAYMENT_COLUMNS]
        .clip(lower=0)
        .max(axis=1)
    )

    # Most recent repayment-delay code.
    features["LATEST_DELAY_MONTHS"] = (
        features["PAY_0"].clip(lower=0)
    )

    # Six-period historical sums.
    features["TOTAL_BILL_AMT_6M"] = (
        features[BILL_COLUMNS].sum(axis=1)
    )

    features["TOTAL_PAY_AMT_6M"] = (
        features[PAYMENT_COLUMNS].sum(axis=1)
    )

    return features


# ---------------------------------------------------------
# 6. RISK CATEGORIES
# ---------------------------------------------------------

def assign_risk_category(probability):
    if probability < 0.30:
        return "Low"

    elif probability < 0.60:
        return "Medium"

    return "High"


# ---------------------------------------------------------
# 7. RUN MODEL ANALYSIS
# ---------------------------------------------------------

def analyse_uploaded_dataset(data, original_filename, owner_id=None):

    # =====================================================
    # 1. VALIDATE UPLOADED DATA
    # =====================================================

    validation = validate_dataset(data)

    if not validation["valid"]:
        raise ValueError(
            "\n".join(validation["errors"])
        )

    # =====================================================
    # 2. LOAD MODEL AND GENERATE FEATURES
    # =====================================================

    model = joblib.load(MODEL_PATH)

    # 23 original columns + 5 engineered columns
    features = build_model_features(data)

    # Generate next-month default probabilities.
    probabilities = model.predict_proba(features)[:, 1]

    # Apply illustrative 50% prediction threshold.
    predictions = (probabilities >= 0.50).astype(int)

    # =====================================================
    # 3. BUILD COMPLETE BORROWER RESULTS
    # =====================================================

    results = features.copy().reset_index(drop=True)

    results.insert(
        0,
        "BORROWER_REFERENCE",
        [
            f"UPLOAD-{i + 1:06d}"
            for i in range(len(data))
        ]
    )

    results["PREDICTED_DEFAULT_PROBABILITY"] = probabilities
    results["PREDICTED_DEFAULT"] = predictions

    results["RISK_CATEGORY"] = [
        assign_risk_category(value)
        for value in probabilities
    ]

    # =====================================================
    # 4. PRESERVE OPTIONAL ACTUAL OUTCOMES
    # =====================================================

    target_available = TARGET_COLUMN in data.columns

    if target_available:
        target_values = pd.to_numeric(
            data[TARGET_COLUMN],
            errors="coerce"
        ).reset_index(drop=True)

        results[TARGET_COLUMN] = target_values

    # =====================================================
    # 5. GENERATE UNIQUE FILE REFERENCES
    # =====================================================

    run_id = str(uuid.uuid4())

    safe_filename = f"{run_id}.csv"
    result_filename = f"{run_id}_predictions.csv"

    stored_path = UPLOAD_DIR / safe_filename
    result_path = RESULT_DIR / result_filename

    # =====================================================
    # 6. STORE ORIGINAL DATA AND PREDICTION RESULTS
    # =====================================================

    data.to_csv(
        stored_path,
        index=False
    )

    results.to_csv(
        result_path,
        index=False
    )

    # =====================================================
    # 7. CALCULATE PORTFOLIO SUMMARY
    # =====================================================

    actual_default_rate = (
        float(results[TARGET_COLUMN].mean())
        if target_available
        else None
    )

    uploaded_at = datetime.now(
        timezone.utc
    ).isoformat()

    predicted_defaults = int(predictions.sum())

    average_probability = float(
        probabilities.mean()
    )

    # =====================================================
    # 8. PREPARE BORROWER PREDICTION RECORDS
    # =====================================================

    prediction_rows = []

    for row in results.itertuples(index=False):

        actual_outcome = (
            int(getattr(row, TARGET_COLUMN))
            if target_available
            else None
        )

        prediction_rows.append((
            run_id,
            row.BORROWER_REFERENCE,
            float(row.PREDICTED_DEFAULT_PROBABILITY),
            int(row.PREDICTED_DEFAULT),
            str(row.RISK_CATEGORY),
            actual_outcome
        ))

    # =====================================================
    # 9. SAVE ANALYSIS THROUGH DATABASE REPOSITORY
    # =====================================================

    save_analysis_run(
    run_id=run_id,
    original_filename=original_filename,
    stored_filename=safe_filename,
    uploaded_at=uploaded_at,
    record_count=len(data),
    predicted_defaults=predicted_defaults,
    average_probability=average_probability,
    model_name="Random Forest",
    model_version="creditlens_random_forest_tuned_v1",
    target_available=target_available,
    actual_default_rate=actual_default_rate,
    result_filename=result_filename,
    prediction_rows=prediction_rows,
    owner_id=owner_id
)

    # =====================================================
    # 10. RETURN ANALYSIS RESULTS TO STREAMLIT
    # =====================================================

    return {
        "run_id": run_id,
        "validation": validation,
        "results": results,
        "features": features,
        "record_count": len(data),
        "predicted_defaults": predicted_defaults,
        "average_probability": average_probability,
        "target_available": target_available,
        "actual_default_rate": actual_default_rate,
        "stored_path": str(stored_path),
        "result_path": str(result_path)
    }
