
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# 1. Load training data only
df = pd.read_csv("data/processed/train_data.csv")

TARGET = "DEFAULT_NEXT_MONTH"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 2. Create an internal training/validation split
X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Identify feature types
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# 4. Prepare preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# 5. Define three candidate algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

# 6. Train and evaluate each model
for name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_fit, y_fit)

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_val, probabilities),
        "PR_AUC": average_precision_score(y_val, probabilities)
    }

    results.append(metrics)

    # Save each fitted candidate
    filename = name.lower().replace(" ", "_")
    joblib.dump(
        pipeline,
        f"models/{filename}_candidate.joblib"
    )

    print(f"{name} completed.")

# 7. Compare results
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS")
print("=" * 90)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nComparison saved to reports/model_comparison.csv")
