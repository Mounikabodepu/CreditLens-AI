
import pandas as pd
import joblib

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)

# 1. Load training data only
df = pd.read_csv("data/processed/train_data.csv")

TARGET = "DEFAULT_NEXT_MONTH"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 2. Identify feature types
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# 3. Preprocessing
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

# 4. Random Forest pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=1
            )
        )
    ]
)

# 5. Hyperparameter search space
parameter_grid = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [8, 12, 16, None],
    "classifier__min_samples_split": [2, 10, 20],
    "classifier__min_samples_leaf": [5, 10, 20],
    "classifier__max_features": ["sqrt", 0.5]
}

# 6. Stratified cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=5,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# 7. Find the best configuration
print("Starting Random Forest hyperparameter tuning...")

search.fit(X, y)

best_model = search.best_estimator_

print("\nBEST PARAMETERS")
print(search.best_params_)

print("\nBEST CROSS-VALIDATION PR-AUC")
print(round(search.best_score_, 4))

# 8. Save the tuned model
joblib.dump(
    best_model,
    "models/creditlens_random_forest_tuned.joblib"
)

# 9. Save experiment results
results = pd.DataFrame(search.cv_results_)

results.to_csv(
    "reports/random_forest_tuning_results.csv",
    index=False
)

print("\nTUNING COMPLETED")
print("Best model saved:")
print("models/creditlens_random_forest_tuned.joblib")
