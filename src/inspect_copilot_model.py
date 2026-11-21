
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "creditlens_random_forest_tuned.joblib"

model = joblib.load(MODEL_PATH)

print("\n========== CREDITLENS MODEL INSPECTION ==========")

print("\nPipeline:")
print(model)

print("\nPipeline steps:")
if hasattr(model, "named_steps"):
    for name, step in model.named_steps.items():
        print(f"{name}: {type(step).__name__}")

print("\nModel classes:")
print(model.classes_)

print("\n=================================================")
