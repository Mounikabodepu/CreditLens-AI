
from borrower_context_service import get_borrower_context


# Select the first record in the test dataset
result = get_borrower_context(row_position=0)

print("\n========== BORROWER CONTEXT TEST ==========")

print("\nBorrower reference:")
print(result["borrower_reference"])

print("\nPredicted default probability:")
print(f'{result["default_probability"]:.2%}')

print("\nBorrower profile:")
for feature, value in result["profile"].items():
    print(f"{feature}: {value}")

print("\nTop SHAP contributions:")
print(
    result["shap_contributions"].head(10).to_string(index=False)
)

print("\nGenAI context:")
print(result["context"])

print("\n===========================================")
