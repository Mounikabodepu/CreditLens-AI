
from ai_copilot_service import generate_copilot_response


test_context = """
CreditLens model information:
Dataset: Taiwan Credit Card Default
Prediction target: Default next month

Illustrative borrower evidence:
Predicted probability: 95.06%
Latest repayment delay code: 2
Maximum delay in six months: 8
Number of delayed months: 4

Important:
This is an illustrative test example, not a live borrower record.
"""


question = "Explain why this borrower may be considered high risk."

answer = generate_copilot_response(
    question=question,
    context=test_context,
)

print("\n========== CREDITLENS AI COPILOT ==========\n")
print(answer)
print("\n===========================================\n")
