
from creditlens_copilot import ask_borrower_question


result = ask_borrower_question(
    row_position=0,
    question="Why does this borrower have a 35.76% predicted default probability, even though there are no recorded repayment delays?"
)

print("\n========== CREDITLENS AI COPILOT ==========")

print("\nBorrower:", result["borrower_reference"])

print(
    "\nPredicted default probability:",
    f'{result["default_probability"]:.2%}'
)

print("\nAI COPILOT RESPONSE:\n")
print(result["answer"])

print("\n===========================================")
