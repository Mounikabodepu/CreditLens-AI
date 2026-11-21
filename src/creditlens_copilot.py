
from borrower_context_service import get_borrower_context
from ai_copilot_service import generate_copilot_response


def ask_borrower_question(row_position, question):
    """
    Generate an evidence-grounded AI explanation
    for a selected CreditLens borrower.
    """

    # 1. Retrieve the actual model evidence
    borrower_data = get_borrower_context(row_position)

    # 2. Send the question and evidence to Gemini
    answer = generate_copilot_response(
        question=question,
        context=borrower_data["context"]
    )

    # 3. Return both evidence and AI response
    return {
        "borrower_reference": borrower_data["borrower_reference"],
        "default_probability": borrower_data["default_probability"],
        "profile": borrower_data["profile"],
        "shap_contributions": borrower_data["shap_contributions"],
        "answer": answer,
    }
