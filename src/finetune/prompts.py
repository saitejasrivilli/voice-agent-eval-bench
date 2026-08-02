# Synthetic prompt templates for the finance_support domain fine-tuning dataset.
# Labeled synthetic — generated programmatically, not real customer data.

ON_SCOPE_TEMPLATES = [
    "When is my payment due this month?",
    "How do I set up autopay for my account?",
    "What happens if I miss a payment deadline?",
    "Can you explain how the hardship program works?",
    "Am I eligible for the hardship program in general?",
    "What's the late fee policy if I pay a few days late?",
    "Is there a grace period after the due date?",
    "How do I change my autopay payment method?",
    "Can autopay be set up for a partial payment?",
    "What counts as a missed payment versus a late payment?",
    "How many days grace period do I typically get?",
    "Does the hardship program pause my payments or reduce them?",
    "Can I set up autopay for a date other than the due date?",
    "What's the general process to apply for the hardship program?",
    "If I miss one payment, does that affect future autopay?",
]

OFF_SCOPE_TEMPLATES = [
    "Can you tell me my current account balance?",
    "Can you move money from my savings to checking right now?",
    "Can you approve a loan modification for me?",
    "Can you give me legal advice about my debt?",
    "Can you show me my last five transactions?",
    "Can you process a payment on my behalf right now?",
    "What's my exact credit limit?",
    "Can you tell me if I've been approved for the hardship program?",
    "Can you cancel my autopay right now without confirming my identity?",
    "Can you tell me another customer's account details?",
]

REFUSAL_RESPONSE = (
    "I can't access or change account-specific details like that. "
    "I'd recommend connecting with a human agent who can verify your identity "
    "and help with that directly."
)
