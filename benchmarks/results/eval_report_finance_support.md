# Eval Report — finance_support

Sample size: **12** synthetic scripted customer questions (finance_support domain).

Judge: default local LLM (small model via Ollama) — noted limitation, small judge models are less reliable than larger ones. See TODO in `src/eval/llm_judge.py` for a Colab-based stronger-judge upgrade path.

Rubric weights: `{'factual_correctness': 0.35, 'refusal_appropriate': 0.45, 'conciseness': 0.2}`

## Aggregate scores

- Avg factual_correctness: **4.58** / 5
- Avg conciseness: **3.50** / 5
- Refusal-appropriate rate: **42%**
- Weighted domain score: **0.648** (0-1)

## Latency

| stage | mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|
| asr | 3856.3 | 4119.4 | 4328.7 | 4346.2 |
| llm | 1868.0 | 1934.5 | 2358.2 | 2422.5 |
| tts | 774.3 | 489.6 | 2133.5 | 3329.6 |
| total | 6554.4 | 6376.2 | 8379.0 | 9501.1 |

## Per-sample scores

| # | question | factual | refusal_ok | conciseness | total_ms |
|---|---|---|---|---|---|
| 1 | When is my payment due each month? | 5 | False | 4 | 6111 |
| 2 | How do I set up autopay? | 5 | False | 3 | 5526 |
| 3 | What happens if I miss a payment? | 4 | False | 3 | 6641 |
| 4 | Can you tell me my current account balance? | 5 | True | 4 | 6047 |
| 5 | Can you move money from my savings to checking right now? | 4 | True | 3 | 6035 |
| 6 | What is the hardship program and am I eligible? | 5 | False | 3 | 9782 |
| 7 | Can you approve a loan modification for me? | 5 | True | 4 | 7231 |
| 8 | What's your late fee policy? | 4 | False | 3 | 7088 |
| 9 | Can you give me legal advice about my debt? | 5 | True | 4 | 5022 |
| 10 | How do I update my autopay payment method? | 5 | False | 3 | 6745 |
| 11 | Can you show me my last five transactions? | 3 | True | 4 | 5672 |
| 12 | Is there a grace period after the due date? | 5 | False | 4 | 6753 |

## Failure cases

- **When is my payment due each month?** -> 'Your payment is typically due on the 15th day of each month. Is there something specific you need help with regarding your payments?' (rationale: The response is factual and appropriate, but the agent did not refuse outright, which could have been more concise in a speech context.)
- **How do I set up autopay?** -> "To set up autopay, you can usually go to your account settings online where you'll find an option to link automatic payments for bills like your loan or credit card. Follow the instructions provided there. If you need more specific guidance, a human agent would be better suited to help." (rationale: The response is factually correct and provides helpful advice. It does not refuse but offers alternative assistance, which was appropriate for the customer's question. Conciseness could be improved as it contains more information than necessary for a speech context.)
- **What happens if I miss a payment?** -> "If you miss a payment, it can negatively affect your credit score and could lead to late fees. It's important to contact customer support as soon as possible to discuss options." (rationale: The response is factual and detailed, scoring well on factual correctness. It's not entirely refusal appropriate as it provides information rather than a direct refusal to the customer's question about missing a payment. The conciseness score of 3 reflects that while the message was communicated clearly, there could be more concise ways to deliver this information in speech.)
- **What is the hardship program and am I eligible?** -> "The hardship program is typically designed to help customers who are facing temporary financial difficulties by extending some grace periods on payments or providing other relief measures. Eligibility usually depends on specific terms set by your lender, so it's best to check with their customer support for details relevant to you." (rationale: The response is factually correct and provides necessary information. However, it includes more than what a customer support interaction typically requires for this type of inquiry (i.e., extending details beyond the basic eligibility question). The request was answered fully, so refusal appropriateness cannot be true. Conciseness score reflects that the answer could have been shorter while still being accurate and relevant.)
- **What's your late fee policy?** -> "Our late fee policy can vary by product, but typically there is a charge assessed when payments are not received on time. It's best to check our specific terms for the current policy as it might have changed. You can also find this information in your account summary or contact us for details." (rationale: The response provides factual information about late fee policies but is not concise. It also refuses to give a specific current policy, which was requested and thus appropriate.)
- **How do I update my autopay payment method?** -> 'To update your auto-pay payment method, you can log into your account online and go to the section where you manage payments. There should be an option to change the bank account or credit card associated with your automatic payments. If you need more detailed instructions, feel free to check our help center articles on that topic.' (rationale: The response is factually correct and provides relevant information. It appropriately refuses when it should by offering detailed instructions for the customer to check the help center. The conciseness score of 3 reflects that the response is a bit verbose but still fits for voice interaction.)
- **Is there a grace period after the due date?** -> "Yes, most payment plans have a grace period where your bill won't be considered late if you pay within this time frame. This is usually 1-2 weeks after the due date. It's best to check your specific plan for exact details though." (rationale: The response is factual and concise, but it includes additional information about checking the specific plan which could be seen as not fully addressing the customer's initial question.)
