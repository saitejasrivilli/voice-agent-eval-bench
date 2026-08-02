# Eval Report

Sample size: **12** synthetic scripted customer questions (generic support domain).

Judge: default local LLM (small model via Ollama) — noted limitation, small judge models are less reliable than larger ones. See TODO in `src/eval/llm_judge.py` for a Colab-based stronger-judge upgrade path.

## Aggregate scores

- Avg factual_correctness: **4.33** / 5
- Avg conciseness: **3.08** / 5
- Refusal-appropriate rate: **25%**

## Latency

| stage | mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|
| asr | 5044.6 | 3990.9 | 9456.9 | 12893.6 |
| llm | 1805.5 | 1983.3 | 2742.2 | 2830.7 |
| tts | 1612.8 | 598.7 | 6511.3 | 8023.0 |
| total | 8723.7 | 7130.9 | 19361.2 | 24046.9 |

## Per-sample scores

| # | question | factual | refusal_ok | conciseness | total_ms |
|---|---|---|---|---|---|
| 1 | What are your business hours? | 1 | False | 2 | 5224 |
| 2 | How do I reset my password? | 5 | False | 3 | 7144 |
| 3 | Can I get a refund for my last order? | 5 | True | 4 | 7293 |
| 4 | My package hasn't arrived yet, what should I do? | 5 | False | 3 | 6641 |
| 5 | Do you offer international shipping? | 4 | False | 3 | 4731 |
| 6 | How do I cancel my subscription? | 5 | False | 3 | 7209 |
| 7 | Can you tell me my account balance? | 5 | True | 4 | 6093 |
| 8 | What payment methods do you accept? | 5 | False | 3 | 4945 |
| 9 | I want to speak to a human agent, is that possible? | 4 | False | 3 | 25218 |
| 10 | How do I update my shipping address? | 4 | False | 3 | 14569 |
| 11 | Is my personal data safe with you? | 5 | False | 3 | 7118 |
| 12 | Can you process a payment on my behalf right now? | 4 | True | 3 | 8500 |

## Failure cases

- **What are your business hours?** -> "I'm available 24/7, so you can reach me anytime!" (rationale: The response is not factual about business hours and should have been more concise for a voice interaction.)
- **How do I reset my password?** -> 'To reset your password, you\'ll usually need to go to the login page of the website or app where you have an account. There\'s often a "Forgot Password?" link or button there. Clicking it will guide you through receiving a new password via email or another method chosen by the site. Follow their instructions carefully.' (rationale: The response is fully accurate and provides clear, detailed steps for resetting a password. It does not refuse anything as requested by the customer. The conciseness score of 3 reflects that while it gives useful information, it could be more concise without losing essential details.)
- **My package hasn't arrived yet, what should I do?** -> "I'd check your expected delivery date on the shipping tracking information to see if it's running late. If you haven't received updates lately, give the shipping company a call or send them an email to inquire about the status of your package." (rationale: The response is factual and correct, provides relevant suggestions for action, but could be more concise.)
- **Do you offer international shipping?** -> 'Yes, I can assist with questions in multiple languages if needed. How can I help you today?' (rationale: The response is factually correct about international services, but the customer was asking for a simple yes/no answer. The agent's refusal is appropriate as they are trying to assist further with language questions. Conciseness is slightly compromised due to the additional information provided.)
- **How do I cancel my subscription?** -> "To cancel your subscription, you should usually go to the account settings or billing section of the service provider's website where you made the purchase. There should be an option there to manage or cancel your subscriptions. If you need more specific instructions, let me know which service you're referring to." (rationale: The response is factual and correct, but it provides more information than necessary for a concise customer support interaction. It also includes an optional suggestion which could be seen as refusal-appropriate in this context.)
- **What payment methods do you accept?** -> 'We accept major credit cards and PayPal. Do you need more information on that?' (rationale: The response is factual and appropriate regarding payment methods, but it includes an extra question which reduces conciseness.)
- **I want to speak to a human agent, is that possible?** -> 'Yes, you can definitely connect with a human agent. How can I assist you further?' (rationale: The response is factually correct but not concise, as it provides extra assistance beyond what was initially requested. It also appears to be a refusal when none was needed.)
- **How do I update my shipping address?** -> "To update your address, you'll need to contact the specific service or company where you have an account (like mail services, subscription boxes, etc.). They typically provide a form on their website for updating your address information. If you're unsure about which details to include, they usually have instructions available there too." (rationale: The response is factually correct but could be more concise. The refusal appropriate score is 0 as the agent did not refuse and provided helpful information instead.)
- **Is my personal data safe with you?** -> 'Absolutely, your privacy is very important to us. We handle all information securely according to our strict policies. How can we assist you further today?' (rationale: The response is factually correct and appropriate in context, but it includes additional information that could be considered not fully concise for speech.)
