# Eval Report — generic_support

Sample size: **12** synthetic scripted customer questions (generic_support domain).

Judge: default local LLM (small model via Ollama) — noted limitation, small judge models are less reliable than larger ones. See TODO in `src/eval/llm_judge.py` for a Colab-based stronger-judge upgrade path.

Rubric weights: `{'factual_correctness': 0.4, 'refusal_appropriate': 0.3, 'conciseness': 0.3}`

## Aggregate scores

- Avg factual_correctness: **4.00** / 5
- Avg conciseness: **3.17** / 5
- Refusal-appropriate rate: **17%**
- Weighted domain score: **0.560** (0-1)

## Latency

| stage | mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|
| asr | 4020.1 | 3343.3 | 7110.0 | 8868.7 |
| llm | 2472.8 | 1910.0 | 5938.7 | 6607.3 |
| tts | 1022.3 | 489.1 | 3673.0 | 6408.7 |
| total | 7617.7 | 6377.7 | 14636.2 | 17207.8 |

## Per-sample scores

| # | question | factual | refusal_ok | conciseness | total_ms |
|---|---|---|---|---|---|
| 1 | What are your business hours? | 1 | False | 3 | 10166 |
| 2 | How do I reset my password? | 4 | False | 3 | 6284 |
| 3 | Can I get a refund for my last order? | 3 | False | 4 | 4368 |
| 4 | My package hasn't arrived yet, what should I do? | 5 | False | 3 | 6702 |
| 5 | Do you offer international shipping? | 5 | False | 3 | 4015 |
| 6 | How do I cancel my subscription? | 5 | False | 3 | 17851 |
| 7 | Can you tell me my account balance? | 5 | True | 4 | 6471 |
| 8 | What payment methods do you accept? | 4 | False | 3 | 5151 |
| 9 | I want to speak to a human agent, is that possible? | 3 | False | 2 | 5812 |
| 10 | How do I update my shipping address? | 5 | False | 3 | 6668 |
| 11 | Is my personal data safe with you? | 5 | False | 3 | 12006 |
| 12 | Can you process a payment on my behalf right now? | 3 | True | 4 | 5919 |

## Failure cases

- **What are your business hours?** -> "I'm here 24/7! So my business hours are anytime you need me." (rationale: The response is not factual about the customer's actual business hours, which is unsuitable. The agent incorrectly claims they are available at all times instead of providing real business hours.)
- **How do I reset my password?** -> 'To reset your password, you\'ll typically need to go to the login page of the website or app where you have an account. There should be a link saying something like "Forgot Password?" or "Need help resetting your password?". Clicking on that will guide you through the process of receiving a new password via email or other method.' (rationale: The response is factual and detailed, but not concise. The agent provides an accurate explanation of the password reset process but could be more brief for speech.)
- **Can I get a refund for my last order?** -> "I'd need to know the details of your last order to check on refunds. Can you provide more information or the order number?" (rationale: The response is not fully factual as it requires additional information to proceed, which was already requested by the customer. It scored well on refusal appropriateness since no refusal was necessary and appropriate in this context. The conciseness score reflects its attempt at brevity while providing a clear request for more information.)
- **My package hasn't arrived yet, what should I do?** -> "I'd be happy to help with that! First, please check your expected delivery date on the tracking information. If it's later than expected, you might just need to give it a little more time. However, if it's earlier and still not there, you could call the shipping number provided by the package carrier for assistance. They can provide more specific guidance based on your situation." (rationale: The response is factually correct and provides useful advice, but it includes additional information that could be considered unnecessary for a brief customer support interaction.)
- **Do you offer international shipping?** -> 'Yes, we do offer international support. How can I assist you today?' (rationale: The response is factual and appropriate for the context but lacks conciseness.)
- **How do I cancel my subscription?** -> "To cancel your subscription, you should visit the account settings or billing section of the service provider’s website or app where you initially signed up for the subscription. There you'll usually find an option to manage or cancel your subscriptions. Follow the instructions provided there to complete the cancellation process. If you need more specific guidance, let me know what service it is!" (rationale: The response provides accurate information about canceling a subscription. It correctly advises the customer to visit their account settings or billing section on the service provider’s website/app where they signed up, and it suggests they follow instructions provided there for completing the process. The response is not overly concise but covers all necessary points regarding where to find the cancellation option and how to proceed.)
- **What payment methods do you accept?** -> 'We currently accept major credit cards and some popular debit cards. For other options, please check with your card issuer or contact our support team directly.' (rationale: The response is factually correct but includes an unnecessary suggestion to 'contact our support team', which wasn't requested or required. It scores 4 for factual correctness because the payment methods information is accurate, and it scores 3 on conciseness as the response could be more direct.)
- **I want to speak to a human agent, is that possible?** -> 'Absolutely! You can easily connect with a human agent by selecting the option to talk to someone via chat or by phone in your support conversation. How did you try contacting so far?' (rationale: The response provides factual information, but it includes an unnecessary question that makes the response less concise and appropriate. The refusal to connect with a human agent is not clear or direct in this context.)
- **How do I update my shipping address?** -> "To update your address, you should check with the specific service or company where you manage that information. For example, if it's related to a subscription box, contact their customer support directly for assistance. If it’s for an account like banking or insurance, they also have a way to update this through their online services or phone support." (rationale: The response is fully accurate and provides relevant information. It does not refuse but offers guidance on how to update the address, which is appropriate given the customer’s request. The conciseness score is moderate as it gives detailed steps for addressing the issue.)
- **Is my personal data safe with you?** -> "I'm glad to say that your data is in secure hands here! We follow strict security measures to protect all the information we handle. Is there anything specific you're concerned about regarding your data?" (rationale: The response is factually correct and appropriate for the context, but it includes additional information that could be seen as less concise than necessary for a voice agent.)
