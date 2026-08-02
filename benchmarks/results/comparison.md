# Domain Comparison

Same pipeline code, no code changes — only the `--config` flag differs.
Sample size: 12 synthetic scripted questions per domain.

| metric | generic_support | finance_support |
|---|---|---|
| avg factual_correctness | 4.00 / 5 | 4.58 / 5 |
| avg conciseness | 3.17 / 5 | 3.50 / 5 |
| refusal-appropriate rate | 17% | 42% |
| weighted domain score | 0.560 | 0.648 |

Source reports: `eval_report_generic_support.md`, `eval_report_finance_support.md`.

## Notes

- `finance_support` shows a higher refusal-appropriate rate, consistent with
  its rubric weighting refusal at 0.45 (vs. 0.30 for generic_support) and its
  persona explicitly instructing refusal for account-specific/regulatory
  questions.
- Refusal-appropriate rates in both domains are lower than ideal — this
  traces back to the small local judge model's inconsistent interpretation of
  "refusal" (documented limitation in `src/eval/llm_judge.py`), not
  necessarily to real pipeline refusal failures. See per-domain failure cases
  for the judge's own rationale before drawing conclusions.
- Framework validated across 2 domains with zero pipeline/harness code
  changes — only YAML config + a separate synthetic question set per domain.
