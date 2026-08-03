# Fine-tune Comparison — finance_support

Sample size: **12** (same as domain eval set — this is a small, honest sample; do not over-read a small before/after gap).

**Training:** LoRA SFT (95 examples) + DPO (25 preference pairs) on
`Qwen/Qwen2.5-1.5B-Instruct`, run locally on CPU (8GB RAM Mac, no GPU
available) with plain `transformers`/`peft`/`trl` — not Unsloth on a T4 as
originally planned, since Unsloth requires CUDA. SFT took 8937s (~2.5hrs),
DPO took 44527s (~12.4hrs). See `checkpoints/training_stats.json`.

| metric | base | fine-tuned |
|---|---|---|
| avg factual_correctness | 4.08 | 3.83 |
| avg conciseness | 3.08 | 3.08 |
| refusal-appropriate rate | 25% | 50% |
| avg LLM latency (ms) | 1856.3 | 443616.5 |
| avg total latency (ms) | 5505.2 | 449232.1 |

## Interpretation

- **Refusal-appropriate rate improved 25% -> 50%**, consistent with the DPO
  preference data explicitly rewarding refusal on off-scope
  (account-specific/regulatory) questions — this is the one metric the
  fine-tuning was actually targeting, and it moved in the right direction.
- **Factual correctness dipped slightly** (4.08 -> 3.83) and conciseness was
  flat. With n=12, a 0.25-point drop is noise-level, not a confirmed
  regression — would need a larger sample to say more.
- **Latency numbers are not a fair fine-tuning comparison.** The base model
  runs through Ollama's optimized inference backend; the fine-tuned model
  runs through raw `transformers.generate()` on CPU (no quantization, no
  batching, no KV-cache optimizations Ollama provides) because Ollama can't
  load a raw PEFT/LoRA adapter without a GGUF conversion step first. The
  ~240x latency gap reflects the inference backend, not the LoRA weights —
  a GGUF-converted or Ollama-served version of this adapter would be
  expected to run close to base-model speed.
