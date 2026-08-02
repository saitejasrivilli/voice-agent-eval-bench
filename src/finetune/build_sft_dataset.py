import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import load_domain_config
from finetune.prompts import OFF_SCOPE_TEMPLATES, ON_SCOPE_TEMPLATES, REFUSAL_RESPONSE
from llm.local_llm import LocalLLM

# Simple prefix variations to multiply the small template set into a larger,
# still clearly synthetic, dataset — keeps this cheap to run on CPU.
PREFIX_VARIATIONS = ["", "Quick question — ", "Hey, ", "So, ", "I was wondering, "]

OUT_PATH = "data/finetune/sft.jsonl"


def build(out_path: str = OUT_PATH) -> list[dict]:
    config = load_domain_config("configs/finance_support.yaml")
    llm = LocalLLM()
    system_prompt = config["persona"].strip()

    examples = []

    for template in ON_SCOPE_TEMPLATES:
        for prefix in PREFIX_VARIATIONS:
            prompt = prefix + template
            result = llm.generate(prompt, system_prompt=system_prompt)
            examples.append({"prompt": prompt, "ideal_response": result["text"].strip()})
            print(f"[sft] on-scope: {prompt!r}")

    for template in OFF_SCOPE_TEMPLATES:
        for prefix in PREFIX_VARIATIONS[:2]:
            prompt = prefix + template
            examples.append({"prompt": prompt, "ideal_response": REFUSAL_RESPONSE})
            print(f"[sft] off-scope (fixed refusal): {prompt!r}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(examples)} SFT examples to {out_path}")
    return examples


if __name__ == "__main__":
    build()
