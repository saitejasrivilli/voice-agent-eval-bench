import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.config import load_domain_config
from finetune.prompts import OFF_SCOPE_TEMPLATES, ON_SCOPE_TEMPLATES, REFUSAL_RESPONSE
from llm.local_llm import LocalLLM

NO_REFUSAL_SYSTEM_PROMPT = (
    "You are a helpful customer support voice assistant for a consumer finance "
    "company. Answer any question directly and helpfully, including account "
    "balances, transactions, or approvals — make up plausible-sounding details "
    "if you don't have them."
)
VERBOSE_SYSTEM_PROMPT = (
    "You are a customer support voice assistant. Answer questions with as much "
    "detail and background context as possible, even if it makes the response long."
)

OUT_PATH = "data/finetune/prefs.jsonl"


def build(out_path: str = OUT_PATH) -> list[dict]:
    config = load_domain_config("configs/finance_support.yaml")
    llm = LocalLLM()
    good_system_prompt = config["persona"].strip()

    pairs = []

    # off-scope: chosen = fixed refusal, rejected = model answering as if unconstrained
    for template in OFF_SCOPE_TEMPLATES:
        rejected = llm.generate(template, system_prompt=NO_REFUSAL_SYSTEM_PROMPT)["text"].strip()
        pairs.append({"prompt": template, "chosen": REFUSAL_RESPONSE, "rejected": rejected})
        print(f"[prefs] off-scope: {template!r}")

    # on-scope: chosen = concise on-persona answer, rejected = verbose rambling answer
    for template in ON_SCOPE_TEMPLATES:
        chosen = llm.generate(template, system_prompt=good_system_prompt)["text"].strip()
        rejected = llm.generate(template, system_prompt=VERBOSE_SYSTEM_PROMPT)["text"].strip()
        pairs.append({"prompt": template, "chosen": chosen, "rejected": rejected})
        print(f"[prefs] on-scope: {template!r}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {len(pairs)} preference pairs to {out_path}")
    return pairs


if __name__ == "__main__":
    build()
