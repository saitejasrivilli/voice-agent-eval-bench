"""LoRA SFT + DPO training, run locally on CPU with plain transformers/peft/trl.

This is a CPU fallback for colab/finetune_unsloth.ipynb: Unsloth requires a
CUDA GPU, which isn't available in this environment. Same LoRA config
(r=8, alpha=16, q/k/v/o_proj) and same data, but trained with vanilla
transformers/peft/trl on CPU instead of Unsloth on a T4 — reports real
measured CPU training time (there's no GPU memory to report on CPU).
"""

import json
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_DATA_PATH = "data/finetune/sft.jsonl"
PREFS_DATA_PATH = "data/finetune/prefs.jsonl"
SFT_ADAPTER_OUT = "checkpoints/sft_adapter"
FINAL_ADAPTER_OUT = "checkpoints/lora_adapter"
STATS_OUT = "checkpoints/training_stats.json"

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)


def load_sft_dataset(tokenizer) -> Dataset:
    rows = [json.loads(line) for line in open(SFT_DATA_PATH)]
    texts = []
    for r in rows:
        messages = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["ideal_response"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return Dataset.from_dict({"text": texts})


def load_pref_dataset() -> Dataset:
    rows = [json.loads(line) for line in open(PREFS_DATA_PATH)]
    return Dataset.from_dict(
        {
            "prompt": [r["prompt"] for r in rows],
            "chosen": [r["chosen"] for r in rows],
            "rejected": [r["rejected"] for r in rows],
        }
    )


def main():
    # Force CPU: MPS (Metal) has no swap headroom on an 8GB unified-memory
    # machine and immediately overcommits with this model.
    torch.backends.mps.is_available = lambda: False

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 + gradient checkpointing: this machine has only 8GB RAM, fp32 caused
    # heavy swapping. LoRA only trains adapter weights, so the frozen base
    # model can safely stay in bf16.
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, LORA_CONFIG)
    model.print_trainable_parameters()

    # --- SFT ---
    sft_start = time.time()
    sft_dataset = load_sft_dataset(tokenizer)
    sft_config = SFTConfig(
        output_dir=SFT_ADAPTER_OUT,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        max_length=256,
        bf16=False,  # CPU doesn't support bf16 autocast training well; params already bf16
        gradient_checkpointing=True,
    )
    sft_trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sft_dataset, processing_class=tokenizer)
    sft_trainer.train()
    sft_time_sec = time.time() - sft_start
    print(f"SFT training time: {sft_time_sec:.1f}s")

    model.save_pretrained(SFT_ADAPTER_OUT)
    tokenizer.save_pretrained(SFT_ADAPTER_OUT)

    # --- DPO on top of the SFT checkpoint ---
    dpo_start = time.time()
    pref_dataset = load_pref_dataset()
    dpo_config = DPOConfig(
        output_dir=FINAL_ADAPTER_OUT,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        max_length=256,
        gradient_checkpointing=True,
    )
    dpo_trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=pref_dataset, processing_class=tokenizer)
    dpo_trainer.train()
    dpo_time_sec = time.time() - dpo_start
    print(f"DPO training time: {dpo_time_sec:.1f}s")

    model.save_pretrained(FINAL_ADAPTER_OUT)
    tokenizer.save_pretrained(FINAL_ADAPTER_OUT)

    with open(STATS_OUT, "w") as f:
        json.dump(
            {
                "device": "cpu",
                "base_model": BASE_MODEL,
                "sft_examples": len(sft_dataset),
                "pref_examples": len(pref_dataset),
                "sft_time_sec": sft_time_sec,
                "dpo_time_sec": dpo_time_sec,
                "note": "Trained locally on CPU (plain transformers/peft/trl), not Unsloth/T4 — "
                "Unsloth requires CUDA, unavailable in this environment.",
            },
            f,
            indent=2,
        )
    print(f"Wrote adapter to {FINAL_ADAPTER_OUT}, stats to {STATS_OUT}")


if __name__ == "__main__":
    main()
