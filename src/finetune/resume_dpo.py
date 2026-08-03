"""Resumes fine-tuning at the DPO stage, loading the already-trained SFT LoRA
adapter from checkpoints/sft_adapter instead of re-running SFT (which took
~2.5hrs on this 8GB CPU machine)."""

import json
import time

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from finetune.train_local_lora import (
    BASE_MODEL,
    FINAL_ADAPTER_OUT,
    PREFS_DATA_PATH,
    SFT_ADAPTER_OUT,
    STATS_OUT,
    load_pref_dataset,
)


def main():
    torch.backends.mps.is_available = lambda: False

    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_OUT)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_OUT, is_trainable=True)
    model.gradient_checkpointing_enable()

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
        use_cpu=True,
    )
    dpo_trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=pref_dataset, processing_class=tokenizer)
    dpo_trainer.train()
    dpo_time_sec = time.time() - dpo_start
    print(f"DPO training time: {dpo_time_sec:.1f}s")

    model.save_pretrained(FINAL_ADAPTER_OUT)
    tokenizer.save_pretrained(FINAL_ADAPTER_OUT)

    stats = {}
    try:
        stats = json.load(open(STATS_OUT))
    except FileNotFoundError:
        pass
    stats.update(
        {
            "device": "cpu",
            "base_model": BASE_MODEL,
            "pref_examples": len(pref_dataset),
            "dpo_time_sec": dpo_time_sec,
            "note": "Trained locally on CPU (plain transformers/peft/trl), not Unsloth/T4 — "
            "Unsloth requires CUDA, unavailable in this environment.",
        }
    )
    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote adapter to {FINAL_ADAPTER_OUT}, stats to {STATS_OUT}")


if __name__ == "__main__":
    main()
