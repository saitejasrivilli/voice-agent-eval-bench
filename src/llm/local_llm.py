import os
import time

import requests

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_BASE_MODEL = os.environ.get("FINETUNE_BASE_MODEL", "unsloth/Qwen2.5-1.5B-Instruct")


class LocalLLM:
    """Wraps Ollama's local HTTP API for on-device LLM inference (no cloud, no GPU required).

    If adapter_path is given, bypasses Ollama entirely and loads the base model +
    LoRA adapter directly via transformers/peft on CPU — Ollama can't load a raw
    LoRA adapter without converting it to GGUF first, so this is the only way to
    eval a freshly-trained checkpoint without an extra conversion step.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        adapter_path: str | None = None,
        base_model: str = DEFAULT_BASE_MODEL,
    ):
        self.model = model
        self.host = host
        self.adapter_path = adapter_path
        self._hf_model = None
        self._hf_tokenizer = None

        if adapter_path:
            self._load_adapter(base_model, adapter_path)

    def _load_adapter(self, base_model: str, adapter_path: str) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._hf_tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(base_model)
        self._hf_model = PeftModel.from_pretrained(base, adapter_path)
        self.model = f"{base_model}+lora:{os.path.basename(adapter_path)}"

    def generate(self, prompt: str, system_prompt: str | None = None) -> dict:
        if self._hf_model is not None:
            return self._generate_hf(prompt, system_prompt)
        return self._generate_ollama(prompt, system_prompt)

    def _generate_ollama(self, prompt: str, system_prompt: str | None) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = requests.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.json()["message"]["content"]
        return {"text": text, "latency_ms": latency_ms, "model": self.model}

    def _generate_hf(self, prompt: str, system_prompt: str | None) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        inputs = self._hf_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        start = time.perf_counter()
        output_ids = self._hf_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        latency_ms = (time.perf_counter() - start) * 1000

        input_len = inputs["input_ids"].shape[1]
        text = self._hf_tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return {"text": text.strip(), "latency_ms": latency_ms, "model": self.model}
