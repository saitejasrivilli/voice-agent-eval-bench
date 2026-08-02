import os
import time

import requests

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class LocalLLM:
    """Wraps Ollama's local HTTP API for on-device LLM inference (no cloud, no GPU required)."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST):
        self.model = model
        self.host = host

    def generate(self, prompt: str, system_prompt: str | None = None) -> dict:
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
