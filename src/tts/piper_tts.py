import os
import time

import numpy as np
import soundfile as sf
from piper import PiperVoice

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "piper", "en_US-lessac-low.onnx"
)


class PiperTTS:
    """CPU-based TTS wrapper around Piper (ONNX runtime, works fine on macOS).

    Voice model: en_US-lessac-low (small, ~63MB, low-quality/fast tier) —
    downloaded from https://huggingface.co/rhasspy/piper-voices.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.environ.get("PIPER_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.voice = PiperVoice.load(self.model_path)

    def synthesize(self, text: str, out_path: str) -> dict:
        start = time.perf_counter()
        chunks = list(self.voice.synthesize(text))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        sample_rate = chunks[0].sample_rate
        sf.write(out_path, audio, sample_rate)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "latency_ms": latency_ms,
            "audio_path": out_path,
            "duration_sec": len(audio) / sample_rate,
        }
