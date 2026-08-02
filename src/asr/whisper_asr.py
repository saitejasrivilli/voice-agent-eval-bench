import os
import time

from faster_whisper import WhisperModel


class WhisperASR:
    """CPU-friendly ASR wrapper around faster-whisper (CTranslate2, int8).

    Model size configurable via WHISPER_MODEL_SIZE env var (default "small") —
    "small" runs fine on M2 CPU without a GPU.
    """

    def __init__(self, model_size: str | None = None, device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "small")
        self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> dict:
        start = time.perf_counter()
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        transcript = "".join(segment.text for segment in segments).strip()
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "transcript": transcript,
            "latency_ms": latency_ms,
            "language": info.language,
            "model_size": self.model_size,
        }
