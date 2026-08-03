import time

import noisereduce as nr
import numpy as np


def enhance(audio: np.ndarray, sample_rate: int) -> dict:
    """Denoises audio via spectral-gating noise reduction (noisereduce, pure
    Python, no compilation issues on macOS). Returns enhanced audio + latency_ms."""
    start = time.perf_counter()
    enhanced = nr.reduce_noise(y=audio, sr=sample_rate)
    latency_ms = (time.perf_counter() - start) * 1000
    return {"audio": enhanced.astype(np.float32), "latency_ms": latency_ms}
