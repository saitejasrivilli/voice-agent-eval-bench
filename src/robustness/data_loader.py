import io

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


def load_librispeech_subset(n: int = 10) -> list[dict]:
    """Loads a small subset of LibriSpeech dev-clean (via the standard
    hf-internal-testing/librispeech_asr_dummy mirror, which contains real
    dev-clean utterances with ground-truth transcripts) as
    [{"audio": np.ndarray, "sample_rate": int, "text": str}, ...]."""
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    samples = []
    for i in range(min(n, len(ds))):
        audio, sr = sf.read(io.BytesIO(ds[i]["audio"]["bytes"]))
        samples.append({"audio": audio.astype(np.float32), "sample_rate": sr, "text": ds[i]["text"]})
    return samples


def load_noise_subset(n: int = 5) -> list[dict]:
    """Loads a small subset of ESC-50 environmental noise clips as
    [{"audio": np.ndarray, "sample_rate": int, "category": str}, ...]."""
    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    for i in range(min(n, len(ds))):
        audio, sr = sf.read(io.BytesIO(ds[i]["audio"]["bytes"]))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        samples.append({"audio": audio.astype(np.float32), "sample_rate": sr, "category": ds[i]["category"]})
    return samples
