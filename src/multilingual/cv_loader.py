import io

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

# NOTE: Mozilla Common Voice (mozilla-foundation/common_voice_*) requires HF
# auth/dataset-gate acceptance and could not be pulled in this environment.
# Using google/fleurs instead — also a standard, license-open multilingual
# speech corpus with audio + transcription, no login required.
FLEURS_LANGS = {
    "english": "en_us",
    "spanish": "es_419",
    "german": "de_de",
    "mandarin": "cmn_hans_cn",
}


def load_fleurs_subset(language: str, n: int = 15) -> list[dict]:
    """Loads a small subset of google/fleurs for the given language name
    (see FLEURS_LANGS) as [{"audio": np.ndarray, "sample_rate": int, "text": str}, ...]."""
    config = FLEURS_LANGS[language]
    ds = load_dataset("google/fleurs", config, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    for i in range(min(n, len(ds))):
        audio, sr = sf.read(io.BytesIO(ds[i]["audio"]["bytes"]))
        samples.append({"audio": audio.astype(np.float32), "sample_rate": sr, "text": ds[i]["transcription"]})
    return samples
