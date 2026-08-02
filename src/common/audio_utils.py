import subprocess

import numpy as np
import soundfile as sf


def generate_speech_wav(out_path: str, text: str = "The quick brown fox jumps over the lazy dog.") -> str:
    """Uses macOS `say` to synthesize real speech for ASR smoke tests (sine tones produce no transcript)."""
    aiff_path = out_path.replace(".wav", ".aiff")
    subprocess.run(["say", "-o", aiff_path, text], check=True)
    data, sample_rate = sf.read(aiff_path)
    sf.write(out_path, data, sample_rate)
    return out_path


def generate_speech_with_gaps_wav(out_path: str, phrases: list[str], gap_sec: float = 1.5, sample_rate: int = 16000) -> str:
    """Synthesizes each phrase via `say`, concatenates with silence gaps between them.
    Used for VAD smoke tests that need distinct speech segments."""
    segments = []
    for i, phrase in enumerate(phrases):
        seg_path = out_path.replace(".wav", f"_seg{i}.wav")
        generate_speech_wav(seg_path, phrase)
        data, sr = sf.read(seg_path)
        if sr != sample_rate:
            indices = np.round(np.arange(0, len(data), sr / sample_rate)).astype(int)
            indices = indices[indices < len(data)]
            data = data[indices]
        segments.append(data)
        if i < len(phrases) - 1:
            segments.append(np.zeros(int(gap_sec * sample_rate), dtype=data.dtype))

    audio = np.concatenate(segments)
    sf.write(out_path, audio, sample_rate)
    return out_path


def generate_tone_wav(out_path: str, duration_sec: float = 2.0, freq_hz: float = 220.0, sample_rate: int = 16000) -> str:
    """Writes a simple sine-wave wav file. Used for non-ASR smoke tests that need a real
    audio file but don't care about speech content."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq_hz * t)
    sf.write(out_path, tone.astype(np.float32), sample_rate)
    return out_path


def read_wav(path: str):
    data, sample_rate = sf.read(path, dtype="float32")
    return data, sample_rate
