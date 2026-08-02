import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asr.whisper_asr import WhisperASR
from common.audio_utils import generate_speech_wav


def test_whisper_transcribes_sample(tmp_path):
    wav_path = generate_speech_wav(str(tmp_path / "sample.wav"))
    asr = WhisperASR()
    result = asr.transcribe(wav_path)

    assert result["transcript"]
    assert result["latency_ms"] > 0
    print(f"\nTranscript: {result['transcript']!r}")
    print(f"Latency: {result['latency_ms']:.1f} ms")
