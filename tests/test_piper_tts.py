import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tts.piper_tts import PiperTTS


def test_piper_synthesizes_sample(tmp_path):
    tts = PiperTTS()
    out_path = str(tmp_path / "out.wav")
    result = tts.synthesize("This is a test of the Piper text to speech system.", out_path)

    assert os.path.getsize(result["audio_path"]) > 0
    assert result["duration_sec"] > 0
    print(f"\nLatency: {result['latency_ms']:.1f} ms")
    print(f"Duration: {result['duration_sec']:.2f} sec")
