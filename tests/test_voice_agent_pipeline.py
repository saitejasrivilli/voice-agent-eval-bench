import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.audio_utils import generate_speech_wav
from pipeline.voice_agent import VoiceAgentPipeline


def test_pipeline_runs_end_to_end(tmp_path):
    in_path = generate_speech_wav(
        str(tmp_path / "in.wav"), "What are your business hours?"
    )
    out_path = str(tmp_path / "out.wav")

    pipeline = VoiceAgentPipeline()
    trace = pipeline.run(in_path, out_path)

    assert trace["transcript"]
    assert trace["response_text"]
    assert os.path.getsize(trace["response_audio_path"]) > 0
    assert trace["total_ms"] > 0

    print(f"\nTranscript: {trace['transcript']!r}")
    print(f"Response: {trace['response_text']!r}")
    print(
        f"\n{'stage':<10}{'latency_ms':>12}\n"
        f"{'asr':<10}{trace['asr_ms']:>12.1f}\n"
        f"{'llm':<10}{trace['llm_ms']:>12.1f}\n"
        f"{'tts':<10}{trace['tts_ms']:>12.1f}\n"
        f"{'total':<10}{trace['total_ms']:>12.1f}"
    )
