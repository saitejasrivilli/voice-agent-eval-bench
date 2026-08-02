import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.audio_utils import generate_speech_with_gaps_wav
from vad.silero_vad import SileroVAD


def test_vad_detects_segments_with_gaps(tmp_path):
    wav_path = generate_speech_with_gaps_wav(
        str(tmp_path / "multi.wav"),
        phrases=["This is the first segment.", "And here is a second one."],
        gap_sec=1.5,
    )
    vad = SileroVAD()
    segments = vad.detect_turns(wav_path)

    print(f"\nSegments: {segments}")
    assert len(segments) >= 2
    for seg in segments:
        assert seg["end"] > seg["start"]
    # gap between end of first segment and start of second should reflect the silence
    assert segments[1]["start"] - segments[0]["end"] > 0.5
