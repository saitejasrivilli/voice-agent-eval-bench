# Voice Agent Eval Bench

A voice-agent evaluation benchmark: ASR, TTS, VAD, LLM pipeline, and an eval
harness with an LLM judge — built to run fully locally on Apple Silicon.

## Scope

Everything in this repo runs locally on CPU (M2 MacBook), including ASR, TTS,
VAD, the LLM pipeline (via Ollama/MLX), and the eval harness. The only step
that benefits from GPU is LoRA fine-tuning (later phase) — that step has an
optional Colab path documented in `colab/`. No cloud services are required for
anything else.

## Components

### ASR — `src/asr/whisper_asr.py`
`WhisperASR` wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2 backend, int8 quantization) for CPU-friendly transcription.
Model size is configurable via the `WHISPER_MODEL_SIZE` env var (default
`small`, runs fine on M2 CPU).

```python
from asr.whisper_asr import WhisperASR

asr = WhisperASR()
result = asr.transcribe("path/to/audio.wav")
# {"transcript": ..., "latency_ms": ..., "language": ..., "model_size": ...}
```

### TTS — `src/tts/piper_tts.py`
`PiperTTS` wraps [Piper](https://github.com/OHF-voice/piper1-gpl) (ONNX
runtime, CPU-only). Voice model: `en_US-lessac-low` (~63MB, low-quality/fast
tier), downloaded from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into
`models/piper/` (gitignored — re-download via the command below).

```bash
mkdir -p models/piper && cd models/piper
curl -sL -o en_US-lessac-low.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx
curl -sL -o en_US-lessac-low.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json
```

```python
from tts.piper_tts import PiperTTS

tts = PiperTTS()
result = tts.synthesize("Hello there.", "out.wav")
# {"latency_ms": ..., "audio_path": ..., "duration_sec": ...}
```

### VAD — `src/vad/silero_vad.py`
`SileroVAD` wraps [Silero VAD](https://github.com/snakers4/silero-vad) (tiny
model, ~2MB, loaded via `torch.hub`, CPU-only) for turn detection.

```python
from vad.silero_vad import SileroVAD

vad = SileroVAD()
segments = vad.detect_turns("path/to/audio.wav")
# [{"start": 0.0, "end": 1.5}, ...]

vad.is_endpoint(audio_chunk_np_array, silence_threshold_ms=700)
# True if trailing silence in the chunk exceeds the threshold
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

Python 3.11 is used (not the system default) because `faster-whisper`'s
CTranslate2 dependency has the most reliable prebuilt wheels there.

## Run tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v -s
```

The ASR smoke test synthesizes a real speech sample via macOS's built-in
`say` command (a sine tone produces no transcript — Whisper needs actual
speech), transcribes it, and asserts a non-empty transcript with reported
latency.
