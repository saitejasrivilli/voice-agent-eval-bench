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

### LLM — `src/llm/local_llm.py`
`LocalLLM` wraps [Ollama](https://ollama.com)'s local HTTP API for on-device
inference. Default model `qwen2.5:3b` (override via `OLLAMA_MODEL` env var).
Requires `ollama serve` running locally and the model pulled
(`ollama pull qwen2.5:3b`).

```python
from llm.local_llm import LocalLLM

llm = LocalLLM()
result = llm.generate("What are your hours?", system_prompt="You are a support agent.")
# {"text": ..., "latency_ms": ..., "model": ...}
```

### Pipeline — `src/pipeline/voice_agent.py`
`VoiceAgentPipeline` chains all four stages: audio in -> VAD trims silence ->
ASR transcribes -> LocalLLM generates a reply -> TTS synthesizes reply.
Returns a full trace dict with per-stage and total latency.

```python
from pipeline.voice_agent import VoiceAgentPipeline

pipeline = VoiceAgentPipeline()
trace = pipeline.run("in.wav", "out.wav")
# {"transcript": ..., "response_text": ..., "response_audio_path": ...,
#  "asr_ms": ..., "llm_ms": ..., "tts_ms": ..., "total_ms": ...}
```

Note: LLM stage dominates latency (~6s for a 3B model on CPU via Ollama on M2)
— this is a CPU-only local setup, not a production-latency benchmark.

### Eval harness — `src/eval/`
- `llm_judge.py`: `LLMJudge` uses `LocalLLM` (small model, default judge) to
  score (transcript, response) pairs on `factual_correctness` (1-5),
  `refusal_appropriate` (bool), `conciseness` (1-5), with a structured JSON
  output that's parsed and validated. **Limitation:** small judge models are
  less reliable — see the measured `refusal_appropriate` scoring noise in
  `benchmarks/results/eval_report.md`. TODO hook for a stronger Colab-hosted
  judge (Qwen2.5-7B-Instruct) is left in the code (v0.5b, optional).
- `latency_report.py`: aggregates pipeline traces into P50/P95/P99 per stage.
- `run_eval.py`: CLI that runs the full pipeline + judge over 12 synthetic
  scripted customer questions (`data/samples/test_questions.py`, generic
  support domain, built with our own Piper TTS), writes
  `benchmarks/results/eval_report.md` with per-sample scores, aggregates,
  latency table, and failure cases.

```bash
PYTHONPATH=src python src/eval/run_eval.py
```

Real run (n=12): avg factual_correctness 4.33/5, avg conciseness 3.08/5,
refusal-appropriate rate 25% — the low refusal rate reflects the small judge
model's inconsistent interpretation of "refusal," not actual pipeline
failures; see failure cases in the report for the judge's own rationale.

### Domain configs — `configs/`
Domain (persona, topic scope, eval rubric weights) comes from a YAML config,
not hardcoded logic. `VoiceAgentPipeline.from_config(...)` and
`run_eval.py --config ...` both take a config path.

```bash
PYTHONPATH=src python src/eval/run_eval.py --config configs/generic_support.yaml
PYTHONPATH=src python src/eval/run_eval.py --config configs/finance_support.yaml
```

Validated across 2 domains with **zero pipeline code changes**:
- `configs/generic_support.yaml` — general FAQ support
- `configs/finance_support.yaml` — generic informational finance FAQ only
  (payment due dates, autopay, hardship basics) — explicitly refuses
  account-specific/regulatory questions

See `benchmarks/results/comparison.md` for a side-by-side of both domains'
real eval results.

### Fine-tuning — `src/finetune/` (data prep local, training on Colab)
- `build_sft_dataset.py`: generates 95 synthetic (prompt, ideal_response) pairs
  for the `finance_support` domain — on-scope questions answered by the local
  LLM under the domain persona, off-scope questions paired with a fixed
  refusal response. Writes `data/finetune/sft.jsonl`.
- `build_preference_pairs.py`: generates 25 synthetic (prompt, chosen,
  rejected) preference pairs — off-scope prompts pair a refusal (chosen)
  against an unconstrained answer (rejected); on-scope prompts pair a concise
  on-persona answer (chosen) against a deliberately verbose one (rejected).
  Writes `data/finetune/prefs.jsonl`.
- All data here is clearly labeled synthetic, generated by the small local
  LLM — not real customer data.

Training (LoRA SFT + DPO on Qwen2.5-1.5B-Instruct via Unsloth) runs on
**Colab's free T4 GPU**: see `colab/finetune_unsloth.ipynb`. Clone the repo on
Colab, run the notebook, download the resulting adapter checkpoint into
`checkpoints/` locally.

Evaluate the fine-tuned checkpoint locally against the base model:
```bash
PYTHONPATH=src python src/finetune/eval_finetune_comparison.py --adapter_path checkpoints/lora_adapter
```
`LocalLLM(adapter_path=...)` bypasses Ollama and loads the base model + LoRA
adapter directly via `transformers`/`peft` on CPU, since Ollama can't load a
raw LoRA adapter without a GGUF conversion step.

**Status:** data prep is complete and committed (`data/finetune/*.jsonl`).
Actual GPU training in `colab/finetune_unsloth.ipynb` requires manually
running the notebook on Colab — it has GPU-training cells that can't execute
in this local CPU-only environment. `benchmarks/results/finetune_comparison.md`
will be added after that checkpoint is downloaded and evaluated locally.

## Sub-study: Noise Robustness — `src/robustness/`, `benchmarks/run_noise_benchmark.py`
Extends the flagship eval with a WER-vs-SNR robustness study, fully local/CPU.
Reuses `src/asr/whisper_asr.py` (no duplicated ASR code).

- Data: 10 utterances from LibriSpeech dev-clean (via
  `hf-internal-testing/librispeech_asr_dummy`, real dev-clean audio + ground
  truth transcripts) and 5 environmental noise clips from ESC-50
  (`ashraq/esc50`) — dog, chirping birds, vacuum cleaner (x2), thunderstorm.
- `noise_mixer.py` mixes speech + noise at SNR levels [20, 10, 5, 0, -5] dB.
- `wer.py` computes WER via `jiwer` with lowercase/punctuation-normalized text
  (LibriSpeech references are uppercase, unpunctuated — normalizing both
  sides avoids inflating WER with formatting mismatches, not real errors).

```bash
python benchmarks/run_noise_benchmark.py
```

Real result (n=10 utterances, `WhisperASR("small")`):

| condition | mean WER | mean WER (enhanced) |
|---|---|---|
| clean | 0.081 | - |
| 20 dB SNR | 0.066 | 0.077 |
| 10 dB SNR | 0.070 | 0.090 |
| 5 dB SNR | 0.064 | 0.098 |
| 0 dB SNR | 0.110 | 0.198 |
| -5 dB SNR | 0.219 | 0.247 |

WER stays roughly flat down to 5dB, then degrades sharply below 0dB — see
`benchmarks/results/noise_robustness.md` and `.png` for the full table/plot.

`src/robustness/enhancer.py` adds a `noisereduce` (spectral gating) denoising
step and reports enhanced WER + latency (~97ms/utterance) at each SNR level.
**Honest finding:** enhancement made WER *worse* at every tested SNR in this
run, not better — spectral gating likely distorts speech features Whisper
relies on more than it removes noise, at this noise type/level combination.
This is a real measured result, not the outcome we expected going in; see the
Interpretation section of `noise_robustness.md` for the actual tradeoff.

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
