# Voice Agent Eval Bench

A voice-agent evaluation benchmark: ASR, TTS, VAD, an LLM pipeline, and an
eval harness with an LLM judge — built to run fully locally on Apple Silicon,
plus two sub-studies (noise robustness, multilingual turn detection) and a
LoRA fine-tuning workflow.

## Architecture

```
audio in -> VAD (Silero, trims silence)
         -> ASR (faster-whisper, CPU int8)
         -> LocalLLM (Ollama, domain persona from configs/*.yaml)
         -> TTS (Piper) -> audio out
                |
                v
       LLMJudge scores (transcript, response)
       -> benchmarks/results/eval_report_<domain>.md
```
See `src/pipeline/voice_agent.py` for the orchestrating code.

## What's local vs. Colab, and why

| Stage | Where | Why |
|---|---|---|
| ASR, TTS, VAD, LLM pipeline, eval harness | Local (M2 CPU) | All small enough to run fast on CPU; no GPU benefit |
| Domain configs, both sub-studies (noise robustness, multilingual VAD) | Local (M2 CPU) | Same — CPU-friendly models/datasets throughout |
| LoRA SFT + DPO fine-tuning | **Local CPU (ran here, not Colab)** | No GPU/Colab access in this environment; trained locally instead with plain transformers/peft/trl (Unsloth needs CUDA) — slower (~15hrs total vs. Unsloth/T4's expected minutes), but real measured numbers. `colab/finetune_unsloth.ipynb` remains available if you want the fast GPU path later |
| Stronger judge model (optional, v0.5b) | Colab (free T4 GPU) | Optional polish — a 7B judge doesn't fit comfortably in this CPU setup |

## Reports (all from real measured runs)

- `benchmarks/results/eval_report_generic_support.md`, `eval_report_finance_support.md`, `comparison.md` — eval harness results per domain
- `benchmarks/results/noise_robustness.md` (+ `.png`) — WER vs SNR, with/without enhancement
- `benchmarks/results/multilingual_vad.md` — VAD boundary accuracy across 4 languages
- `benchmarks/results/finetune_comparison.md` — base vs. LoRA-fine-tuned comparison (trained locally on CPU, see Fine-tuning section below)

## Honest scope

- All eval data (test questions, SFT/preference-pair prompts) is **synthetic**,
  clearly labeled as such in the code and data files — not real customer data.
- Judge model is small (3B via Ollama) by default; documented as a reliability
  limitation, with measured evidence in the eval reports themselves (e.g. the
  `refusal_appropriate` scoring noise).
- Sample sizes are small everywhere (10-15 per experiment) and stated plainly
  in each report — this is a benchmark harness demonstration, not a
  statistically powered study.
- Latency numbers are local CPU numbers on one M2 MacBook, not production
  infrastructure numbers — a real deployment would need GPU-backed ASR/LLM
  serving, load balancing, and streaming (not batch) audio I/O to hit
  production latency targets.
- Mozilla Common Voice (the originally planned multilingual dataset) requires
  HF dataset-gate auth unavailable in this environment; `google/fleurs` was
  substituted — also a standard, open multilingual speech corpus.
- LoRA SFT + DPO fine-tuning ran locally on CPU (8GB RAM Mac, no GPU/Colab
  access in this environment) via plain `transformers`/`peft`/`trl` instead
  of Unsloth on a Colab T4 — SFT took ~2.5hrs, DPO took ~12.4hrs (real
  numbers in `checkpoints/training_stats.json`). The `finetune_comparison.md`
  latency numbers compare Ollama (base) vs. raw HF CPU generation
  (fine-tuned) — an inference-backend artifact, not a fair reflection of the
  LoRA weights' own speed; see the report's Interpretation section.

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
  `benchmarks/results/eval_report_generic_support.md`. TODO hook for a stronger Colab-hosted
  judge (Qwen2.5-7B-Instruct) is left in the code (v0.5b, optional).
- `latency_report.py`: aggregates pipeline traces into P50/P95/P99 per stage.
- `run_eval.py`: CLI that runs the full pipeline + judge over 12 synthetic
  scripted customer questions (`data/samples/test_questions.py`, generic
  support domain, built with our own Piper TTS), writes
  `benchmarks/results/eval_report_<domain>.md` with per-sample scores, aggregates,
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

### Fine-tuning — `src/finetune/`
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

**Training ran locally on CPU**, not Colab: `src/finetune/train_local_lora.py`
(SFT) + `src/finetune/resume_dpo.py` (DPO, resumes from the saved SFT
adapter). This environment had no GPU/Colab access, so training used plain
`transformers`/`peft`/`trl` on an 8GB-RAM Mac instead of Unsloth on a T4 —
`colab/finetune_unsloth.ipynb` is still available as the fast-path option if
you want to redo this on a real GPU later. LoRA config matches the plan
(r=8, alpha=16, q/k/v/o_proj). Real measured times:
SFT 8937s (~2.5hrs), DPO 44527s (~12.4hrs) — see `checkpoints/training_stats.json`.

Evaluate the fine-tuned checkpoint locally against the base model:
```bash
PYTHONPATH=src FINETUNE_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct" python src/finetune/eval_finetune_comparison.py --adapter_path checkpoints/lora_adapter
```
`LocalLLM(adapter_path=...)` bypasses Ollama and loads the base model + LoRA
adapter directly via `transformers`/`peft` on CPU, since Ollama can't load a
raw LoRA adapter without a GGUF conversion step.

Real result (n=12, `benchmarks/results/finetune_comparison.md`): refusal-
appropriate rate improved 25% -> 50% (the behavior DPO's preference data
specifically targeted); factual_correctness dipped slightly (4.08 -> 3.83,
noise-level at this sample size); the latency columns in that report compare
different inference backends (Ollama vs. raw HF CPU generation), not the
LoRA weights' own speed — see the report for the full caveat.

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

## Sub-study: Multilingual Turn Detection — `src/multilingual/`, `benchmarks/run_multilingual_vad.py`
Extends the flagship eval with a per-language VAD boundary-accuracy study,
fully local/CPU. Reuses `src/vad/silero_vad.py` (no duplicated VAD code).

- Data: 15 utterances per language (English, Spanish, German, Mandarin) from
  `google/fleurs`. **Note:** Mozilla Common Voice requires HF dataset-gate
  auth not available in this environment — FLEURS was substituted, also a
  standard, license-open multilingual speech corpus.
- Metric: FLEURS clips are trimmed read-speech with minimal silence, so we
  treat the full clip span as expected speech and compute boundary error
  (mean |detected onset - 0| and |detected offset - clip end|, seconds) plus
  a coverage ratio (detected speech duration / clip duration).

```bash
python benchmarks/run_multilingual_vad.py
```

Real result (n=15/language, Silero VAD):

| language | mean coverage ratio |
|---|---|
| english | 0.787 |
| spanish | 0.711 |
| german | 0.730 |
| mandarin | 0.656 |

Mandarin shows the lowest coverage — see `benchmarks/results/multilingual_vad.md`
for the full table and clearly-labeled hypotheses (not confirmed root causes)
about why.

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
