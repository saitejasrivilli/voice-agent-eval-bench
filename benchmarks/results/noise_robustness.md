# Noise Robustness — WER vs SNR

Sample size: **10** LibriSpeech dev-clean utterances (`hf-internal-testing/librispeech_asr_dummy`), **5** ESC-50 environmental noise clips (dog, chirping_birds, vacuum_cleaner, vacuum_cleaner, thunderstorm), ASR: WhisperASR (small).

| condition | mean WER | mean WER (enhanced) | enhancement latency (ms) |
|---|---|---|---|
| clean | 0.081 | - | - |
| 20 dB SNR | 0.066 | 0.077 | 96.6 |
| 10 dB SNR | 0.070 | 0.090 | 96.6 |
| 5 dB SNR | 0.064 | 0.098 | 96.6 |
| 0 dB SNR | 0.110 | 0.198 | 96.6 |
| -5 dB SNR | 0.219 | 0.247 | 96.6 |

## Interpretation

Enhancement did not reduce WER at any tested SNR in this run — its added latency (~97ms per utterance) is not justified here.
