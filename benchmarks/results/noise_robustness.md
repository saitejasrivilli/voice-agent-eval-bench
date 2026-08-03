# Noise Robustness — WER vs SNR

Sample size: **10** LibriSpeech dev-clean utterances (`hf-internal-testing/librispeech_asr_dummy`), **5** ESC-50 environmental noise clips (dog, chirping_birds, vacuum_cleaner, vacuum_cleaner, thunderstorm), ASR: WhisperASR (small).

| condition | mean WER |
|---|---|
| clean | 0.081 |
| 20 dB SNR | 0.066 |
| 10 dB SNR | 0.070 |
| 5 dB SNR | 0.064 |
| 0 dB SNR | 0.110 |
| -5 dB SNR | 0.219 |
