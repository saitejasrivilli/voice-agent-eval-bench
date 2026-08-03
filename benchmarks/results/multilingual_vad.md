# Multilingual VAD Boundary Accuracy

Sample size: **15** utterances per language from `google/fleurs` (substituted for Mozilla Common Voice, which requires HF dataset-gate auth not available in this environment — FLEURS is also a standard, license-open multilingual speech corpus).

Metric: since FLEURS clips are trimmed read-speech with minimal silence, we treat the full clip span as expected speech and measure boundary error (mean |detected onset - 0| and |detected offset - clip end|, seconds) plus a coverage ratio (detected speech duration / clip duration).

| language | n | n_no_speech_detected | mean onset err (s) | mean offset err (s) | mean coverage |
|---|---|---|---|---|---|
| english | 15 | 0 | 0.853 | 0.911 | 0.787 |
| spanish | 15 | 0 | 1.508 | 1.236 | 0.711 |
| german | 15 | 0 | 1.771 | 1.688 | 0.730 |
| mandarin | 15 | 0 | 1.561 | 1.856 | 0.656 |

## Discussion (hypotheses, not confirmed root causes)

- **mandarin** shows the largest deviation from full coverage (mean coverage ratio 0.656). *Hypothesis:* Silero VAD's underlying model was trained predominantly on English/European-language speech data, so languages with different prosody, tonal patterns (e.g. Mandarin), or phoneme energy distribution may be more prone to under- or over-segmentation. This is a hypothesis based on this small sample, not a confirmed root cause — a larger, stratified sample would be needed to confirm.
- *Hypothesis:* differences in FLEURS per-language recording conditions (studio quality, background noise, speaker pacing) could also explain some of the variance independent of the VAD model's language handling — this study did not control for that.
