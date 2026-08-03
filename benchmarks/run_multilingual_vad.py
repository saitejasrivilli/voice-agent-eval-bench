import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multilingual.cv_loader import FLEURS_LANGS, load_fleurs_subset
from vad.silero_vad import SileroVAD

N_PER_LANG = 15
OUT_MD = "benchmarks/results/multilingual_vad.md"

# FLEURS clips are trimmed read-speech utterances with minimal leading/trailing
# silence, so the whole clip's duration is a reasonable proxy "expected speech
# span" ground truth: boundary error = |detected_start - 0| + |detected_end - duration|.


def eval_language(language: str, n: int) -> dict:
    samples = load_fleurs_subset(language, n)
    vad = SileroVAD()

    onset_errors = []
    offset_errors = []
    coverage_ratios = []
    no_speech_detected = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, sample in enumerate(samples):
            path = os.path.join(tmp_dir, f"{language}_{i}.wav")
            sf.write(path, sample["audio"], sample["sample_rate"])
            duration = len(sample["audio"]) / sample["sample_rate"]

            segments = vad.detect_turns(path)
            if not segments:
                no_speech_detected += 1
                continue

            onset_errors.append(abs(segments[0]["start"] - 0.0))
            offset_errors.append(abs(segments[-1]["end"] - duration))
            speech_duration = sum(s["end"] - s["start"] for s in segments)
            coverage_ratios.append(speech_duration / duration)

    n_evaluated = len(onset_errors)
    return {
        "language": language,
        "n": n,
        "n_no_speech": no_speech_detected,
        "mean_onset_error_sec": float(np.mean(onset_errors)) if onset_errors else None,
        "mean_offset_error_sec": float(np.mean(offset_errors)) if offset_errors else None,
        "mean_coverage_ratio": float(np.mean(coverage_ratios)) if coverage_ratios else None,
        "n_evaluated": n_evaluated,
    }


def _purge_fleurs_cache():
    """FLEURS downloads a full per-language archive (~1-2GB) that isn't needed
    once we've extracted our small subset — purge between languages to fit
    limited local disk."""
    home = Path.home()
    for path in [
        home / ".cache/huggingface/hub/datasets--google--fleurs",
        home / ".cache/huggingface/datasets/google___fleurs",
    ]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def main():
    results = []
    for language in FLEURS_LANGS:
        print(f"Evaluating {language}...")
        result = eval_language(language, N_PER_LANG)
        results.append(result)
        print(f"  {result}")
        _purge_fleurs_cache()

    md = ["# Multilingual VAD Boundary Accuracy\n"]
    md.append(
        f"Sample size: **{N_PER_LANG}** utterances per language from "
        "`google/fleurs` (substituted for Mozilla Common Voice, which requires "
        "HF dataset-gate auth not available in this environment — FLEURS is "
        "also a standard, license-open multilingual speech corpus).\n"
    )
    md.append(
        "Metric: since FLEURS clips are trimmed read-speech with minimal "
        "silence, we treat the full clip span as expected speech and measure "
        "boundary error (mean |detected onset - 0| and |detected offset - "
        "clip end|, seconds) plus a coverage ratio (detected speech duration / "
        "clip duration).\n"
    )
    md.append("| language | n | n_no_speech_detected | mean onset err (s) | mean offset err (s) | mean coverage |")
    md.append("|---|---|---|---|---|---|")
    for r in results:
        onset = f"{r['mean_onset_error_sec']:.3f}" if r["mean_onset_error_sec"] is not None else "n/a"
        offset = f"{r['mean_offset_error_sec']:.3f}" if r["mean_offset_error_sec"] is not None else "n/a"
        coverage = f"{r['mean_coverage_ratio']:.3f}" if r["mean_coverage_ratio"] is not None else "n/a"
        md.append(f"| {r['language']} | {r['n']} | {r['n_no_speech']} | {onset} | {offset} | {coverage} |")

    md.append("\n## Discussion (hypotheses, not confirmed root causes)\n")
    worst = max(
        (r for r in results if r["mean_coverage_ratio"] is not None),
        key=lambda r: abs(1.0 - r["mean_coverage_ratio"]),
        default=None,
    )
    if worst:
        md.append(
            f"- **{worst['language']}** shows the largest deviation from full coverage "
            f"(mean coverage ratio {worst['mean_coverage_ratio']:.3f}). *Hypothesis:* Silero VAD's "
            "underlying model was trained predominantly on English/European-language speech data, "
            "so languages with different prosody, tonal patterns (e.g. Mandarin), or phoneme "
            "energy distribution may be more prone to under- or over-segmentation. This is a "
            "hypothesis based on this small sample, not a confirmed root cause — a larger, "
            "stratified sample would be needed to confirm."
        )
    md.append(
        "- *Hypothesis:* differences in FLEURS per-language recording conditions (studio quality, "
        "background noise, speaker pacing) could also explain some of the variance independent of "
        "the VAD model's language handling — this study did not control for that."
    )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
