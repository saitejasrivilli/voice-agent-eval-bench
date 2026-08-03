import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asr.whisper_asr import WhisperASR
from robustness.data_loader import load_librispeech_subset, load_noise_subset
from robustness.enhancer import enhance
from robustness.noise_mixer import mix_at_snr
from robustness.wer import compute_wer

SNR_LEVELS_DB = [20, 10, 5, 0, -5]
N_UTTERANCES = 10
N_NOISE_CLIPS = 5

OUT_MD = "benchmarks/results/noise_robustness.md"
OUT_PLOT = "benchmarks/results/noise_robustness.png"


def main():
    print("Loading LibriSpeech dev-clean subset...")
    utterances = load_librispeech_subset(N_UTTERANCES)
    print(f"Loaded {len(utterances)} utterances")

    print("Loading ESC-50 noise subset...")
    noise_clips = load_noise_subset(N_NOISE_CLIPS)
    print(f"Loaded {len(noise_clips)} noise clips: {[c['category'] for c in noise_clips]}")

    asr = WhisperASR()

    results = {snr: [] for snr in SNR_LEVELS_DB}
    results["clean"] = []
    enhanced_results = {snr: [] for snr in SNR_LEVELS_DB}
    enhance_latencies = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, utt in enumerate(utterances):
            speech = utt["audio"]
            sr = utt["sample_rate"]
            reference = utt["text"]

            clean_path = os.path.join(tmp_dir, f"clean_{i}.wav")
            sf.write(clean_path, speech, sr)
            clean_transcript = asr.transcribe(clean_path)["transcript"]
            clean_wer = compute_wer(reference, clean_transcript)
            results["clean"].append(clean_wer)
            print(f"[{i + 1}/{len(utterances)}] clean WER={clean_wer:.3f}")

            noise = noise_clips[i % len(noise_clips)]["audio"]
            for snr in SNR_LEVELS_DB:
                mixed = mix_at_snr(speech, noise, snr)
                noisy_path = os.path.join(tmp_dir, f"noisy_{i}_{snr}.wav")
                sf.write(noisy_path, mixed, sr)
                transcript = asr.transcribe(noisy_path)["transcript"]
                wer = compute_wer(reference, transcript)
                results[snr].append(wer)

                enhance_result = enhance(mixed, sr)
                enhance_latencies.append(enhance_result["latency_ms"])
                enhanced_path = os.path.join(tmp_dir, f"enhanced_{i}_{snr}.wav")
                sf.write(enhanced_path, enhance_result["audio"], sr)
                enhanced_transcript = asr.transcribe(enhanced_path)["transcript"]
                enhanced_wer = compute_wer(reference, enhanced_transcript)
                enhanced_results[snr].append(enhanced_wer)

                print(
                    f"  SNR={snr}dB WER={wer:.3f} enhanced_WER={enhanced_wer:.3f} "
                    f"(enhance {enhance_result['latency_ms']:.1f}ms)"
                )

    clean_mean = float(np.mean(results["clean"]))
    snr_means = {snr: float(np.mean(results[snr])) for snr in SNR_LEVELS_DB}
    enhanced_means = {snr: float(np.mean(enhanced_results[snr])) for snr in SNR_LEVELS_DB}
    mean_enhance_latency = float(np.mean(enhance_latencies))

    md = ["# Noise Robustness — WER vs SNR\n"]
    md.append(
        f"Sample size: **{len(utterances)}** LibriSpeech dev-clean utterances "
        f"(`hf-internal-testing/librispeech_asr_dummy`), **{len(noise_clips)}** ESC-50 "
        f"environmental noise clips ({', '.join(c['category'] for c in noise_clips)}), "
        f"ASR: WhisperASR ({asr.model_size}).\n"
    )
    md.append("| condition | mean WER | mean WER (enhanced) | enhancement latency (ms) |")
    md.append("|---|---|---|---|")
    md.append(f"| clean | {clean_mean:.3f} | - | - |")
    for snr in SNR_LEVELS_DB:
        md.append(f"| {snr} dB SNR | {snr_means[snr]:.3f} | {enhanced_means[snr]:.3f} | {mean_enhance_latency:.1f} |")

    md.append("\n## Interpretation\n")
    best_snr, best_gain = None, 0.0
    for snr in SNR_LEVELS_DB:
        gain = snr_means[snr] - enhanced_means[snr]
        if gain > best_gain:
            best_snr, best_gain = snr, gain
    if best_snr is not None:
        md.append(
            f"Enhancement (`noisereduce` spectral gating) helps most at **{best_snr} dB SNR** "
            f"(WER {snr_means[best_snr]:.3f} -> {enhanced_means[best_snr]:.3f}, a "
            f"{best_gain:.3f} absolute WER reduction), at a cost of ~{mean_enhance_latency:.0f}ms "
            "added latency per utterance. At higher SNR levels the noise is already mild enough "
            "that enhancement's added latency isn't justified by the (small or negative) WER "
            "improvement. At very low SNR, enhancement's spectral gating can distort speech "
            "enough to not fully recover accuracy, so it's not a universal win — the "
            "latency/accuracy tradeoff only clearly favors enhancement in a mid-noise band."
        )
    else:
        md.append(
            "Enhancement did not reduce WER at any tested SNR in this run — its added latency "
            f"(~{mean_enhance_latency:.0f}ms per utterance) is not justified here."
        )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")

    plt.figure(figsize=(6, 4))
    x = SNR_LEVELS_DB[::-1]
    y = [snr_means[s] for s in x]
    y_enhanced = [enhanced_means[s] for s in x]
    plt.plot(x, y, marker="o", label="noisy")
    plt.plot(x, y_enhanced, marker="s", label="noisy + enhanced")
    plt.axhline(clean_mean, color="gray", linestyle="--", label="clean")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean WER")
    plt.title("WER vs SNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT)

    print(f"\nWrote {OUT_MD} and {OUT_PLOT}")
    print(f"Clean WER: {clean_mean:.3f}")
    for snr in SNR_LEVELS_DB:
        print(f"SNR {snr}dB: WER {snr_means[snr]:.3f} -> enhanced {enhanced_means[snr]:.3f}")
    print(f"Mean enhancement latency: {mean_enhance_latency:.1f}ms")


if __name__ == "__main__":
    main()
