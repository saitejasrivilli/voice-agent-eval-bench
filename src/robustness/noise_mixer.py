import numpy as np


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mixes speech and noise at the given SNR (dB). Noise is tiled/cropped to
    match speech length."""
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(speech)]

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return speech.copy()

    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = speech + scale * noise

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)
