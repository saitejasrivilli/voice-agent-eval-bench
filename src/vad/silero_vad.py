import numpy as np
import soundfile as sf
import torch


class SileroVAD:
    """CPU-based voice activity detection using Silero VAD (tiny model, torch.hub)."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, force_reload=False
        )
        self.get_speech_timestamps = utils[0]

    def _load_wav(self, audio_path: str) -> torch.Tensor:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            indices = np.round(np.arange(0, len(data), sr / self.sample_rate)).astype(int)
            indices = indices[indices < len(data)]
            data = data[indices]
        return torch.from_numpy(data)

    def detect_turns(self, audio_path: str) -> list[dict]:
        """Returns list of {"start": float, "end": float} speech segments, in seconds."""
        wav = self._load_wav(audio_path)
        timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sample_rate)
        return [
            {"start": t["start"] / self.sample_rate, "end": t["end"] / self.sample_rate}
            for t in timestamps
        ]

    def is_endpoint(self, audio_chunk: np.ndarray, silence_threshold_ms: float = 700) -> bool:
        """Returns True if the tail of audio_chunk contains at least
        silence_threshold_ms of continuous non-speech (i.e. the speaker has
        likely finished their turn)."""
        tensor = torch.from_numpy(audio_chunk).float()
        timestamps = self.get_speech_timestamps(tensor, self.model, sampling_rate=self.sample_rate)
        if not timestamps:
            return len(audio_chunk) / self.sample_rate * 1000 >= silence_threshold_ms

        last_speech_end_sec = timestamps[-1]["end"] / self.sample_rate
        chunk_duration_sec = len(audio_chunk) / self.sample_rate
        trailing_silence_ms = (chunk_duration_sec - last_speech_end_sec) * 1000
        return trailing_silence_ms >= silence_threshold_ms
