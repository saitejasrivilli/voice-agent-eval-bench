import time

from asr.whisper_asr import WhisperASR
from common.config import load_domain_config
from llm.local_llm import LocalLLM
from tts.piper_tts import PiperTTS
from vad.silero_vad import SileroVAD

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful customer support voice assistant. Keep replies short, "
    "clear, and conversational since they will be spoken aloud."
)


class VoiceAgentPipeline:
    """Full voice-agent turn: audio in -> VAD trims silence -> ASR transcribes ->
    LocalLLM generates a reply -> TTS synthesizes reply. Returns a trace dict
    with per-stage latency_ms and total_ms."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.vad = SileroVAD()
        self.asr = WhisperASR()
        self.llm = LocalLLM()
        self.tts = PiperTTS()
        self.system_prompt = system_prompt

    @classmethod
    def from_config(cls, config_path: str) -> "VoiceAgentPipeline":
        config = load_domain_config(config_path)
        return cls(system_prompt=config["persona"].strip())

    def run(self, audio_path: str, out_path: str) -> dict:
        start = time.perf_counter()

        segments = self.vad.detect_turns(audio_path)
        asr_result = self.asr.transcribe(audio_path)
        llm_result = self.llm.generate(asr_result["transcript"], system_prompt=self.system_prompt)
        tts_result = self.tts.synthesize(llm_result["text"], out_path)

        total_ms = (time.perf_counter() - start) * 1000

        return {
            "vad_segments": segments,
            "transcript": asr_result["transcript"],
            "response_text": llm_result["text"],
            "response_audio_path": tts_result["audio_path"],
            "asr_ms": asr_result["latency_ms"],
            "llm_ms": llm_result["latency_ms"],
            "tts_ms": tts_result["latency_ms"],
            "total_ms": total_ms,
        }
