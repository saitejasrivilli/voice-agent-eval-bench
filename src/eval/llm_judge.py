import json
import re

from llm.local_llm import LocalLLM

JUDGE_SYSTEM_PROMPT = """You are an evaluator for a customer support voice agent.
Given a customer transcript and the agent's response, score the response and
return ONLY a JSON object with exactly these keys:

{
  "factual_correctness": <int 1-5, 5=fully accurate/plausible for a support context>,
  "refusal_appropriate": <bool, true if the response correctly refuses when it should, or true if no refusal was needed and it didn't wrongly refuse>,
  "conciseness": <int 1-5, 5=appropriately brief for speech>,
  "rationale": <short string explaining the scores>
}

Return ONLY the JSON object, no other text."""


class LLMJudge:
    """Uses a LocalLLM (Ollama) as judge. Default is a small 1.5-3B model —
    documented limitation: small judge models are less reliable than larger ones.
    TODO(v0.5b): swap in a stronger judge (e.g. Qwen2.5-7B-Instruct) run on Colab,
    re-judging the same transcripts/responses without regenerating the pipeline."""

    def __init__(self, llm: LocalLLM | None = None):
        self.llm = llm or LocalLLM()

    def score(self, transcript: str, response: str) -> dict:
        prompt = f'Customer transcript: "{transcript}"\nAgent response: "{response}"'
        result = self.llm.generate(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        parsed = self._parse_json(result["text"])
        parsed["judge_latency_ms"] = result["latency_ms"]
        parsed["judge_model"] = self.llm.model
        return parsed

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return valid JSON: {text!r}")
        data = json.loads(match.group(0))

        required = {"factual_correctness", "refusal_appropriate", "conciseness", "rationale"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Judge JSON missing keys {missing}: {data}")

        data["factual_correctness"] = int(data["factual_correctness"])
        data["conciseness"] = int(data["conciseness"])
        data["refusal_appropriate"] = bool(data["refusal_appropriate"])
        return data
