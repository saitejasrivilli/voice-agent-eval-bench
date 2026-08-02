import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.audio_utils import generate_speech_wav
from common.config import load_domain_config, load_test_questions
from eval.latency_report import latency_report_markdown
from eval.llm_judge import LLMJudge
from pipeline.voice_agent import VoiceAgentPipeline


def run(config_path: str, out_report_path: str) -> None:
    config = load_domain_config(config_path)
    questions = load_test_questions(config)
    pipeline = VoiceAgentPipeline.from_config(config_path)
    judge = LLMJudge()

    rows = []
    traces = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, question in enumerate(questions):
            in_path = os.path.join(tmp_dir, f"in_{i}.wav")
            out_path = os.path.join(tmp_dir, f"out_{i}.wav")
            generate_speech_wav(in_path, question)

            trace = pipeline.run(in_path, out_path)
            score = judge.score(trace["transcript"], trace["response_text"])

            rows.append({"question": question, "trace": trace, "score": score})
            traces.append(trace)
            print(f"[{i + 1}/{len(questions)}] {question!r} -> scored")

    n = len(rows)
    weights = config["eval_rubric_weights"]
    avg_factual = sum(r["score"]["factual_correctness"] for r in rows) / n
    avg_conciseness = sum(r["score"]["conciseness"] for r in rows) / n
    refusal_rate = sum(r["score"]["refusal_appropriate"] for r in rows) / n
    weighted_score = (
        weights["factual_correctness"] * (avg_factual / 5)
        + weights["refusal_appropriate"] * refusal_rate
        + weights["conciseness"] * (avg_conciseness / 5)
    )

    failures = [r for r in rows if r["score"]["factual_correctness"] <= 2 or not r["score"]["refusal_appropriate"]]

    md = []
    md.append(f"# Eval Report — {config['domain']}\n")
    md.append(f"Sample size: **{n}** synthetic scripted customer questions ({config['domain']} domain).\n")
    md.append(
        "Judge: default local LLM (small model via Ollama) — noted limitation, "
        "small judge models are less reliable than larger ones. See TODO in `src/eval/llm_judge.py` for a Colab-based stronger-judge upgrade path.\n"
    )
    md.append(f"Rubric weights: `{weights}`\n")
    md.append("## Aggregate scores\n")
    md.append(f"- Avg factual_correctness: **{avg_factual:.2f}** / 5")
    md.append(f"- Avg conciseness: **{avg_conciseness:.2f}** / 5")
    md.append(f"- Refusal-appropriate rate: **{refusal_rate * 100:.0f}%**")
    md.append(f"- Weighted domain score: **{weighted_score:.3f}** (0-1)\n")
    md.append("## Latency\n")
    md.append(latency_report_markdown(traces))
    md.append("\n## Per-sample scores\n")
    md.append("| # | question | factual | refusal_ok | conciseness | total_ms |")
    md.append("|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        s = r["score"]
        md.append(
            f"| {i + 1} | {r['question']} | {s['factual_correctness']} | "
            f"{s['refusal_appropriate']} | {s['conciseness']} | {r['trace']['total_ms']:.0f} |"
        )

    md.append("\n## Failure cases\n")
    if failures:
        for r in failures:
            md.append(f"- **{r['question']}** -> {r['trace']['response_text']!r} (rationale: {r['score']['rationale']})")
    else:
        md.append("None — all samples scored factual_correctness >= 3 and refusal_appropriate = true.")

    with open(out_report_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nWrote report to {out_report_path}")
    print(
        f"Avg factual_correctness: {avg_factual:.2f}, Avg conciseness: {avg_conciseness:.2f}, "
        f"refusal_rate: {refusal_rate * 100:.0f}%, weighted_score: {weighted_score:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generic_support.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out
    if out is None:
        domain = load_domain_config(args.config)["domain"]
        out = f"benchmarks/results/eval_report_{domain}.md"

    run(args.config, out)
