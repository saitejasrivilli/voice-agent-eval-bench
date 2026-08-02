import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.audio_utils import generate_speech_wav
from common.config import load_domain_config, load_test_questions
from eval.llm_judge import LLMJudge
from llm.local_llm import LocalLLM
from pipeline.voice_agent import VoiceAgentPipeline


def run_one(pipeline: VoiceAgentPipeline, judge: LLMJudge, questions: list[str], label: str) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, question in enumerate(questions):
            in_path = os.path.join(tmp_dir, f"in_{i}.wav")
            out_path = os.path.join(tmp_dir, f"out_{i}.wav")
            generate_speech_wav(in_path, question)
            trace = pipeline.run(in_path, out_path)
            score = judge.score(trace["transcript"], trace["response_text"])
            rows.append({"question": question, "trace": trace, "score": score})
            print(f"[{label} {i + 1}/{len(questions)}] {question!r} -> scored")
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "avg_factual_correctness": sum(r["score"]["factual_correctness"] for r in rows) / n,
        "avg_conciseness": sum(r["score"]["conciseness"] for r in rows) / n,
        "refusal_rate": sum(r["score"]["refusal_appropriate"] for r in rows) / n,
        "avg_llm_ms": sum(r["trace"]["llm_ms"] for r in rows) / n,
        "avg_total_ms": sum(r["trace"]["total_ms"] for r in rows) / n,
        "n": n,
    }


def main(adapter_path: str, out_path: str) -> None:
    config = load_domain_config("configs/finance_support.yaml")
    questions = load_test_questions(config)
    judge = LLMJudge()

    base_pipeline = VoiceAgentPipeline.from_config("configs/finance_support.yaml")
    base_rows = run_one(base_pipeline, judge, questions, "base")
    base_summary = summarize(base_rows)

    ft_pipeline = VoiceAgentPipeline.from_config("configs/finance_support.yaml")
    ft_pipeline.llm = LocalLLM(adapter_path=adapter_path)
    ft_rows = run_one(ft_pipeline, judge, questions, "fine-tuned")
    ft_summary = summarize(ft_rows)

    md = [
        "# Fine-tune Comparison — finance_support\n",
        f"Sample size: **{base_summary['n']}** (same as domain eval set — this is a small, honest sample; "
        "do not over-read a small before/after gap).\n",
        "| metric | base | fine-tuned |",
        "|---|---|---|",
        f"| avg factual_correctness | {base_summary['avg_factual_correctness']:.2f} | {ft_summary['avg_factual_correctness']:.2f} |",
        f"| avg conciseness | {base_summary['avg_conciseness']:.2f} | {ft_summary['avg_conciseness']:.2f} |",
        f"| refusal-appropriate rate | {base_summary['refusal_rate'] * 100:.0f}% | {ft_summary['refusal_rate'] * 100:.0f}% |",
        f"| avg LLM latency (ms) | {base_summary['avg_llm_ms']:.1f} | {ft_summary['avg_llm_ms']:.1f} |",
        f"| avg total latency (ms) | {base_summary['avg_total_ms']:.1f} | {ft_summary['avg_total_ms']:.1f} |",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--out", default="benchmarks/results/finetune_comparison.md")
    args = parser.parse_args()
    main(args.adapter_path, args.out)
