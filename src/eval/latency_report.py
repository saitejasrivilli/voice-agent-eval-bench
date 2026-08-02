import numpy as np


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def aggregate_latencies(traces: list[dict]) -> dict:
    """Aggregates a list of pipeline trace dicts into P50/P95/P99 per stage."""
    stages = ["asr_ms", "llm_ms", "tts_ms", "total_ms"]
    report = {}
    for stage in stages:
        values = [t[stage] for t in traces if stage in t]
        if not values:
            continue
        report[stage] = {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "mean": float(np.mean(values)),
        }
    return report


def latency_report_markdown(traces: list[dict]) -> str:
    agg = aggregate_latencies(traces)
    lines = ["| stage | mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |", "|---|---|---|---|---|"]
    for stage, stats in agg.items():
        name = stage.replace("_ms", "")
        lines.append(
            f"| {name} | {stats['mean']:.1f} | {stats['p50']:.1f} | {stats['p95']:.1f} | {stats['p99']:.1f} |"
        )
    return "\n".join(lines)
