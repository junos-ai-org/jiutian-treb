"""Side-by-side model comparison on TReB.

Usage:
    python scripts/compare_models.py \
        --qwen_json results/qwen_summary.json \
        --t5gemma_json results/t5gemma_summary.json
"""

import argparse
import json


def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(qwen: dict, t5gemma: dict, metric: str):
    """Print comparison table for a single metric."""
    all_tasks = sorted(set(qwen["tasks"]) | set(t5gemma["tasks"]))

    print(f"\n### {metric}\n")
    print("| Task | Qwen2.5-14B | T5Gemma-9B | Delta (Qwen-T5) |")
    print("|---|---|---|---|")

    qwen_scores, t5_scores = [], []
    for task in all_tasks:
        q = qwen["tasks"].get(task, {}).get(metric)
        t = t5gemma["tasks"].get(task, {}).get(metric)
        if q is not None and t is not None:
            delta = round(q - t, 4)
            sign = "+" if delta > 0 else ""
            bold = "**" if abs(delta) >= 0.05 else ""
            print(f"| {task} | {q:.4f} | {t:.4f} | {bold}{sign}{delta:.4f}{bold} |")
            qwen_scores.append(q)
            t5_scores.append(t)
        else:
            q_str = f"{q:.4f}" if q is not None else "—"
            t_str = f"{t:.4f}" if t is not None else "—"
            print(f"| {task} | {q_str} | {t_str} | — |")

    if qwen_scores and t5_scores:
        q_avg = sum(qwen_scores) / len(qwen_scores)
        t_avg = sum(t5_scores) / len(t5_scores)
        delta = round(q_avg - t_avg, 4)
        sign = "+" if delta > 0 else ""
        print(f"| **OVERALL** | **{q_avg:.4f}** | **{t_avg:.4f}** | **{sign}{delta:.4f}** |")


def main():
    parser = argparse.ArgumentParser(description="Compare models on TReB")
    parser.add_argument("--qwen_json", required=True)
    parser.add_argument("--t5gemma_json", required=True)
    args = parser.parse_args()

    qwen = load_summary(args.qwen_json)
    t5gemma = load_summary(args.t5gemma_json)

    # Find common metrics
    metrics = set()
    for task_scores in qwen["tasks"].values():
        metrics.update(k for k in task_scores if k != "n")
    metrics = sorted(metrics)

    print("# TReB: Qwen2.5-14B vs T5Gemma-9B Comparison")
    for metric in metrics:
        compare(qwen, t5gemma, metric)


if __name__ == "__main__":
    main()
