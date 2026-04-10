"""Aggregate TReB judge output into per-task score summaries.

Reads the JSONL judge output files from eval_output/ and computes
per-task mean scores for each metric (ROUGE, EM, LLM_score).

Usage:
    python scripts/analyze_results.py --version bidir_attn_smoke --model Qwen2.5-14B-Instruct
    python scripts/analyze_results.py --version bidir_attn_large --model t5gemma-9b-9b-ul2-it
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def extract_task(sample_id: str) -> str:
    return sample_id.split("|")[0]


def load_judge_results(eval_output_dir: Path, version: str, model: str) -> list[dict]:
    """Load all judge JSONL files for a given version/model."""
    base = eval_output_dir / version / model
    if not base.exists():
        raise FileNotFoundError(f"No output directory: {base}")

    samples = []
    for path in sorted(base.glob("*--judge.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
    return samples


def aggregate_scores(samples: list[dict]) -> dict:
    """Compute per-task mean scores."""
    task_scores = defaultdict(lambda: defaultdict(list))

    for s in samples:
        task = extract_task(s["id"])
        scores = s.get("judge_scores", {})
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                task_scores[task][metric].append(value)

    summary = {}
    for task in sorted(task_scores):
        summary[task] = {}
        for metric, values in task_scores[task].items():
            summary[task][metric] = round(sum(values) / len(values), 4)
        summary[task]["n"] = len(next(iter(task_scores[task].values())))

    return summary


def compute_overall(summary: dict) -> dict:
    """Compute overall mean across tasks (macro average)."""
    all_metrics = set()
    for scores in summary.values():
        all_metrics.update(k for k in scores if k != "n")

    overall = {}
    for metric in sorted(all_metrics):
        values = [s[metric] for s in summary.values() if metric in s]
        if values:
            overall[metric] = round(sum(values) / len(values), 4)
    overall["n_tasks"] = len(summary)
    return overall


def print_markdown(summary: dict, overall: dict, model: str):
    """Print results as markdown table."""
    metrics = sorted(k for k in overall if k != "n_tasks")

    header = f"| Task | n | {' | '.join(metrics)} |"
    sep = f"|---|---|{'|'.join(['---'] * len(metrics))}|"

    print(f"\n## {model}\n")
    print(header)
    print(sep)
    for task in sorted(summary):
        cols = [str(summary[task].get(m, "—")) for m in metrics]
        print(f"| {task} | {summary[task]['n']} | {' | '.join(cols)} |")
    overall_cols = [str(overall.get(m, "—")) for m in metrics]
    print(f"| **OVERALL** | {overall['n_tasks']} | {' | '.join(overall_cols)} |")


def main():
    parser = argparse.ArgumentParser(description="Analyze TReB judge results")
    parser.add_argument("--version", required=True, help="Experiment version (e.g. bidir_attn_smoke)")
    parser.add_argument("--model", required=True, help="Model name in output dir")
    parser.add_argument("--eval_output", default="../eval_output",
                        help="Path to eval_output directory")
    parser.add_argument("--save_json", default=None, help="Optional: save summary as JSON")
    args = parser.parse_args()

    eval_output_dir = Path(args.eval_output)
    samples = load_judge_results(eval_output_dir, args.version, args.model)
    print(f"Loaded {len(samples)} judged samples.")

    summary = aggregate_scores(samples)
    overall = compute_overall(summary)

    print_markdown(summary, overall, args.model)

    if args.save_json:
        result = {"tasks": summary, "overall": overall, "model": args.model}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.save_json}")


if __name__ == "__main__":
    main()
