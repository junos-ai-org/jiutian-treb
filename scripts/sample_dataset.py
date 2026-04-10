"""Sample TReB datasets from HuggingFace for bidir_attn experiments.

Downloads the English subset from JT-LM/JIUTIAN-TReB, groups by task,
applies stratified sampling, and outputs individual JSON files per task
in TReB's expected format.

Usage:
    python scripts/sample_dataset.py --size smoke --output experiments/bidir_attn/data/smoke/ --seed 42
    python scripts/sample_dataset.py --size large --output experiments/bidir_attn/data/large/ --seed 42
"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


# Fields expected by Sample constructor
SAMPLE_FIELDS = [
    "id", "file_path", "instruction", "question", "answer",
    "title", "columnslable", "Table_markdown", "Table_html", "number_answer",
]


def extract_task_name(sample_id: str) -> str:
    """Extract task name from the id field (everything before first '|')."""
    return sample_id.split("|")[0]


def group_by_task(dataset) -> dict[str, list[dict]]:
    """Group dataset records by task name."""
    groups = defaultdict(list)
    for row in dataset:
        record = {k: row[k] for k in SAMPLE_FIELDS if k in row}
        task = extract_task_name(record["id"])
        groups[task].append(record)
    return dict(groups)


def stratified_sample(
    groups: dict[str, list[dict]],
    total: int,
    floor: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Stratified sampling with a per-task floor.

    1. Each task gets at least `floor` samples (or all if fewer available).
    2. Remaining quota allocated proportionally to task size.
    """
    rng = random.Random(seed)
    task_sizes = {t: len(samples) for t, samples in groups.items()}
    n_tasks = len(groups)

    # Phase 1: allocate floor
    allocations = {}
    reserved = 0
    for task, size in task_sizes.items():
        alloc = min(floor, size)
        allocations[task] = alloc
        reserved += alloc

    # Phase 2: distribute remaining proportionally
    remaining = total - reserved
    if remaining > 0:
        # Tasks that can still give more samples
        eligible = {t: task_sizes[t] - allocations[t]
                    for t in groups if task_sizes[t] > allocations[t]}
        total_eligible = sum(eligible.values())

        if total_eligible > 0:
            for task, extra_capacity in eligible.items():
                proportional = math.floor(remaining * extra_capacity / total_eligible)
                add = min(proportional, extra_capacity)
                allocations[task] += add

            # Distribute any leftover one-by-one to largest tasks
            current_total = sum(allocations.values())
            shortfall = total - current_total
            sorted_tasks = sorted(eligible.keys(),
                                  key=lambda t: task_sizes[t], reverse=True)
            for task in sorted_tasks:
                if shortfall <= 0:
                    break
                can_add = task_sizes[task] - allocations[task]
                if can_add > 0:
                    allocations[task] += 1
                    shortfall -= 1

    # Phase 3: sample
    sampled = {}
    for task, samples in groups.items():
        n = allocations[task]
        if n >= len(samples):
            sampled[task] = list(samples)
        else:
            sampled[task] = rng.sample(samples, n)
    return sampled


def smoke_sample(
    groups: dict[str, list[dict]],
    per_task: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Simple per-task sampling for smoke tests."""
    rng = random.Random(seed)
    sampled = {}
    for task, samples in groups.items():
        n = min(per_task, len(samples))
        sampled[task] = rng.sample(samples, n)
    return sampled


def nullify_csv_paths(samples: list[dict]) -> list[dict]:
    """Clear file_path so Sample falls back to Table_markdown.

    The CSV files referenced in file_path are relative to the original
    dataset location and won't exist in our experiment directory.
    TCoT_md mode reads from Table_markdown anyway.
    """
    for s in samples:
        s["file_path"] = [""]
    return samples


def save_task_json(task: str, samples: list[dict], output_dir: Path):
    """Save samples as a JSON list (TReB's expected format)."""
    path = output_dir / f"{task}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def save_manifest(sampled: dict[str, list[dict]], groups: dict[str, list[dict]],
                  output_dir: Path):
    """Write manifest.csv documenting sample counts."""
    path = output_dir / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "n_total", "n_sampled", "file"])
        for task in sorted(sampled.keys()):
            writer.writerow([
                task,
                len(groups[task]),
                len(sampled[task]),
                f"{task}.json",
            ])


def main():
    parser = argparse.ArgumentParser(description="Sample TReB datasets")
    parser.add_argument("--size", choices=["smoke", "large"], required=True)
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for sampled JSON files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_per_task", type=int, default=3,
                        help="Samples per task for smoke size")
    parser.add_argument("--large_total", type=int, default=1000,
                        help="Total samples for large size")
    parser.add_argument("--large_floor", type=int, default=10,
                        help="Minimum samples per task for large size")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TReB dataset from HuggingFace (JT-LM/JIUTIAN-TReB)...")
    dataset = load_dataset("JT-LM/JIUTIAN-TReB", split="train")
    print(f"  Loaded {len(dataset)} total samples.")

    groups = group_by_task(dataset)
    print(f"  Found {len(groups)} tasks: {sorted(groups.keys())}")
    for task in sorted(groups.keys()):
        print(f"    {task}: {len(groups[task])}")

    if args.size == "smoke":
        sampled = smoke_sample(groups, args.smoke_per_task, args.seed)
    else:
        sampled = stratified_sample(
            groups, args.large_total, args.large_floor, args.seed)

    total_sampled = sum(len(v) for v in sampled.values())
    print(f"\nSampled {total_sampled} total across {len(sampled)} tasks:")

    for task in sorted(sampled.keys()):
        samples = nullify_csv_paths(sampled[task])
        save_task_json(task, samples, output_dir)
        print(f"  {task}: {len(samples)} samples → {task}.json")

    save_manifest(sampled, groups, output_dir)
    print(f"\nManifest written to {output_dir / 'manifest.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
