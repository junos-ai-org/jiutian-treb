#!/usr/bin/env python3
"""Stratified subsample of the 7018-id filtered pool down to ~1000 ids.

Strategy: for each task, take a proportional slice (round(total * 1000 / 7018))
with a floor of min(all-available, 20). Small tasks (<20) are kept whole so
per-category reads stay well-powered. Seeded for reproducibility.

Writes:
  - dataset/sample_1000.json  — list of ids
  - dataset/sample_1000_stats.json — per-task counts

Run:
  python experiments/t5gemmav1_vs_qwen3_treb/sample_1000.py --n 1000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept-ids", default=None,
                    help="Path to kept_ids.json from phase-0 filter.")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min-per-task", type=int, default=20,
                    help="Floor per task; small tasks may fall below.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    kept_path = Path(args.kept_ids) if args.kept_ids else here / "dataset/kept_ids.json"
    out_path = Path(args.out) if args.out else here / "dataset/sample_1000.json"
    stats_path = Path(args.stats_out) if args.stats_out else here / "dataset/sample_1000_stats.json"

    all_ids = json.loads(kept_path.read_text())
    total = len(all_ids)

    by_task: dict[str, list[str]] = defaultdict(list)
    for sid in all_ids:
        by_task[sid.split("|")[0]].append(sid)

    rng = random.Random(args.seed)
    selected: list[str] = []
    per_task_counts: dict[str, dict[str, int]] = {}

    for task, ids in sorted(by_task.items()):
        proportional = round(len(ids) * args.n / total)
        target = max(min(len(ids), args.min_per_task), proportional)
        target = min(target, len(ids))
        sample = rng.sample(ids, target)
        selected.extend(sample)
        per_task_counts[task] = {
            "available": len(ids),
            "sampled": target,
        }

    if len(selected) > args.n:
        # Trim largest over-sampled tasks first until we hit n.
        from collections import Counter as _C
        while len(selected) > args.n:
            cur_counts = _C(sid.split("|")[0] for sid in selected)
            biggest = max(cur_counts, key=lambda t: cur_counts[t])
            victims = [sid for sid in selected if sid.startswith(biggest + "|")]
            drop = rng.choice(victims)
            selected.remove(drop)
            per_task_counts[biggest]["sampled"] -= 1

    rng.shuffle(selected)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, indent=2))

    stats = {
        "seed": args.seed,
        "n_requested": args.n,
        "n_selected": len(selected),
        "min_per_task_floor": args.min_per_task,
        "per_task": per_task_counts,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[sample] wrote {len(selected)} ids to {out_path}")
    print(f"[sample] wrote stats to {stats_path}")
    distribution = Counter(sid.split("|")[0] for sid in selected)
    for t, c in distribution.most_common():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
