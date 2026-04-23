#!/usr/bin/env python3
"""Quick side-by-side comparison of T5Gemma v1 vs Qwen3 on the same 250 samples.

Consumes two predictions.jsonl files, computes per-task and overall
ROUGE-L / EM / extract-rate, and writes a markdown table to stdout.

This is deliberately simple — the full LLM-judge pass (via score.py with
--judge deepseek-v3) comes later. This gives us a first read to decide
whether the full judge is worth running.

Usage:
  python compare.py \\
    results/t5gemma_v1_2b2b_ul2_it/predictions.jsonl \\
    results/qwen3_4b_instruct/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Reuse score.py's helpers so metric definitions stay consistent.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import exact_match, extract_json_answer, numeric_match, rouge_l  # noqa: E402


def summarize(path: Path) -> tuple[str, dict]:
    preds = [json.loads(l) for l in path.open()]
    total = len(preds)
    by_task: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "extract": 0, "em": 0.0, "num": 0.0, "rouge": 0.0, "num_n": 0
    })
    for r in preds:
        task = r["task"]
        gold = (r.get("gold_answer") or "").strip()
        pred_text = r.get("prediction", "") or ""
        ans = extract_json_answer(pred_text)
        by_task[task]["n"] += 1
        if ans and ans.strip():
            by_task[task]["extract"] += 1
        by_task[task]["em"] += exact_match(ans, gold)
        by_task[task]["rouge"] += rouge_l(ans, gold)
        num_gold = r.get("number_answer")
        if num_gold not in (None, ""):
            try:
                by_task[task]["num"] += numeric_match(ans, float(num_gold))
                by_task[task]["num_n"] += 1
            except (TypeError, ValueError):
                pass
    return path.name, {
        "total": total,
        "by_task": {t: v for t, v in sorted(by_task.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="first predictions.jsonl")
    ap.add_argument("b", help="second predictions.jsonl")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    _, sa = summarize(Path(args.a))
    _, sb = summarize(Path(args.b))

    # Header
    print(f"# {args.label_a} vs {args.label_b}\n")
    print(f"- {args.label_a}: n = {sa['total']}  ({args.a})")
    print(f"- {args.label_b}: n = {sb['total']}  ({args.b})\n")

    print("| Task | n | "
          f"{args.label_a} EM / ROUGE-L / extract | "
          f"{args.label_b} EM / ROUGE-L / extract |")
    print("|---|---:|---|---|")

    all_tasks = sorted(set(sa["by_task"]) | set(sb["by_task"]))
    agg_a = dict(n=0, em=0.0, rouge=0.0, extract=0)
    agg_b = dict(n=0, em=0.0, rouge=0.0, extract=0)
    for t in all_tasks:
        va = sa["by_task"].get(t, {"n": 0, "em": 0, "rouge": 0, "extract": 0})
        vb = sb["by_task"].get(t, {"n": 0, "em": 0, "rouge": 0, "extract": 0})
        na = va["n"] or 1
        nb = vb["n"] or 1
        agg_a["n"] += va["n"]; agg_a["em"] += va["em"]; agg_a["rouge"] += va["rouge"]; agg_a["extract"] += va["extract"]
        agg_b["n"] += vb["n"]; agg_b["em"] += vb["em"]; agg_b["rouge"] += vb["rouge"]; agg_b["extract"] += vb["extract"]
        print(f"| {t} | {va['n']}/{vb['n']} | "
              f"{va['em']/na:.2f} / {va['rouge']/na:.2f} / {va['extract']}/{va['n']} | "
              f"{vb['em']/nb:.2f} / {vb['rouge']/nb:.2f} / {vb['extract']}/{vb['n']} |")

    na = max(1, agg_a["n"]); nb = max(1, agg_b["n"])
    print(f"| **TOTAL** | {agg_a['n']}/{agg_b['n']} | "
          f"{agg_a['em']/na:.3f} / {agg_a['rouge']/na:.3f} / {agg_a['extract']}/{agg_a['n']} | "
          f"{agg_b['em']/nb:.3f} / {agg_b['rouge']/nb:.3f} / {agg_b['extract']}/{agg_b['n']} |")


if __name__ == "__main__":
    main()
