#!/usr/bin/env python3
"""Phase 0: build the bug-safe TReB subset for this experiment.

What this does
--------------
1. Loads all TReB English samples (3895 across 26 tasks) via the same path
   `eval.py` uses (snapshot_download of the HF dataset, then the `English/*.json`
   files).
2. Assembles the TCoT prompt (via eval.build_prompt_tcot — in its char-capped
   fallback mode) for each sample.
3. Tokenizes the prompt with the T5Gemma v1 tokenizer
   (`google/t5gemma-2b-2b-ul2-it`, or override via --tokenizer).
4. Drops samples whose token count > --max-tokens (default 4000). Rationale:
   T5Gemma 2 exhibits an SWA-mask-shape bug above ~4094 tokens
   (HF transformers#45521); v1's SWA window differs (4096 vs 1024), so the
   bug may not trigger, but we start conservatively and can relax the cap
   after empirical verification against v1 on the pod.
5. Writes:
   - dataset/kept_ids.json    — list of IDs consumed by both variant configs
                                 via eval.py's `target_ids` mechanism.
   - dataset/dropped_ids.json — IDs that exceeded the cap (for transparency
                                 and optional later runs on the long tail).
   - dataset/stats.json       — token-count summary, per-task retention %.

Both variants (T5Gemma v1 HF-seq2seq and Qwen3-4B vLLM) read the *same*
kept_ids.json so the comparison is on a shared sample set. Qwen tokenizes
those samples to a different count (vocab difference), but the semantic set
is identical.

Usage
-----
  python experiments/t5gemmav1_vs_qwen3_treb/build_dataset.py \\
         --max-tokens 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

# Reuse load_treb + build_prompt_tcot from the sibling-copied eval.py in this
# same directory. We import by path so we don't depend on package layout.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval import build_prompt_tcot, load_treb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/t5gemma-2b-2b-ul2-it",
                    help="HF model ID whose tokenizer we use to gate length.")
    ap.add_argument("--tokenizer-fallback", default="google/t5gemma-2b-2b-ul2",
                    help="Fallback tokenizer if the primary is gated and we "
                         "haven't accepted its license yet. IT has the same "
                         "tokenizer as its pretrained base.")
    ap.add_argument("--dataset", default="JT-LM/JIUTIAN-TReB")
    ap.add_argument("--language", default="English")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="Drop samples whose TCoT prompt exceeds this count.")
    ap.add_argument("--out-dir", default=str(HERE / "dataset"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (try gated IT first, fall back to public pretrained).
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok_used = args.tokenizer
    except Exception as e:
        print(f"[build_dataset] primary tokenizer {args.tokenizer} failed: "
              f"{type(e).__name__}: {str(e)[:160]}")
        print(f"[build_dataset] falling back to {args.tokenizer_fallback}")
        tok = AutoTokenizer.from_pretrained(args.tokenizer_fallback)
        tok_used = args.tokenizer_fallback
    print(f"[build_dataset] tokenizer: {tok_used}  vocab={tok.vocab_size}")

    # Load all English samples — no smoke, no stratify.
    samples = load_treb(args.dataset, args.language, smoke_n=None)
    print(f"[build_dataset] loaded {len(samples)} samples")

    # Tokenize each sample's TCoT prompt. We pass the tokenizer so
    # build_prompt_tcot does the *budgeted* assembly — but here we want the
    # natural length (no forced budget), so pass tokenizer=None for pure
    # char-assembly, then tokenize the assembled string ourselves.
    kept: list[str] = []
    dropped: list[str] = []
    token_counts: list[int] = []
    by_task_keep: defaultdict[str, int] = defaultdict(int)
    by_task_total: Counter = Counter()

    for s in samples:
        by_task_total[s["task"]] += 1
        prompt = build_prompt_tcot(s, tokenizer=None, max_input_tokens=None)
        n = len(tok.encode(prompt, add_special_tokens=False))
        token_counts.append(n)
        if n <= args.max_tokens:
            kept.append(s["id"])
            by_task_keep[s["task"]] += 1
        else:
            dropped.append(s["id"])

    token_counts.sort()
    n = len(token_counts)
    p50 = token_counts[n // 2]
    p90 = token_counts[int(0.90 * n)]
    p95 = token_counts[int(0.95 * n)]
    p99 = token_counts[int(0.99 * n)]

    retention = {
        task: {
            "kept": by_task_keep[task],
            "total": by_task_total[task],
            "retention_pct": round(100.0 * by_task_keep[task] / by_task_total[task], 1),
        }
        for task in sorted(by_task_total)
    }

    stats = {
        "tokenizer": tok_used,
        "max_tokens_cap": args.max_tokens,
        "counts": {
            "total": len(samples),
            "kept": len(kept),
            "dropped": len(dropped),
            "kept_pct": round(100.0 * len(kept) / len(samples), 1),
        },
        "prompt_tokens": {
            "min": token_counts[0],
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "max": token_counts[-1],
            "mean": round(sum(token_counts) / n, 1),
        },
        "per_task_retention": retention,
    }

    (out_dir / "kept_ids.json").write_text(json.dumps(kept, indent=2))
    (out_dir / "dropped_ids.json").write_text(json.dumps(dropped, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"[build_dataset] kept {len(kept)}/{len(samples)} "
          f"({stats['counts']['kept_pct']}%)")
    print(f"[build_dataset] prompt tokens: p50={p50} p95={p95} "
          f"max={token_counts[-1]} cap={args.max_tokens}")
    print(f"[build_dataset] wrote: {out_dir}/kept_ids.json, "
          f"dropped_ids.json, stats.json")

    # Flag tasks with >50% drop for the README reader.
    hit = [(t, r) for t, r in retention.items() if r["retention_pct"] < 50]
    if hit:
        print("[build_dataset] WARN: tasks with <50% retention:")
        for t, r in sorted(hit, key=lambda x: x[1]["retention_pct"]):
            print(f"  {t}: {r['retention_pct']}%  "
                  f"({r['kept']}/{r['total']})")


if __name__ == "__main__":
    main()
