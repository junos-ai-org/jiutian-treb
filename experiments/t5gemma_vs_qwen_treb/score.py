#!/usr/bin/env python3
"""Score predictions.jsonl → metrics.json.

Metrics:
  - Exact Match (text, case-insensitive, whitespace-normalized)
  - Numeric Match (when `number_answer` is present)
  - ROUGE-L (if `rouge-score` is installed)
  - DeepSeek V3 LLM-as-Judge via OpenRouter (if --judge deepseek-v3).
    Returns score (0-10), a structured failure-mode category, and rationale.

Optional --wandb logs scalar metrics + a wandb.Table of per-sample rows so
we can filter/sort/cross-compare variants in the wandb UI.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---- Local metrics ------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def numeric_match(pred: str, gold_num: float, rel_tol: float = 1e-3) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", pred or "")
    if not m:
        return 0.0
    try:
        p = float(m.group())
    except ValueError:
        return 0.0
    if abs(p - gold_num) < 1e-6:
        return 1.0
    denom = max(abs(gold_num), 1e-9)
    return 1.0 if abs(p - gold_num) / denom < rel_tol else 0.0


_ROUGE_SCORER = None

def rouge_l(pred: str, gold: str) -> float:
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        try:
            from rouge_score import rouge_scorer
            _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            _ROUGE_SCORER = False
    if _ROUGE_SCORER is False:
        return -1.0
    return _ROUGE_SCORER.score(gold or "", pred or "")["rougeL"].fmeasure


def extract_json_answer(text: str) -> str:
    """Model outputs are asked for {"answer": ...} on the last line — extract if possible,
    else return the whole text as the prediction."""
    if not text:
        return ""
    m = re.search(r'\{[^{}]*"answer"\s*:\s*([^{}]+?)\}', text, re.DOTALL)
    if not m:
        return text.strip()
    try:
        obj = json.loads(m.group(0))
        v = obj.get("answer", "")
        return str(v) if not isinstance(v, str) else v
    except Exception:
        return m.group(1).strip().strip('"').strip(",").strip()


# ---- OpenRouter DeepSeek V3 judge ---------------------------------------

FAILURE_MODES = [
    "CORRECT",          # fully right
    "PARTIAL",          # right direction, wrong detail
    "FORMAT",           # output not parseable / wrong structure
    "HALLUCINATION",    # fact not supported by the table
    "WRONG_SELECTION",  # picked wrong row / column / cell
    "NUMERIC",          # arithmetic or numeric off
    "REASONING",        # correct steps, wrong final answer
    "REFUSAL",          # model declined to answer
    "EMPTY",            # no output / only whitespace
    "MISUNDERSTAND",    # answered a different question
    "TRUNCATED",        # output cuts off mid-thought (hit max_new_tokens)
    "OTHER",            # doesn't fit the above
]


def _judge_prompt(s: dict) -> str:
    modes = "\n".join(f"  - {m}" for m in FAILURE_MODES)
    return (
        "You are evaluating a language model's answer to a table-reasoning question.\n\n"
        f"Question: {s['question']}\n"
        f"Gold answer: {s['gold_answer']}\n"
        f"Model answer: {s['prediction']}\n\n"
        "Score 0-10:\n"
        "  0  wrong / unrelated / no answer\n"
        "  5  partially correct (right direction, wrong detail)\n"
        "  10 fully correct (semantically equivalent to gold)\n\n"
        "Classify the model's output into exactly one category:\n"
        f"{modes}\n\n"
        'Reply with ONLY a JSON object:\n'
        '{"score": <int 0-10>, "mode": "<one of the categories above>", '
        '"rationale": "<one sentence>"}'
    )


def deepseek_judge(
    samples: list[dict],
    model: str = "deepseek/deepseek-chat-v3",
    max_workers: int = 8,
) -> dict[str, dict]:
    import httpx
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set; source ~/.claude/.env")

    def judge_one(s: dict) -> tuple[str, dict]:
        prompt = _judge_prompt(s)
        with httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/junos-ai-org/jiutian-treb",
                "X-Title": "treb-eval",
            },
            timeout=60.0,
        ) as client:
            r = client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 250,
                },
            )
        score = -1.0
        mode = "OTHER"
        rationale = ""
        raw = ""
        try:
            data = r.json()
            raw = data["choices"][0]["message"]["content"]
            # Be lenient — the judge might wrap JSON in prose or code fences.
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    score = float(obj.get("score", -1))
                    mode_candidate = str(obj.get("mode", "OTHER")).upper().strip()
                    mode = mode_candidate if mode_candidate in FAILURE_MODES else "OTHER"
                    rationale = str(obj.get("rationale", ""))
                except json.JSONDecodeError:
                    # Fall back to regex
                    sm = re.search(r'"score"\s*:\s*(\d+)', raw)
                    mm = re.search(r'"mode"\s*:\s*"([A-Z_]+)"', raw)
                    rm = re.search(r'"rationale"\s*:\s*"([^"]*)"', raw)
                    if sm:
                        score = float(sm.group(1))
                    if mm and mm.group(1) in FAILURE_MODES:
                        mode = mm.group(1)
                    if rm:
                        rationale = rm.group(1)
        except Exception as e:
            rationale = f"ERR: {e}"
        return s["id"], {
            "score": score,
            "mode": mode,
            "rationale": rationale,
            "raw": raw,
        }

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge_one, s): s["id"] for s in samples}
        for i, fut in enumerate(as_completed(futures)):
            sid, j = fut.result()
            results[sid] = j
            if i % 50 == 0:
                print(f"[judge] {i}/{len(samples)} done", flush=True)
    return results


# ---- Aggregation --------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def _mode_distribution(items: list[dict]) -> dict[str, float]:
    total = sum(1 for s in items if s.get("_judge_mode"))
    if not total:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for s in items:
        m = s.get("_judge_mode")
        if m:
            counts[m] += 1
    return {m: round(c / total, 4) for m, c in sorted(counts.items())}


def aggregate(samples: list[dict], use_judge: bool) -> dict[str, Any]:
    by_task: dict[str, list] = defaultdict(list)
    for s in samples:
        by_task[s["task"]].append(s)

    def task_metrics(items: list[dict]) -> dict:
        out = {
            "n": len(items),
            "em": _mean([s["_em"] for s in items]),
            "rouge_l": _mean([s["_rouge"] for s in items if s["_rouge"] >= 0]),
        }
        numeric_scored = [s["_numeric"] for s in items if "_numeric" in s]
        if numeric_scored:
            out["numeric_match"] = _mean(numeric_scored)
        if use_judge:
            out["judge_mean"] = _mean([s["_judge_score"] for s in items if s["_judge_score"] >= 0])
            mode_dist = _mode_distribution(items)
            if mode_dist:
                out["failure_modes"] = mode_dist
        return out

    return {
        "overall": task_metrics(samples),
        "per_task": {task: task_metrics(items) for task, items in sorted(by_task.items())},
    }


# ---- Main ---------------------------------------------------------------

def log_to_wandb(
    variant: str,
    samples: list[dict],
    metrics: dict,
    use_judge: bool,
    project: str,
    run_name: str | None,
) -> None:
    """Log scalar metrics + a per-sample predictions table to wandb."""
    import wandb
    run = wandb.init(
        project=project,
        name=run_name or variant,
        config={"variant": variant, "judge": use_judge, "n": len(samples)},
        reinit=True,
    )
    # Scalars: overall + per-task (flattened)
    flat: dict[str, float] = {}
    for k, v in metrics["overall"].items():
        if isinstance(v, (int, float)):
            flat[f"overall/{k}"] = v
        elif isinstance(v, dict):  # failure_modes
            for sub, val in v.items():
                flat[f"overall/failure_modes/{sub}"] = val
    for task, tm in metrics.get("per_task", {}).items():
        for k, v in tm.items():
            if isinstance(v, (int, float)):
                flat[f"per_task/{task}/{k}"] = v
    run.log(flat)

    # Per-sample table for filtering / cross-variant comparison
    cols = [
        "id", "task", "variant", "question", "gold_answer", "prediction",
        "pred_extracted", "em", "rouge_l",
    ]
    if any("_numeric" in s for s in samples):
        cols.append("numeric_match")
    if use_judge:
        cols.extend(["judge_score", "judge_mode", "judge_rationale"])
    table = wandb.Table(columns=cols)
    for s in samples:
        row = [
            s.get("id", ""), s.get("task", ""), s.get("variant", variant),
            s.get("question", ""), s.get("gold_answer", ""), s.get("prediction", ""),
            s.get("_pred_extracted", ""),
            s.get("_em", -1.0), s.get("_rouge", -1.0),
        ]
        if "numeric_match" in cols:
            row.append(s.get("_numeric", -1.0))
        if use_judge:
            row.extend([
                s.get("_judge_score", -1.0),
                s.get("_judge_mode", ""),
                s.get("_judge_rationale", ""),
            ])
        table.add_data(*row)
    run.log({"predictions": table})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--judge", choices=["none", "deepseek-v3"], default="none")
    parser.add_argument("--output", default=None)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--wandb", action="store_true",
                        help="Log scalar metrics + per-sample table to wandb")
    parser.add_argument("--wandb-project", default="treb-encoder-vs-decoder-eval")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Defaults to the variant name from predictions.")
    args = parser.parse_args()

    preds_path = Path(args.predictions)
    samples = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]
    out_path = Path(args.output) if args.output else preds_path.parent / "metrics.json"
    variant = samples[0].get("variant", "unknown") if samples else "unknown"
    print(f"[score] {len(samples)} predictions from {preds_path} (variant={variant})")

    for s in samples:
        extracted = extract_json_answer(s.get("prediction", ""))
        s["_pred_extracted"] = extracted
        s["_em"] = exact_match(extracted, s.get("gold_answer", ""))
        s["_rouge"] = rouge_l(extracted, s.get("gold_answer", ""))
        if s.get("number_answer") is not None:
            s["_numeric"] = numeric_match(extracted, float(s["number_answer"]))

    use_judge = args.judge == "deepseek-v3"
    if use_judge:
        print(f"[score] running DeepSeek V3 judge on {len(samples)} samples via OpenRouter")
        judgments = deepseek_judge(samples, max_workers=args.judge_workers)
        for s in samples:
            j = judgments.get(s["id"], {})
            s["_judge_score"] = j.get("score", -1.0)
            s["_judge_mode"] = j.get("mode", "")
            s["_judge_rationale"] = j.get("rationale", "")

    metrics = aggregate(samples, use_judge=use_judge)
    metrics["variant"] = variant
    metrics["judge"] = args.judge

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[score] wrote {out_path}")
    print(json.dumps(metrics["overall"], indent=2))

    # Also dump a scored-samples jsonl for error analysis
    scored = preds_path.with_name(preds_path.stem + ".scored.jsonl")
    with scored.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[score] wrote per-sample scores to {scored}")

    if args.wandb:
        print(f"[score] logging to wandb project={args.wandb_project}")
        log_to_wandb(
            variant=variant, samples=samples, metrics=metrics, use_judge=use_judge,
            project=args.wandb_project, run_name=args.wandb_run_name,
        )


if __name__ == "__main__":
    main()
