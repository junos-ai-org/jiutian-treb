"""Cross-benchmark comparison: MMTU + TReB.

Reads analysis JSON from both benchmarks and produces a unified
comparison table mapping tasks to common categories.

Usage:
    python scripts/cross_benchmark.py \
        --mmtu_qwen ~/research/MMTU/projects/tabular-llms-research/experiments/encoder_vs_decoder_baseline/output/Qwen2.5-14B-Instruct/latest/analysis.json \
        --mmtu_t5gemma ~/research/MMTU/projects/tabular-llms-research/experiments/encoder_vs_decoder_baseline/output/t5gemma-9b-9b-ul2-it/latest/analysis.json \
        --treb_qwen results/qwen_summary.json \
        --treb_t5gemma results/t5gemma_summary.json
"""

import argparse
import json


# Map tasks from both benchmarks to common categories
CATEGORY_MAP = {
    # MMTU tasks
    "Table-QA":                 ("MMTU", "Table QA"),
    "Table-Fact-Verification":  ("MMTU", "Fact Verification"),
    "Entity-Matching":          ("MMTU", "Entity/Schema"),
    "Schema-Matching":          ("MMTU", "Entity/Schema"),
    "Data-Imputation":          ("MMTU", "Data Imputation"),
    "List-to-table":            ("MMTU", "Structure Extraction"),
    "Cell-entity-annotation":   ("MMTU", "Annotation"),
    "Column-type-annotation":   ("MMTU", "Annotation"),
    "semantic-transform":       ("MMTU", "Transformation"),
    "semantic-join":            ("MMTU", "Join/Dependency"),
    "Functional-Dependency":    ("MMTU", "Join/Dependency"),
    "equi-join-detect":         ("MMTU", "Join/Dependency"),
    "header-value-matching":    ("MMTU", "Matching"),
    # TReB tasks
    "Table_Query":              ("TReB", "Table QA"),
    "Table_Selection":          ("TReB", "Table QA"),
    "Table_Fact_Checking":      ("TReB", "Fact Verification"),
    "Table_Retrieval":          ("TReB", "Table QA"),
    "Table_Summary":            ("TReB", "Summarization"),
    "Table_Column_Naming":      ("TReB", "Annotation"),
    "Table_Title_Naming":       ("TReB", "Annotation"),
    "Table_General_Operations": ("TReB", "Operations"),
    "Table_Domain-specific_Operations": ("TReB", "Operations"),
    "Table_Distribution_Testing": ("TReB", "Data Analysis"),
    "Table_Hypothesis_Testing": ("TReB", "Data Analysis"),
    "Table_Correlation_Analysis": ("TReB", "Data Analysis"),
    "Table_Outlier_Detection":  ("TReB", "Data Analysis"),
}


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_scores(data: dict, metric: str) -> dict[str, float]:
    """Extract task->score from analysis JSON. Handles both MMTU and TReB formats."""
    tasks = data.get("tasks", data)
    scores = {}
    for task, info in tasks.items():
        if isinstance(info, dict):
            score = info.get(metric) or info.get("score")
        else:
            score = info
        if score is not None:
            scores[task] = score
    return scores


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark comparison")
    parser.add_argument("--mmtu_qwen", required=True)
    parser.add_argument("--mmtu_t5gemma", required=True)
    parser.add_argument("--treb_qwen", required=True)
    parser.add_argument("--treb_t5gemma", required=True)
    parser.add_argument("--metric", default="EM",
                        help="TReB metric to compare (default: EM)")
    args = parser.parse_args()

    mmtu_q = extract_scores(load_json(args.mmtu_qwen), "score")
    mmtu_t = extract_scores(load_json(args.mmtu_t5gemma), "score")
    treb_q = extract_scores(load_json(args.treb_qwen), args.metric)
    treb_t = extract_scores(load_json(args.treb_t5gemma), args.metric)

    print("# Cross-Benchmark Comparison: MMTU + TReB")
    print(f"\nTReB metric: {args.metric} | MMTU metric: accuracy/F1\n")
    print("| Benchmark | Category | Task | Qwen2.5-14B | T5Gemma-9B | Delta |")
    print("|---|---|---|---|---|---|")

    all_scores = {**mmtu_q, **treb_q}
    for task in sorted(all_scores.keys(), key=lambda t: CATEGORY_MAP.get(t, ("ZZ", "ZZ"))):
        bench, cat = CATEGORY_MAP.get(task, ("?", "?"))
        q = mmtu_q.get(task) if bench == "MMTU" else treb_q.get(task)
        t = mmtu_t.get(task) if bench == "MMTU" else treb_t.get(task)

        if q is not None and t is not None:
            delta = round(q - t, 4)
            sign = "+" if delta > 0 else ""
            print(f"| {bench} | {cat} | {task} | {q:.3f} | {t:.3f} | {sign}{delta:.3f} |")
        else:
            q_str = f"{q:.3f}" if q is not None else "—"
            t_str = f"{t:.3f}" if t is not None else "—"
            print(f"| {bench} | {cat} | {task} | {q_str} | {t_str} | — |")


if __name__ == "__main__":
    main()
