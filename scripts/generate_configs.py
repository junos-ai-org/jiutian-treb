"""Generate experiment config JSONs for bidir_attn experiment.

Reads manifest.csv (produced by sample_dataset.py) to discover tasks,
then generates 4 configs: smoke/large × qwen/t5gemma.

Usage:
    python scripts/generate_configs.py \
        --data_dir experiments/bidir_attn/data \
        --config_dir experiments/bidir_attn/configs
"""

import argparse
import csv
import json
from pathlib import Path


# Task settings from TReB's config_example.json.
# NLU tasks use TCoT (no table), table tasks use TCoT_md.
# We only run TCoT_md (or TCoT for non-table tasks) in this experiment.
TASK_SETTINGS = {
    # NLU tasks — no table, use TCoT
    "Code_Generation":              {"mode": "TCoT", "if_instruction": False, "if_title": False},
    "Instruction_Following":        {"mode": "TCoT", "if_instruction": False, "if_title": False,
                                     "metrics": ["Instruction_Following"]},
    "Robustness_Evaluation":        {"mode": "TCoT", "if_instruction": False, "if_title": False},
    "Understanding":                {"mode": "TCoT", "if_instruction": False, "if_title": False},
    "Hallucination_Evaluation":     {"mode": "TCoT", "if_instruction": False, "if_title": False},
    "Mathematical_Reasoning":       {"mode": "TCoT", "if_instruction": False, "if_title": False},
    # Table tasks — use TCoT_md
    "Table_Selection":              {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_Query":                  {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_General_Operations":     {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_Domain-specific_Operations": {"mode": "TCoT_md", "if_instruction": False, "if_title": True},
    "Table_Title_Naming":           {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_Column_Naming":          {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_Summary":                {"mode": "TCoT_md", "if_instruction": False, "if_title": True},
    "Table_Fact_Checking":          {"mode": "TCoT_md", "if_instruction": False, "if_title": True},
    "Table_Plausibility_Verification": {"mode": "TCoT_md", "if_instruction": False, "if_title": True},
    "Table_Retrieval":              {"mode": "TCoT_md", "if_instruction": False, "if_title": True},
    "Table_Distribution_Testing":   {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_Hypothesis_Testing":     {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_Correlation_Analysis":   {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Table_Outlier_Detection":      {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    # ADA tasks (multi-step) — default to TCoT_md
    "Multi-step_Retrieval":         {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Multi-step_Fact_Checking":     {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Multi-step_Operations":        {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Multi-step_Correlation_Analysis": {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Multi-step_Hypothesis_Testing": {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
    "Multi-step_Conditional_Calculation": {"mode": "TCoT_md", "if_instruction": False, "if_title": False},
}

DEFAULT_METRICS = ["ROUGE", "EM", "LLM_score"]

MODEL_CONFIGS = {
    "qwen": {
        "caller_type": "vllm",
        "reason_model_name": "Qwen2.5-14B-Instruct",
        "reason_model_path": "Qwen/Qwen2.5-14B-Instruct",
        "reason_model_cards": "0,1,2,3",
        "reason_max_model_len": 8192,
        "reason_temperature": 0.0,
        "judge_caller_type": "openai",
        "judge_model_name": "gpt-5.4-mini",
        "judge_model_path": "",
        "judge_model_cards": "",
        "judge_max_model_len": 10240,
    },
    "t5gemma": {
        "caller_type": "hf_seq2seq",
        "reason_model_name": "t5gemma-9b-9b-ul2-it",
        "reason_model_path": "google/t5gemma-9b-9b-ul2-it",
        "reason_model_cards": "",
        "reason_max_model_len": 4096,
        "reason_temperature": 0.0,
        "max_new_tokens": 512,
        "hf_batch_size": 4,
        "judge_caller_type": "openai",
        "judge_model_name": "gpt-5.4-mini",
        "judge_model_path": "",
        "judge_model_cards": "",
        "judge_max_model_len": 10240,
    },
}


def read_manifest(data_dir: Path, size: str) -> list[str]:
    """Read task names from manifest.csv."""
    manifest = data_dir / size / "manifest.csv"
    tasks = []
    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row["task"])
    return tasks


def build_dataset_config(tasks: list[str], data_dir: Path, size: str) -> dict:
    """Build dataset_config section for TReB config JSON."""
    dataset_config = {}
    for task in tasks:
        settings = TASK_SETTINGS.get(task, {
            "mode": "TCoT_md", "if_instruction": False, "if_title": False,
        })
        metrics = settings.get("metrics", DEFAULT_METRICS)
        dataset_config[task] = {
            "path": f"../{data_dir}/{size}/{task}.json",
            "reason_mode": [settings["mode"]],
            "if_instruction": settings["if_instruction"],
            "if_title": settings["if_title"],
            "metrics": metrics,
        }
    return dataset_config


def generate_config(model: str, size: str, tasks: list[str],
                    data_dir: Path) -> dict:
    """Generate a complete TReB config JSON."""
    return {
        "version": f"bidir_attn_{size}",
        "round": 1,
        "model_config": MODEL_CONFIGS[model],
        "dataset_config": build_dataset_config(tasks, data_dir, size),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate TReB experiment configs")
    parser.add_argument("--data_dir", type=str,
                        default="experiments/bidir_attn/data")
    parser.add_argument("--config_dir", type=str,
                        default="experiments/bidir_attn/configs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    for size in ["smoke", "large"]:
        manifest = data_dir / size / "manifest.csv"
        if not manifest.exists():
            print(f"Skipping {size} — {manifest} not found. Run sample_dataset.py first.")
            continue

        tasks = read_manifest(data_dir, size)
        print(f"\n{size}: {len(tasks)} tasks")

        for model in ["qwen", "t5gemma"]:
            config = generate_config(model, size, tasks, data_dir)
            path = config_dir / f"config_{model}_{size}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"  Written: {path}")


if __name__ == "__main__":
    main()
