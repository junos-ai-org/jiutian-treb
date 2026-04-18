"""One-shot FLAN prep: sample, tokenize, save to disk on the network volume.

Run once per dataset config; subsequent train.py runs load from tokenized_dir
and skip tokenization. Safe to re-run — overwrites the output dir.

Usage:
    python prepare_data.py --config configs/sft_flan.yaml
"""
import argparse
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    d, m = cfg["data"], cfg["model"]

    tok = AutoTokenizer.from_pretrained(m["name_or_path"])

    # Stream the dataset and take only what we need. Non-streaming mode on
    # Open-Orca/FLAN pulls all 2157 parquet shards (hundreds of GB, ~2 hours),
    # which is wasteful when we only want 100K rows.
    total = d["train_size"] + d["eval_size"]
    stream = load_dataset(
        d["dataset_name"], split="train", streaming=True, cache_dir=d["cache_dir"]
    )
    # Light shuffle within a buffer to de-bias from shard order.
    stream = stream.shuffle(seed=d["seed"], buffer_size=10_000)
    samples = list(stream.take(total))
    ds = Dataset.from_list(samples)
    split = ds.train_test_split(test_size=d["eval_size"], seed=d["seed"])

    def preprocess(batch):
        enc = tok(
            batch[d["input_field"]],
            max_length=d["max_input_length"],
            truncation=True,
            padding=False,
        )
        lab = tok(
            batch[d["target_field"]],
            max_length=d["max_target_length"],
            truncation=True,
            padding=False,
        )
        enc["labels"] = lab["input_ids"]
        return enc

    keep = list(split["train"].column_names)
    tokenized = split.map(preprocess, batched=True, remove_columns=keep)
    tokenized.save_to_disk(d["tokenized_dir"])
    print(f"saved tokenized splits to {d['tokenized_dir']}")
    print({k: len(v) for k, v in tokenized.items()})


if __name__ == "__main__":
    main()
