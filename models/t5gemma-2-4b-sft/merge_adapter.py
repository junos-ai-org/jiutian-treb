"""Merge a trained LoRA adapter back into the base T5Gemma and push the
merged model to the Hub. Run after train.py finishes.

Usage:
    # Typical — use the base model the adapter was trained against (read
    # from adapter_config.json automatically):
    python merge_adapter.py \
        --adapter /workspace/checkpoints/t5gemma-2-4b-flan-sft \
        --out /workspace/merged/t5gemma-2-4b-flan-sft-merged \
        --push-to DiffusionTableQA/t5gemma-2-4b-flan-sft-merged

    # Override the base (rare — only if you intentionally want to apply
    # this adapter to a different base; note PeftModel does NOT validate
    # architecture compatibility, so mismatches produce junk weights):
    python merge_adapter.py --base google/t5gemma-2-4b-4b \
        --adapter ... --out ...
"""
import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def resolve_base(adapter: str) -> str:
    """Read `base_model_name_or_path` from the adapter's config. PEFT writes
    this automatically at training time. Avoids hand-edit drift (e.g. a
    README pointing at a non-existent `-ul2` suffix) silently attaching the
    adapter onto the wrong base — which PeftModel doesn't shape-check."""
    path = Path(adapter)
    if path.is_dir():
        cfg = path / "adapter_config.json"
        data = json.loads(cfg.read_text())
    else:
        local = hf_hub_download(repo_id=adapter, filename="adapter_config.json")
        data = json.loads(Path(local).read_text())
    base = data.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"adapter_config.json at {adapter} has no base_model_name_or_path; "
            "pass --base explicitly."
        )
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                    help="Base model id. Defaults to the adapter's "
                         "adapter_config.json:base_model_name_or_path.")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push-to", default=None)
    args = ap.parse_args()

    base_id = args.base or resolve_base(args.adapter)
    print(f"[merge] base={base_id}  adapter={args.adapter}")

    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_id, dtype=torch.bfloat16, attn_implementation="eager"
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"merged model saved to {args.out}")

    if args.push_to:
        merged.push_to_hub(args.push_to, token=os.environ.get("HF_TOKEN"), private=True)
        tok.push_to_hub(args.push_to, token=os.environ.get("HF_TOKEN"), private=True)
        print(f"pushed to {args.push_to}")


if __name__ == "__main__":
    main()
