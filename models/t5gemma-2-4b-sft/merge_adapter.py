"""Merge a trained LoRA adapter back into the base T5Gemma and push the
merged model to the Hub. Run after train.py finishes.

Usage:
    python merge_adapter.py \
        --base google/t5gemma-2-4b-4b-ul2 \
        --adapter /workspace/checkpoints/t5gemma-2-4b-flan-sft \
        --out /workspace/merged/t5gemma-2-4b-flan-sft-merged \
        --push-to DiffusionTableQA/t5gemma-2-4b-flan-sft-merged
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push-to", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="eager"
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
