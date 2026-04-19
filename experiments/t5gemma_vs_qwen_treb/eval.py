#!/usr/bin/env python3
"""Eval driver for TReB — loads a config, runs one variant, writes predictions.jsonl.

Backends:
  - vllm            : vLLM.LLM (decoder-only, e.g. Qwen)
  - hf_seq2seq      : HF AutoModelForSeq2SeqLM (T5Gemma base)
  - hf_seq2seq_peft : + PeftModel wrapper (T5Gemma SFT adapter)

Data: TReB English JSONs from JT-LM/JIUTIAN-TReB. `Table_markdown` field is
already prepared, so we don't need CSV/CSV.zip for TCoT mode.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download


# ---- Config & data -------------------------------------------------------

def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_treb(dataset_id: str, language: str, smoke_n: int | None = None) -> list[dict]:
    """Download language-specific JSONs + return flat sample list."""
    local = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=f"{language}/*.json",
    )
    samples: list[dict] = []
    for p in sorted(Path(local).glob(f"{language}/*.json")):
        task = p.stem
        for item in json.loads(p.read_text()):
            title = item.get("title") or [""]
            md = item.get("Table_markdown") or [""]
            samples.append({
                "id": item["id"],
                "task": task,
                "instruction": item.get("instruction", ""),
                "question": item["question"],
                "gold_answer": item.get("answer", ""),
                "number_answer": item.get("number_answer"),
                "title": title[0] if isinstance(title, list) and title else str(title or ""),
                "table_markdown": md[0] if isinstance(md, list) and md else str(md or ""),
            })
    if smoke_n:
        by_task: dict[str, list] = defaultdict(list)
        for s in samples:
            by_task[s["task"]].append(s)
        rng = random.Random(42)
        per_task = max(1, smoke_n // len(by_task))
        out: list[dict] = []
        for items in by_task.values():
            rng.shuffle(items)
            out.extend(items[:per_task])
        rng.shuffle(out)
        return out[:smoke_n]
    return samples


MAX_TABLE_CHARS = 30000  # Conservative — some dense TReB tables tokenize at
                         # nearly 1 token per char (wall-of-pipes markdown), so
                         # 40K chars still produced >32K token prompts. 30K chars
                         # gives a safe upper bound well inside Qwen's 32K window
                         # (with 2048 reserved for generation).


def build_prompt_tcot(sample: dict) -> str:
    table_md = sample["table_markdown"]
    if len(table_md) > MAX_TABLE_CHARS:
        table_md = table_md[:MAX_TABLE_CHARS] + "\n... [table truncated for length] ..."
    parts = [sample["instruction"]]
    if sample["title"]:
        parts.append(sample["title"])
    if table_md:
        parts.append(table_md)
    parts.append(f"Question: {sample['question']}")
    parts.append(
        'Reason step by step. Then output ONLY a single JSON object on the last line: {"answer": <value>}'
    )
    return "\n\n".join(parts)


# ---- Backends ------------------------------------------------------------

def run_vllm(config: dict, samples: list[dict]) -> list[str]:
    from vllm import LLM, SamplingParams
    m, g = config["model"], config["generation"]
    llm = LLM(
        model=m["name_or_path"],
        revision=m.get("revision") or None,
        dtype=m.get("dtype", "bfloat16"),
        gpu_memory_utilization=m.get("gpu_memory_utilization", 0.90),
        max_model_len=m.get("max_model_len", 8192),
        tensor_parallel_size=m.get("tensor_parallel_size", 1),
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=g.get("temperature", 0.0),
        top_p=g.get("top_p", 1.0),
        max_tokens=g.get("max_new_tokens", 512),
    )
    prompts = []
    for s in samples:
        msgs = [{"role": "user", "content": build_prompt_tcot(s)}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    outs = llm.generate(prompts, sp)
    # vLLM returns outputs in same order as prompts
    return [o.outputs[0].text for o in outs]


def run_hf_seq2seq(config: dict, samples: list[dict], use_peft: bool = False) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    m, g = config["model"], config["generation"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[m.get("dtype", "bfloat16")]

    tokenizer = AutoTokenizer.from_pretrained(m["name_or_path"], revision=m.get("revision") or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        m["name_or_path"],
        revision=m.get("revision") or None,
        dtype=dtype,
        attn_implementation=m.get("attn_implementation", "eager"),
        device_map=m.get("device_map", "auto"),
    )
    if use_peft:
        from peft import PeftModel
        a = m["adapter"]
        model = PeftModel.from_pretrained(model, a["repo_id"], revision=a.get("revision") or None)
        # merge_and_unload folds LoRA deltas into base weights and returns a
        # plain AutoModelForSeq2SeqLM. Without this, PEFT wraps every linear
        # layer in Python at inference — ~2-3× slowdown vs the base model.
        model = model.merge_and_unload()
        print(f"[eval] PEFT adapter merged into base weights (merge_and_unload)")
    model.eval()

    batch_size = config["eval"].get("batch_size", 8)
    max_input_tokens = config["eval"].get("max_input_tokens", 32768)
    gen_kwargs = dict(
        max_new_tokens=g.get("max_new_tokens", 512),
        do_sample=g.get("do_sample", False),
        num_beams=g.get("num_beams", 1),
    )

    # Length-sorted batching: tokenize all prompts up-front, sort by length
    # ascending, process in batches of similar length, then unsort predictions
    # to original order. Removes most of the padding waste that comes from
    # TReB's wild length variance (200-char tables next to 80K-char tables).
    texts_all = [build_prompt_tcot(s) for s in samples]
    lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_all]
    sort_idx = sorted(range(len(samples)), key=lambda i: lens[i])
    sorted_texts = [texts_all[i] for i in sort_idx]
    print(f"[eval] lengths: min={min(lens)}  max={max(lens)}  median={sorted(lens)[len(lens)//2]}  "
          f"batch_size={batch_size}  max_input={max_input_tokens}", flush=True)

    sorted_outs: list[str] = [""] * len(samples)
    for i in range(0, len(samples), batch_size):
        batch_texts = sorted_texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=max_input_tokens,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, d in enumerate(decoded):
            sorted_outs[i + j] = d
        if (i // batch_size) % 10 == 0:
            print(f"[eval]   progress {i + len(batch_texts)}/{len(samples)}", flush=True)

    # Unsort: place each sorted output back at its original index
    outs: list[str] = [""] * len(samples)
    for sort_pos, orig_idx in enumerate(sort_idx):
        outs[orig_idx] = sorted_outs[sort_pos]
    return outs


# ---- Main ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    smoke_n = config["eval"].get("smoke_size", 100) if args.smoke else None

    print(f"[eval] variant={config['variant']} backend={config['model']['backend']} smoke={args.smoke}")
    samples = load_treb(
        config["eval"]["dataset"], config["eval"]["language"], smoke_n
    )
    n_tasks = len({s["task"] for s in samples})
    print(f"[eval] loaded {len(samples)} samples across {n_tasks} tasks")

    backend = config["model"]["backend"]
    if backend == "vllm":
        preds = run_vllm(config, samples)
    elif backend == "hf_seq2seq":
        preds = run_hf_seq2seq(config, samples, use_peft=False)
    elif backend == "hf_seq2seq_peft":
        preds = run_hf_seq2seq(config, samples, use_peft=True)
    else:
        raise ValueError(f"unknown backend: {backend}")

    out_path = Path(config["output"]["predictions_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s, p in zip(samples, preds):
            rec: dict[str, Any] = {**s, "prediction": p, "variant": config["variant"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
