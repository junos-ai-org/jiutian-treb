#!/usr/bin/env python3
"""Interactive REPL for a configured model. Loads the model once, drops into
a prompt loop where you can paste a question (or an instruction + table +
question) and see the generation.

Usage (on pod, after SSH):
    python play.py --config configs/t5gemma_base.yaml
    python play.py --config configs/t5gemma_sft.yaml
    python play.py --config configs/qwen_7b_instruct.yaml --backend vllm

Tip: multi-line input ends with a line containing only `///`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import yaml

from eval import build_prompt_tcot, load_treb


def load_hf(config):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    m = config["model"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[m.get("dtype", "bfloat16")]
    tok = AutoTokenizer.from_pretrained(m["name_or_path"], revision=m.get("revision") or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        m["name_or_path"], revision=m.get("revision") or None,
        dtype=dtype,
        attn_implementation=m.get("attn_implementation", "eager"),
        device_map=m.get("device_map", "auto"),
    )
    if m.get("adapter"):
        from peft import PeftModel
        a = m["adapter"]
        model = PeftModel.from_pretrained(model, a["repo_id"], revision=a.get("revision") or None)
        model = model.merge_and_unload()
        print(f"[play] adapter merged: {a['repo_id']}@{a.get('revision')}")
    model.eval()
    return model, tok, "hf"


def load_vllm(config):
    from vllm import LLM
    m = config["model"]
    llm = LLM(
        model=m["name_or_path"],
        revision=m.get("revision") or None,
        dtype=m.get("dtype", "bfloat16"),
        gpu_memory_utilization=m.get("gpu_memory_utilization", 0.90),
        max_model_len=m.get("max_model_len", 8192),
        tensor_parallel_size=m.get("tensor_parallel_size", 1),
        trust_remote_code=True,
    )
    return llm, llm.get_tokenizer(), "vllm"


def generate_hf(model, tok, prompt, max_new_tokens=1024):
    import torch
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4000).to(model.device)
    print(f"[play] input tokens: {enc['input_ids'].shape[-1]}")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def generate_vllm(llm, tok, prompt, max_new_tokens=1024):
    from vllm import SamplingParams
    msgs = [{"role": "user", "content": prompt}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    outs = llm.generate([full], sp)
    return outs[0].outputs[0].text


def read_multiline(prompt: str = "prompt> ") -> str:
    print(prompt + "  (end with a line containing only `///`)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "///":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--sample-id", default=None,
                    help="Load this TReB sample's full TCoT prompt to start")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    print(f"[play] variant={config['variant']} backend={config['model']['backend']}")
    print(f"[play] loading model...")
    if config["model"]["backend"] == "vllm":
        model, tok, kind = load_vllm(config)
        gen = lambda p: generate_vllm(model, tok, p, args.max_new_tokens)
    else:
        use_peft = config["model"]["backend"] == "hf_seq2seq_peft"
        model, tok, kind = load_hf(config)
        gen = lambda p: generate_hf(model, tok, p, args.max_new_tokens)
    print(f"[play] loaded. Commands: `/sample <id>` to load a TReB sample, Ctrl-D to quit.")
    print()

    if args.sample_id:
        samples = load_treb(config["eval"]["dataset"], config["eval"]["language"])
        by_id = {s["id"]: s for s in samples}
        if args.sample_id in by_id:
            s = by_id[args.sample_id]
            prompt = build_prompt_tcot(s, tokenizer=tok, max_input_tokens=config["eval"].get("max_input_tokens"))
            print(f"--- prompt ---\n{prompt[:800]}{'...' if len(prompt)>800 else ''}\n---")
            print(f"GOLD: {s.get('gold_answer', '')[:200]}")
            print(f"\n=== generating ===")
            print(gen(prompt))
            print("=== end ===\n")

    while True:
        try:
            text = read_multiline()
        except (KeyboardInterrupt, EOFError):
            print("\n[play] bye")
            return
        if not text.strip():
            continue
        if text.startswith("/sample "):
            sid = text.split(None, 1)[1].strip()
            samples = load_treb(config["eval"]["dataset"], config["eval"]["language"])
            by_id = {s["id"]: s for s in samples}
            if sid not in by_id:
                print(f"[play] sample {sid} not found")
                continue
            s = by_id[sid]
            prompt = build_prompt_tcot(s, tokenizer=tok, max_input_tokens=config["eval"].get("max_input_tokens"))
            print(f"--- prompt (truncated for display) ---\n{prompt[:600]}{'...' if len(prompt)>600 else ''}\n---")
            print(f"GOLD: {s.get('gold_answer', '')[:200]}")
            print(f"\n=== generating ===")
        else:
            prompt = text
            print(f"\n=== generating ===")
        print(gen(prompt))
        print("=== end ===\n")


if __name__ == "__main__":
    main()
