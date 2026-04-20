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
        stripped = text.strip()
        if not stripped:
            continue
        # Command detection is strip-aware (paste-friendly).
        if stripped.startswith("/sample"):
            parts = stripped.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                print("[play] usage: /sample <full_id_with_task_prefix>  (e.g. Table_Query|2646e8725e97437)")
                continue
            sid = parts[1].strip()
            samples = load_treb(config["eval"]["dataset"], config["eval"]["language"])
            by_id = {s["id"]: s for s in samples}
            # Fuzzy: if exact id not found, try suffix match on the hash portion
            if sid not in by_id:
                matches = [i for i in by_id if i.endswith("|" + sid) or i == sid or sid in i]
                if len(matches) == 1:
                    sid = matches[0]
                    print(f"[play] matched: {sid}")
                elif len(matches) > 1:
                    print(f"[play] '{sid}' ambiguous — {len(matches)} matches, first few:")
                    for m in matches[:5]:
                        print(f"       {m}")
                    continue
                else:
                    print(f"[play] sample {sid!r} not found. Use full id like 'Task_Name|hash'.")
                    continue
            s = by_id[sid]
            prompt = build_prompt_tcot(s, tokenizer=tok, max_input_tokens=config["eval"].get("max_input_tokens"))
            print(f"--- prompt (truncated for display) ---\n{prompt[:600]}{'...' if len(prompt)>600 else ''}\n---")
            print(f"GOLD: {s.get('gold_answer', '')[:200]}")
            print(f"\n=== generating ===")
        elif stripped.startswith("/list"):
            samples = load_treb(config["eval"]["dataset"], config["eval"]["language"])
            by_task: dict[str, list] = {}
            for s in samples:
                by_task.setdefault(s["task"], []).append(s["id"])
            print(f"[play] {len(by_task)} tasks:")
            for task, ids in sorted(by_task.items()):
                print(f"  {task} ({len(ids)} samples)  e.g. {ids[0]}")
            continue
        elif stripped.startswith("/help"):
            print("[play] commands:")
            print("  /sample <id>   load a TReB sample (full id, or suffix match)")
            print("  /list          list all tasks with one example id each")
            print("  /debug [prompt] show tokenizer + EOS inspection; generate w/ raw-ids")
            print("  <free text>    raw prompt to the model")
            print("  end input with /// on its own line")
            continue
        elif stripped.startswith("/debug"):
            parts = stripped.split(None, 1)
            probe = parts[1] if len(parts) > 1 else "Q: What is 2+2? A:"
            print(f"\n--- tokenizer special tokens ---")
            try:
                print(f"  eos_token: {getattr(tok, 'eos_token', '?')}  id: {getattr(tok, 'eos_token_id', '?')}")
                print(f"  bos_token: {getattr(tok, 'bos_token', '?')}  id: {getattr(tok, 'bos_token_id', '?')}")
                print(f"  pad_token: {getattr(tok, 'pad_token', '?')}  id: {getattr(tok, 'pad_token_id', '?')}")
                print(f"  special_tokens_map: {getattr(tok, 'special_tokens_map', {})}")
                added = getattr(tok, "added_tokens_decoder", {}) or {}
                print(f"  added_tokens (first 15):")
                for i, (tid, tok_obj) in enumerate(list(added.items())[:15]):
                    print(f"    id={tid}  content={getattr(tok_obj, 'content', tok_obj)!r}  special={getattr(tok_obj, 'special', '?')}")
            except Exception as e:
                print(f"  err: {e}")

            print(f"\n--- model.config stop-related ---")
            try:
                cfg = getattr(model, "config", None)
                if cfg is not None:
                    print(f"  eos_token_id: {cfg.eos_token_id}")
                    gc = getattr(cfg, "generation_config", None)
                    dec = getattr(cfg, "decoder_start_token_id", None)
                    print(f"  decoder_start_token_id: {dec}")
                    if hasattr(cfg, "get_text_config"):
                        try:
                            tc = cfg.get_text_config()
                            print(f"  text_config.eos_token_id: {getattr(tc, 'eos_token_id', None)}")
                        except Exception:
                            pass
                # GenerationConfig defaults
                try:
                    from transformers import GenerationConfig
                    gc = GenerationConfig.from_model_config(cfg) if cfg else None
                    if gc:
                        print(f"  GenerationConfig.eos_token_id: {gc.eos_token_id}")
                        print(f"  GenerationConfig.pad_token_id: {gc.pad_token_id}")
                except Exception as e:
                    print(f"  GenerationConfig probe err: {e}")
            except Exception as e:
                print(f"  err: {e}")

            if config["model"]["backend"] == "vllm":
                print("\n[debug] skipping generation-time EOS inspection for vllm backend")
                continue
            print(f"\n--- generating 100 tokens on probe: {probe!r} ---")
            import torch
            enc = tok(probe, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=100, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=False)
            seqs = out.sequences if hasattr(out, "sequences") else out
            raw_ids = seqs[0].tolist()
            print(f"  raw output length: {len(raw_ids)}  (input was {enc['input_ids'].shape[-1]})")
            eos_id = getattr(model.config, "eos_token_id", None)
            if isinstance(eos_id, list):
                eos_ids = set(eos_id)
            else:
                eos_ids = {eos_id}
            eos_positions = [i for i, t in enumerate(raw_ids) if t in eos_ids]
            print(f"  EOS ids {eos_ids} appear at positions: {eos_positions[:10]}{'...' if len(eos_positions)>10 else ''}  (count={len(eos_positions)})")
            print(f"  last 10 token ids: {raw_ids[-10:]}")
            print(f"  decoded (skip_special=False):")
            print(f"    {tok.decode(raw_ids, skip_special_tokens=False)[-300:]!r}")
            print(f"  decoded (skip_special=True, last 200 chars):")
            print(f"    {tok.decode(raw_ids, skip_special_tokens=True)[-200:]!r}")
            continue
        else:
            prompt = text  # preserve original formatting (newlines etc) for raw prompts
            print(f"\n=== generating ===")
        print(gen(prompt))
        print("=== end ===\n")


if __name__ == "__main__":
    main()
