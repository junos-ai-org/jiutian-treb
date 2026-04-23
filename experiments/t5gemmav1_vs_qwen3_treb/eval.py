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


def load_treb(
    dataset_id: str,
    language: str,
    smoke_n: int | None = None,
    stratify_by: str = "length",
) -> list[dict]:
    """Download language-specific JSONs + return flat sample list.

    When `smoke_n` is set, stratify-sample down to that count. Strategies:
      - "length" (default): 4 equal-width bins by prompt-char length → smoke_n/4 from each.
        Use this when we care about *coverage across input size* (e.g. diagnosing
        whether long inputs trigger implementation bugs).
      - "task": ~smoke_n/num_tasks samples per task. Use when we care about
        coverage across the 26 TReB subtasks equally.
    """
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
    if not smoke_n:
        return samples

    rng = random.Random(42)
    if stratify_by == "length":
        # char-count proxy for token-count. Sort + slice into 4 quartile bins;
        # take smoke_n/4 from each bin → guaranteed coverage of short/med/long/xlong.
        def _char_len(s: dict) -> int:
            return (len(s.get("instruction", "")) + len(s.get("question", ""))
                    + len(s.get("table_markdown", "")))
        ordered = sorted(samples, key=_char_len)
        n_bins = 4
        per_bin = smoke_n // n_bins
        out: list[dict] = []
        for b in range(n_bins):
            start = len(ordered) * b // n_bins
            end = len(ordered) * (b + 1) // n_bins
            bucket = list(ordered[start:end])
            rng.shuffle(bucket)
            out.extend(bucket[:per_bin])
        rng.shuffle(out)  # final shuffle so length-sort during eval still orders correctly
        return out[:smoke_n]
    # Fall back: task-stratified
    by_task: dict[str, list] = defaultdict(list)
    for s in samples:
        by_task[s["task"]].append(s)
    per_task = max(1, smoke_n // len(by_task))
    out = []
    for items in by_task.values():
        rng.shuffle(items)
        out.extend(items[:per_task])
    rng.shuffle(out)
    return out[:smoke_n]


MAX_TABLE_CHARS = 30000  # Upstream cap on Table_markdown before prompt assembly.
                         # Rarely triggers given token-budgeted assembly below.

_FORMAT_INSTR = (
    'Reason step by step. Then output ONLY a single JSON object on the last '
    'line: {"answer": <value>}'
)


def build_prompt_tcot(sample: dict, tokenizer=None, max_input_tokens: int | None = None) -> str:
    """Build the TCoT prompt.

    When `tokenizer` + `max_input_tokens` are supplied, we budget tokens so
    that the HEAD (instruction + title + question + format instruction) is
    guaranteed present, and the TABLE takes whatever token budget remains.
    This prevents right-truncation from silently removing the question —
    measured 12% of TReB English samples would have lost the question at a
    2K token cap, producing bogus (question-less) predictions.

    When no tokenizer is given (e.g. load_treb's length-proxy sorting), we
    fall back to the legacy char-capped table assembly — still correct for
    samples that fit the cap, and the HF backend will re-call with
    tokenizer-aware budgeting during batch generation.
    """
    table_md = sample.get("table_markdown") or ""
    instruction = sample.get("instruction") or ""
    title = sample.get("title") or ""
    question = sample["question"]

    # Char-side table cap (cheap pre-filter; token-budgeted truncation below
    # handles the exact fit when a tokenizer is provided).
    if len(table_md) > MAX_TABLE_CHARS:
        table_md = table_md[:MAX_TABLE_CHARS] + "\n... [table truncated: char cap] ..."

    head_parts = [instruction]
    if title:
        head_parts.append(title)
    head = "\n\n".join(head_parts)
    tail = f"Question: {question}\n\n{_FORMAT_INSTR}"

    if tokenizer is None or not max_input_tokens:
        # Legacy path: concatenate. Fine for shorter samples; use with caution
        # on long ones (relies on downstream truncation which can drop `tail`).
        pieces = [p for p in (head, table_md, tail) if p]
        return "\n\n".join(pieces)

    # Token-budgeted assembly: guarantee head+tail survive, budget the table
    # to fill whatever's left. Leave 16 tokens of slack for separators and
    # any BOS/EOS the tokenizer adds during encode().
    SAFETY_SLACK = 16
    head_ids = tokenizer.encode(head, add_special_tokens=False) if head else []
    tail_ids = tokenizer.encode(tail, add_special_tokens=False)
    table_budget = max_input_tokens - len(head_ids) - len(tail_ids) - SAFETY_SLACK
    if table_budget <= 0:
        # Head + tail alone already exceed budget. Just return head+tail;
        # tokenizer truncation will handle (still tail-preserving if truncation
        # side is set appropriately).
        return f"{head}\n\n{tail}" if head else tail

    if not table_md:
        return f"{head}\n\n{tail}" if head else tail

    table_ids = tokenizer.encode(table_md, add_special_tokens=False)
    if len(table_ids) > table_budget:
        table_ids = table_ids[:table_budget]
        table_text = tokenizer.decode(table_ids, skip_special_tokens=True)
        table_text = table_text + "\n... [table truncated: token budget] ..."
    else:
        table_text = table_md

    pieces = []
    if head:
        pieces.append(head)
    pieces.append(table_text)
    pieces.append(tail)
    return "\n\n".join(pieces)


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
    # Token-budgeted prompt assembly — guarantees question + format instruction
    # survive even when the table is long. See build_prompt_tcot docstring.
    max_input = config["eval"].get("max_input_tokens")
    prompts = []
    for s in samples:
        # Leave ~chat_template overhead room when computing budget (approx 20 tokens).
        budget = max(0, (max_input - 20)) if max_input else None
        user_content = build_prompt_tcot(s, tokenizer=tokenizer, max_input_tokens=budget)
        msgs = [{"role": "user", "content": user_content}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    if max_input:
        lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts[:min(500, len(prompts))]]
        lens.sort()
        over = sum(1 for l in lens if l > max_input)
        print(
            f"[eval] prompt-token stats (sample of {len(lens)}): "
            f"p50={lens[len(lens)//2]} p95={lens[int(0.95*len(lens))]} "
            f"max={lens[-1]} cap={max_input} >cap={over}",
            flush=True,
        )
    outs = llm.generate(prompts, sp)
    # vLLM returns outputs in same order as prompts
    return [o.outputs[0].text for o in outs]


def run_hf_causal(config: dict, samples: list[dict]) -> list[str]:
    """HF transformers backend for decoder-only models (Qwen3, Llama, etc.).

    Used as a fallback when vLLM isn't available (e.g. image ships an older
    vllm that doesn't know about Qwen3). Batched, left-padded, with chat
    template applied via tokenizer.apply_chat_template.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m, g = config["model"], config["generation"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[m.get("dtype", "bfloat16")]

    tokenizer = AutoTokenizer.from_pretrained(m["name_or_path"], revision=m.get("revision") or None)
    # Decoder-only generation needs left-padding so attention masks line up
    # and positions of the generated tokens are correct.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        m["name_or_path"],
        revision=m.get("revision") or None,
        dtype=dtype,
        device_map=m.get("device_map", "auto"),
    )
    model.eval()

    batch_size = config["eval"].get("batch_size", 4)
    max_input_tokens = config["eval"].get("max_input_tokens", 8192)
    max_new_tokens = g.get("max_new_tokens", 256)

    # Build chat-templated prompts
    CHAT_OVERHEAD = 40
    raw_prompts = [
        build_prompt_tcot(s, tokenizer=tokenizer, max_input_tokens=max_input_tokens - CHAT_OVERHEAD)
        for s in samples
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]
    lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    sort_idx = sorted(range(len(prompts)), key=lambda i: lens[i])
    sorted_prompts = [prompts[i] for i in sort_idx]
    lens_sorted = sorted(lens)
    print(f"[eval] prompt-token stats: min={lens_sorted[0]} "
          f"p50={lens_sorted[len(lens)//2]} p95={lens_sorted[int(0.95*len(lens))]} "
          f"max={lens_sorted[-1]} batch_size={batch_size}", flush=True)

    # Incremental writes so we survive crashes
    partial_out = Path(config["output"]["predictions_path"]).with_suffix(".partial.jsonl")
    partial_out.parent.mkdir(parents=True, exist_ok=True)
    partial_out.write_text("")

    sorted_outs: list[str] = [""] * len(samples)
    with partial_out.open("a") as pf:
        for i in range(0, len(samples), batch_size):
            batch_prompts = sorted_prompts[i:i + batch_size]
            batch_samples = [samples[sort_idx[i + j]] for j in range(len(batch_prompts))]
            enc = tokenizer(
                batch_prompts, padding=True, truncation=True, max_length=max_input_tokens,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Strip the input prompt prefix from each generated sequence
            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            decoded = []
            for j, (seq, in_len) in enumerate(zip(gen, input_lens)):
                new_tokens = seq[enc["input_ids"].shape[1]:]
                decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
            for j, d in enumerate(decoded):
                sorted_outs[i + j] = d
                rec = {**batch_samples[j], "prediction": d, "variant": config["variant"]}
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i // batch_size) % 5 == 0:
                pf.flush()
                import os; os.fsync(pf.fileno())
            if (i // batch_size) % 5 == 0:
                print(f"[eval]   progress {i + len(batch_prompts)}/{len(samples)}", flush=True)

    outs: list[str] = [""] * len(samples)
    for sort_pos, orig_idx in enumerate(sort_idx):
        outs[orig_idx] = sorted_outs[sort_pos]
    return outs


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
    # Stop on <end_of_turn> (id 107 in Gemma tokenizers) in addition to <eos>.
    # T5Gemma v1 UL2-IT never fires <eos> on its own — it repeats the answer
    # block until max_new_tokens. <end_of_turn> is the chat-template turn
    # marker, which the IT checkpoint *does* emit. Pass BOTH as stopping
    # criteria so generation halts cleanly after the model's single response.
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        base_eos = tokenizer.eos_token_id
        eos_ids = [base_eos, eot_id] if isinstance(base_eos, int) else list(base_eos) + [eot_id]
        gen_kwargs["eos_token_id"] = eos_ids
        print(f"[eval] stop tokens: eos={base_eos} end_of_turn={eot_id}  combined={eos_ids}")

    # Token-budgeted prompt assembly so head (instr/title) + tail (question +
    # format instruction) always survive; table is budgeted with the leftover
    # tokens. See build_prompt_tcot docstring.
    use_chat_template = getattr(tokenizer, "chat_template", None) and \
        config["eval"].get("apply_chat_template", True)
    # Reserve ~40 tokens of budget for the chat template wrapper so that
    # after apply_chat_template the total stays under max_input_tokens and
    # truncation doesn't clip the trailing "<start_of_turn>model\n" cue.
    CHAT_TEMPLATE_OVERHEAD = 40 if use_chat_template else 0
    prompt_budget = max_input_tokens - CHAT_TEMPLATE_OVERHEAD
    raw_prompts = [
        build_prompt_tcot(s, tokenizer=tokenizer, max_input_tokens=prompt_budget)
        for s in samples
    ]
    if use_chat_template:
        # IT checkpoints (T5Gemma v1 UL2-IT, etc.) expect chat-template framing
        # — <start_of_turn>user / <end_of_turn> markers. Without this wrap, the
        # model is OOD from its SFT distribution → repetition loops + wrong
        # EOS behavior. Mirrors what run_vllm already does.
        texts_all = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in raw_prompts
        ]
        print(f"[eval] applied chat template to {len(texts_all)} prompts")
    else:
        texts_all = raw_prompts
    lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_all]

    # Length-sorted batching: sort by length ascending so each batch has
    # similar-length samples → minimal padding waste on high-variance data.
    # Unsort predictions to original order at the end.
    sort_idx = sorted(range(len(samples)), key=lambda i: lens[i])
    sorted_texts = [texts_all[i] for i in sort_idx]
    lens_sorted = sorted(lens)
    over_cap = sum(1 for l in lens if l > max_input_tokens)
    print(
        f"[eval] prompt-token stats: min={lens_sorted[0]} "
        f"p50={lens_sorted[len(lens)//2]} p95={lens_sorted[int(0.95*len(lens))]} "
        f"max={lens_sorted[-1]} cap={max_input_tokens} >cap={over_cap}  "
        f"batch_size={batch_size}",
        flush=True,
    )

    # Incremental writes: append each batch's predictions to a .partial.jsonl
    # immediately (fsync'd). Survives any OOM / pod termination / crash.
    # At end we rewrite as the canonical predictions.jsonl in original order.
    partial_out = Path(config["output"]["predictions_path"]).with_suffix(".partial.jsonl")
    partial_out.parent.mkdir(parents=True, exist_ok=True)
    partial_out.write_text("")  # truncate any prior run

    sorted_outs: list[str] = [""] * len(samples)
    FLUSH_EVERY_N_BATCHES = 5
    continue_on_error = config["eval"].get("continue_on_error", False)
    err_marker_prefix = "[EVAL_ERROR] "
    with partial_out.open("a") as pf:
        for i in range(0, len(samples), batch_size):
            batch_texts = sorted_texts[i:i + batch_size]
            batch_samples = [samples[sort_idx[i + j]] for j in range(len(batch_texts))]
            try:
                enc = tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=max_input_tokens,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, **gen_kwargs)
                decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            except Exception as e:
                if not continue_on_error:
                    raise
                # Record the failure, move on. Useful for threshold diagnostics
                # where later samples may succeed even after an earlier crash.
                msg = f"{type(e).__name__}: {e}"
                print(f"[eval]   batch {i // batch_size} FAILED ({len(batch_texts)} samples): {msg[:300]}", flush=True)
                decoded = [f"{err_marker_prefix}{msg}"] * len(batch_texts)
                try:
                    import torch as _t
                    _t.cuda.empty_cache()
                except Exception:
                    pass
            for j, d in enumerate(decoded):
                sorted_outs[i + j] = d
                rec = {**batch_samples[j], "prediction": d, "variant": config["variant"]}
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Flush + fsync every N batches so partial survives sudden death.
            if (i // batch_size) % FLUSH_EVERY_N_BATCHES == 0:
                pf.flush()
                import os
                os.fsync(pf.fileno())
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

    # target_ids (optional): if set, restrict evaluation to exactly these IDs.
    # Two use cases:
    #   (1) Targeted diagnostics — e.g. bug-threshold bisection on a handful
    #       of specific IDs. Pass a small list, no --smoke.
    #   (2) Pre-filtered benchmark set — e.g. a kept_ids.json from a length/
    #       bug-safety filter. Pass the full filtered list; combine with
    #       --smoke to sample down to smoke_size stratified over the subset.
    target_ids = config["eval"].get("target_ids")
    if isinstance(target_ids, str):
        # support indirection: path to a JSON list of ids
        target_ids = json.loads(Path(target_ids).read_text())
    if target_ids:
        print(f"[eval] target_ids mode: loading full dataset, filtering to {len(target_ids)} specific ids")
        all_samples = load_treb(
            config["eval"]["dataset"], config["eval"]["language"], smoke_n=None
        )
        by_id = {s["id"]: s for s in all_samples}
        missing = [tid for tid in target_ids if tid not in by_id]
        if missing:
            raise ValueError(f"target_ids not found in dataset: {missing[:5]}{'...' if len(missing)>5 else ''}")
        samples = [by_id[tid] for tid in target_ids]
        if smoke_n and len(samples) > smoke_n:
            # Length-stratified sample of the target subset — same logic as
            # load_treb's smoke path, applied here after target filtering.
            def _char_len(s: dict) -> int:
                return (len(s.get("instruction", "")) + len(s.get("question", ""))
                        + len(s.get("table_markdown", "")))
            ordered = sorted(samples, key=_char_len)
            rng = random.Random(42)
            n_bins = 4
            per_bin = smoke_n // n_bins
            out: list[dict] = []
            for b in range(n_bins):
                start = len(ordered) * b // n_bins
                end = len(ordered) * (b + 1) // n_bins
                bucket = list(ordered[start:end])
                rng.shuffle(bucket)
                out.extend(bucket[:per_bin])
            rng.shuffle(out)
            samples = out[:smoke_n]
            print(f"[eval] smoke-stratified {len(samples)} samples from target_ids subset")
    else:
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
    elif backend == "hf_causal":
        preds = run_hf_causal(config, samples)
    else:
        raise ValueError(f"unknown backend: {backend}")

    out_path = Path(config["output"]["predictions_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s, p in zip(samples, preds):
            rec: dict[str, Any] = {**s, "prediction": p, "variant": config["variant"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {len(preds)} predictions to {out_path}")

    # Explicit completion sentinel for the monitor to watch. Only written on
    # successful end-of-run, AFTER predictions.jsonl is fully on disk.
    # If eval crashes, no sentinel → monitor leaves pod alive for diagnosis.
    import hashlib
    from datetime import datetime, timezone
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    done_path = out_path.with_suffix(".jsonl.done")
    done_data = {
        "variant": config["variant"],
        "n_predictions": len(preds),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictions_path": str(out_path),
        "predictions_sha256": sha,
        "predictions_bytes": out_path.stat().st_size,
    }
    done_path.write_text(json.dumps(done_data, indent=2))
    print(f"[eval] sentinel written: {done_path}")


if __name__ == "__main__":
    main()
