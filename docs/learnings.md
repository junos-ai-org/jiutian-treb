# Learnings — T5Gemma 2 QLoRA SFT

Gotchas we hit building this pipeline, with fixes. Scoped to this project's
stack (transformers v5, peft, bitsandbytes, T5Gemma 2, Open-Orca/FLAN).
Generic RunPod-level gotchas live in the `runpod-training-pipeline` skill.

## Model naming

- **T5Gemma 2 dropped the `-ul2` / `-prefixlm` / `-it` suffixes** that v1
  had. The v2 family is just:
  - `google/t5gemma-2-270m-270m`
  - `google/t5gemma-2-1b-1b`
  - `google/t5gemma-2-4b-4b`
- `google/t5gemma-2-4b-4b-ul2` does **not** exist; it will 404 during
  `from_pretrained` with a confusing error.
- v2 requires `transformers >= 5.0.0` (the `t5gemma2/` model dir was added
  in transformers PR #41834, Dec 2025, and shipped in v5.0.0). Pin explicitly.

## transformers v5 Trainer API

These changed from v4 and are easy to miss because v4 examples dominate the web.

| v4 | v5 | Notes |
|---|---|---|
| `Trainer(tokenizer=tokenizer)` | `Trainer(processing_class=tokenizer)` | v5 raises `TypeError: got an unexpected keyword argument 'tokenizer'` — not a deprecation warning, a hard error |
| `from_pretrained(torch_dtype=...)` | `from_pretrained(dtype=...)` | `torch_dtype=` still accepted with a DeprecationWarning; prefer `dtype=` |
| `from_pretrained(load_in_4bit=True)` shortcut | `BitsAndBytesConfig(load_in_4bit=True, ...)` as `quantization_config=` | Top-level shortcut removed in v5 |
| `from_pretrained(...)` with bnb, no device_map | Must pass `device_map="auto"` | Without it, 4-bit weights stay on CPU and forward fails (bnb 4-bit kernels are CUDA-only) |

`eval_strategy`, `hub_token`, `hub_private_repo`, `hub_strategy`, and
`predict_with_generate` are all unchanged in v5.

## Encoder-decoder LoRA gotchas

### `prepare_decoder_input_ids_from_labels` is broken on T5Gemma 2

`DataCollatorForSeq2Seq(model=model, ...)` calls
`model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])` to
precompute `decoder_input_ids`. T5Gemma 2's implementation has a **broken
signature** — doesn't accept `labels=` — so collation crashes on the first
batch:

```
TypeError: T5Gemma2PreTrainedModel.prepare_decoder_input_ids_from_labels()
got an unexpected keyword argument 'labels'
```

**Fix**: don't pass `model=` to the collator. The model will shift labels
internally during forward via `_shift_right` — standard encoder-decoder
behavior.

```python
# BAD (on T5Gemma 2):
collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

# GOOD:
collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)
```

### Encoder input grads don't flow without `enable_input_require_grads`

When training LoRA on a kbit-quantized encoder-decoder with gradient
checkpointing, the encoder's input embeddings need to request gradients
or the LoRA adapters inside the encoder never update. `peft`'s
`prepare_model_for_kbit_training` handles this for the kbit + reentrant
path, but **calling it explicitly is idempotent and protects against the
non-reentrant / non-kbit edge cases**:

```python
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.enable_input_require_grads()   # defensive, safe to always call
```

Symptom if missing: loss plateaus at the pretraining baseline — nothing
actually learns.

### PEFT adapter merging does NOT shape-validate the base

`PeftModel.from_pretrained(base, adapter)` happily attaches LoRA deltas
onto a base with matching layer names, *even if it's the wrong base*. No
error, no warning — you get an output model with junk weights that runs
cleanly but evaluates terribly. Nobody knows why the numbers tanked.

This isn't theoretical: our `models/t5gemma-2-4b-sft/README.md` had a
`--base google/t5gemma-2-4b-4b-ul2` example, a model id that doesn't
exist today (v2 dropped the `-ul2` suffix). Copy-paste would 404 — the
*safe* failure. The dangerous failure is: if someone later publishes a
real `-ul2` variant, that README silently merges our FLAN-SFT adapter
(trained on `t5gemma-2-4b-4b`) onto a different base, producing junk.

**Fix**: read `base_model_name_or_path` from `adapter_config.json` (PEFT
writes this at training time, always correct) and use it as the default
`--base`. See `merge_adapter.py:resolve_base`.

### LoRA task type

Use `task_type=TaskType.SEQ_2_SEQ_LM` — NOT `CAUSAL_LM`. SFTTrainer from
TRL is decoder-only and silently does the wrong thing on encoder-decoder;
use `transformers.Seq2SeqTrainer` instead.

## Data loading

### `load_dataset("Open-Orca/FLAN", split="train")` pulls the entire shard set

Default (non-streaming) mode materializes all 2157 parquet shards — hundreds
of GB, ~2 hours on a typical RunPod connection — even if you only want
100K rows. `.select()` runs *after* the full download.

**Fix**: streaming + `.take(total)`, then optionally materialize:

```python
from datasets import Dataset, load_dataset
stream = load_dataset("Open-Orca/FLAN", split="train", streaming=True,
                      cache_dir="/workspace/hf-cache")
stream = stream.shuffle(seed=42, buffer_size=10_000)   # shuffle over buffer
samples = list(stream.take(total))
ds = Dataset.from_list(samples)
```

`buffer_size=10_000` gives good shuffling within each file-read window.
Note: streaming's shuffle is weaker than full-dataset shuffle, but fine
for SFT.

### FLAN column names

Confirmed columns on `Open-Orca/FLAN` (as of 2026-04-18):
`inputs` (str), `targets` (str), plus metadata
`_template_idx`, `_task_source`, `_task_name`, `_template_type`.
Our preprocess only keeps `inputs`/`targets`; remove_columns drops the rest.

## Smoke test numbers (for sanity-checking future runs)

First successful smoke on 2026-04-19:
- config: 200 samples, 50 steps, QLoRA r=16, max_input=1024, max_target=256, batch=2, grad_accum=4 (effective=8)
- trainable: 68.3M / 7.58B = 0.90%
- loss curve:

| step | train loss | eval loss |
|-----:|-----------:|----------:|
| 5    | 7.93       | —         |
| 25   | 0.96       | 0.78      |
| 50   | 0.49       | 0.32      |

- LoRA adapter size: 273 MB (fp32 weights, ~68M params × 4 B)
- Target modules merged from `all-linear`:
  `q/k/v/o_proj`, `fc1`, `fc2`, `gate_proj`, `up_proj`, `down_proj`,
  `self_attn.out_proj`
- Total SFT duration (model-load + train + eval + save): ~5 min on
  A100 SXM 80GB after base weights cached on volume

If a future smoke doesn't show a clean monotonic loss drop to well under
1.0 by step 50, something's wrong (grad flow, collator, or data).

## Checkpoints: LoRA adapter ≠ full model

The 273 MB `adapter_model.safetensors` is **just the LoRA deltas**. To use
the fine-tuned model you need:

1. **Adapter mode** — base loaded on the fly, PEFT applies the adapter:
   ```python
   base = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-4b-4b")
   model = PeftModel.from_pretrained(base, "org/your-adapter")
   ```
2. **Merged model** — run `merge_adapter.py` to produce a standalone ~17 GB
   model compatible with vLLM/TGI. Required for the TReB eval harness.

## HuggingFace Hub push

Verified flow (confirmed by pushing the smoke adapter on 2026-04-19):

- Fine-grained HF tokens with org-scoped Read+Write permissions work
- `HfApi.create_repo(..., exist_ok=True)` is idempotent
- `HfApi.upload_folder(..., ignore_patterns=["*.pt","optimizer*","scheduler*","rng_state*"])`
  skips the optimizer state (useful if you only want the deployable artifact)
- 614 MB upload takes ~10 s from a US-WA-1 RunPod pod (60 MB/s)
- `Seq2SeqTrainingArguments(push_to_hub=True, hub_strategy="checkpoint")`
  pushes each local checkpoint as a separate commit on the Hub repo —
  survives pod death, and you can resume from the Hub if the network
  volume is wiped

**Always verify push capability BEFORE the full run.** A 6-hour SFT that
crashes at save_steps=500 on a token scope issue is a bad day. Push a
1-byte file to the target repo first.

## RunPod API inconsistencies

RunPod has three surfaces (REST pods, REST volumes, GraphQL) that disagree
about which data centers exist. Concretely, as of 2026-04-19:

- `GET /dataCenters` (graphql) lists `US-MO-1`, `US-NE-1`, `US-MO-2`
  with `storageSupport: True`.
- `POST /networkvolumes` accepts those DC IDs and creates the volume.
- **`POST /pods` has an `enum` on `dataCenterIds` that doesn't include
  them**, so the pod request is rejected even though the volume is
  there waiting.

Workflow: **always cross-check the pod-create enum before creating a
network volume**, or you'll end up with an orphan volume in an
inaccessible DC. The enum from the error response is the source of
truth for pod DCs. Currently missing from the pod enum (but listed by
graphql): `US-MO-1`, `US-MO-2`, `US-NE-1`.

Practical implication: 4× H100 stock in US-MO-1 (Medium) is unreachable
for us; 4× H100 in enum-accessible DCs is Low stock in EU-NL-1,
EUR-IS-3, US-NE-1 only.

## Wall-clock reality for T5Gemma 2 4B-4B QLoRA

Measured on 1× A100 SXM 80GB, batch=8, grad_accum=4 (effective 32),
max_input=1024, max_target=384, bf16, paged_adamw_8bit, grad checkpointing,
`predict_with_generate=false`, attn=eager:

**~12.3 s/step → ~10.7 h for 100K samples, 1 epoch, 3125 steps.**

Why encoder-decoder is slower than a decoder-only of the same param
count: the step runs both an encoder forward AND a decoder forward
(plus the decoder's cross-attention over the encoder output), roughly
2× the compute per step. Budget accordingly.

H100 is ~2× an A100 on this workload → ~5.4 h for the same config on 1×
H100. DDP scaling is near-linear for the forward/backward but each
optimizer step has an all-reduce, so raising effective batch (fewer
optimizer steps) saves more time than adding GPUs at fixed effective
batch.

## Iteration loop: CODE_REPO mode eliminates image rebuilds for code-only changes

`run.sh` supports `CODE_REPO` + `CODE_REF` env vars; when set, the pod
git-clones the repo to `/workspace/code` and `cd`s in. Every restart
does a `git pull`. Verified working — the full SFT pod picked up commit
`8895b37` (config change) without a rebuild.

Caveats:
- `requirements.txt` changes still need a rebuild (deps are in the image).
- `run.sh` changes need a rebuild (run.sh itself is baked into the image
  and used to launch CODE_REPO mode).
- Repo must be public, or add credential helper for private repos.

## wandb in the pod: just set `WANDB_API_KEY`

Don't call `wandb login` in `run.sh`. Setting `WANDB_API_KEY=<key>` and
`WANDB_PROJECT=<project>` as pod env vars is enough — Trainer's wandb
callback reads them at `report_to=["wandb"]` init time and logs in
automatically. Verified working on 2026-04-19.

## DDP with HF Trainer: just use `torchrun`

No code changes required in `train.py`. Trainer auto-detects world size,
handles per-GPU data sharding, and makes rank 0 the sole saver/pusher.
All that's needed in `run.sh`:

```bash
if [[ "${NUM_GPUS:-1}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py --config "$CONFIG"
else
  python train.py --config "$CONFIG"
fi
```

## T5Gemma 2 inference bug — empirical bug threshold

Precisely bisected on 2026-04-20 via targeted 9-sample diagnostic spanning
2.5K-25K input tokens at `batch=1`. Filed upstream as
[huggingface/transformers#45521](https://github.com/huggingface/transformers/issues/45521).

**Threshold: inputs > ~4094 tokens crash.** Deterministic, not stochastic —
every sample >=5015 tokens failed in our test; every sample <=3525 passed.

Error fingerprint: `RuntimeError: The size of tensor a (4097) must match b (N)`
where:
- **`tensor a (4097)` is constant** across every failure regardless of input length
- **`tensor b` = input_length + 3** (three special tokens the tokenizer adds)
- Suggests `4097 = 4 * sliding_window(1024) + 1 query token` — a pre-allocated
  attention buffer that doesn't resize beyond 4 windows.

Workaround matrix:
- `attn_implementation="sdpa"` → same class of shape mismatch
- `attn_implementation="flash_attention_2"` → `ValueError` (T5Gemma2 doesn't
  support it yet)
- Only `max_input_tokens <= 4094` works. At 4094 with token-budgeted prompt
  assembly (see `build_prompt_tcot`), 100% of TReB English samples run.
- ~10% of TReB English samples need their table truncated to fit in 4K.

**Earlier notes in this file claimed the threshold was 2K or 7K.** Both were
overreach from insufficient data. The 4K bisection is the accurate answer.
Config values involved: `sliding_window: 1024`, `_sliding_window_pattern: 6`,
`max_position_embeddings: 131072` (advertised 128K, actual usable at batch=1 is 4K).

## Right-truncation silently drops the question

HF tokenizers default to `truncation_side="right"`. Our original
`build_prompt_tcot` assembled `[instruction, title, table, question,
format_instr]` — so any sample that tokenizes past `max_length` has the
question AND the JSON-format instruction cut off. The model then
generates an answer to a table with no question, and we score that.

Measured: at `max_length=2048` on TReB English, **12% of samples (937/7789)
exceeded the cap**. That's 1-in-8 T5Gemma predictions in the previous
runs that were made without the model ever seeing the question — a real
correctness bug masquerading as poor model performance.

**Fix**: token-budgeted prompt assembly. Encode `head = instruction + title`
and `tail = question + format_instr` separately, budget the table with
the remaining tokens, concatenate. Guarantees Q + format survive.
Implemented in `experiments/t5gemma_vs_qwen_treb/eval.py:build_prompt_tcot`.

## Incremental prediction writes are non-negotiable

eval.py wrote `predictions.jsonl` only at the end of the run. RunPod
force-exited our pods at the $80 spend cap ~10 h into a 25h+ run — all
in-memory predictions were lost, zero on disk.

Fix that shipped: append each batch's predictions to
`predictions.partial.jsonl` with `flush()` + `os.fsync()` every 5
batches. Survives sudden death. At end, rewrite in original order as
`predictions.jsonl`.

Verified on the next diagnostic: eval crashed at sample 96/100, and we
had **95 predictions safely on disk** for analysis. Loss: 1 sample
that was mid-batch.

## RunPod spend cap hits like a guillotine

User-set `spendLimit` is a HARD cap. When spend approaches it, RunPod
force-exits ALL running pods (state `EXITED`, not terminated — container
disk preserved but no active compute). This happens without warning to
the pod; they're SIGKILL'd.

Combined with "predictions.jsonl written only at end" this cost us 10 h
× $6.14 of compute with zero results. Mitigations, in order of ROI:

1. Incremental writes (above) — most important
2. Monitor `myself.clientBalance` via graphql before + during long runs
3. Estimate worst-case cost before launching, reserve 2× headroom
4. Raise spendLimit to match the budget + buffer (requires user approval)

When scaling effective batch with DDP, **sqrt-scale the learning rate**
from your 1-GPU baseline (`lr_new = lr_base × √(batch_new / batch_base)`)
and bump `warmup_ratio` slightly — raising batch 4× at the same LR is
usually fine for LoRA; linear scaling (4×) is more aggressive than
needed.
