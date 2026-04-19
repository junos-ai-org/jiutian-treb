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
