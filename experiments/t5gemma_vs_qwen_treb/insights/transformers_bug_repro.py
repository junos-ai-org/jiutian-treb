"""Minimal reproducer: T5Gemma 2 decoder self-attention has a fixed 4097-element
mask dimension that can't expand for inputs longer than ~4094 tokens. Crashes
even at batch_size=1 (so it isn't the batched-SWA-padding class of bug).

Summary
-------
`google/t5gemma-2-4b-4b` + `transformers >= 5.0.0` (PR #41834):
at input length > 4094 tokens, `model.generate()` raises:

    RuntimeError: The size of tensor a (4097) must match the size of
    tensor b (N) at non-singleton dimension 3

where:
  - `tensor a (4097)` is CONSTANT across every failing sample we tested
    (at lengths 5015 / 6499 / 7493 / 9997 / 14967 / 19808 / 25138)
  - `tensor b (N)` equals `input_length + 3` (the special tokens the
    tokenizer adds)

Evidence (measured 2026-04-20, batch_size=1, real TReB English samples):

| input tokens | result | error "b" value |
|-------------:|:------:|----------------:|
|         2497 | OK     | -               |
|         3525 | OK     | -               |
|         5015 | FAIL   | 5018            |
|         6499 | FAIL   | 6502            |
|         7493 | FAIL   | 7496            |
|         9997 | FAIL   | 10000           |
|        14967 | FAIL   | 14970           |
|        19808 | FAIL   | 19811           |
|        25135 | FAIL   | 25138           |

Stack trace terminates in
`transformers/models/t5gemma2/modeling_t5gemma2.py:244 eager_attention_forward`
at the line `attn_weights = attn_weights + attention_mask`.

T5Gemma 2 decoder config has:
    sliding_window: 1024
    _sliding_window_pattern: 6
    max_position_embeddings: 131072

The constant `4097` strongly suggests a pre-allocated 4-window buffer
(`4 * sliding_window + 1 = 4097`) that isn't being resized when the input
exceeds 4096 tokens. attn_implementation="sdpa" exhibits the same class of
shape mismatch; flash_attention_2 is not supported for
T5Gemma2ForConditionalGeneration (ValueError from _flash_attn_can_dispatch).

Related (all fixed) Gemma-family bugs with the same flavor:
    huggingface/transformers#37219 — RecurrentGemma crashes past SWA width
    huggingface/transformers#35290 — Custom 4D tensor shape mismatch
    huggingface/transformers#31931 — Gemma 2 BF16 inference
    vllm-project/vllm#14881       — Gemma 3 batch + SWA
    huggingface/transformers#41875 — Flash Attention in Seq2SeqLM.generate

Reproducer
----------
Needs: a single GPU with ~40 GB VRAM (H100, A100 80GB, L40, etc.) and HF
auth for the gated Google model. torch >= 2.5, transformers == 5.x.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "google/t5gemma-2-4b-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
).eval()


def try_length(n_target_tokens: int) -> str:
    filler = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |\n"
    prompt = "Answer concisely.\n\nTable: " + (filler * max(1, n_target_tokens // 20)) + "\n\nQuestion: sum?"
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=n_target_tokens).input_ids
    actual = ids.shape[-1]
    try:
        with torch.no_grad():
            out = model.generate(ids.to(model.device), max_new_tokens=16, do_sample=False)
        return f"OK  (input={actual}, output={out.shape[-1]})"
    except RuntimeError as e:
        return f"FAIL (input={actual}): {e}"


if __name__ == "__main__":
    for n in (2500, 3500, 4000, 4090, 4100, 4500, 5000, 8000):
        print(f"\n--- target={n} ---")
        print(try_length(n))
