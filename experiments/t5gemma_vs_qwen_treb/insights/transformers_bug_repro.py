"""Minimal reproducer: T5Gemma 2 batched inference with inputs > ~2K tokens
crashes with a shape-mismatch in attention_mask.

Tested on:
  - transformers == 5.x (whatever shipped T5Gemma 2 support; PR #41834)
  - torch 2.8, CUDA 12.8, H100 NVL 94GB
  - model: google/t5gemma-2-4b-4b

Error:
  RuntimeError: The size of tensor a (X) must match the size of
  tensor b (Y) at non-singleton dimension 3

Where X is the current batch's padded sequence length and Y is some
other value, consistent across max_input_tokens settings above 2K
(observed Y=6326 across attempts at max=8K, 16K, 32K).

Reproduces with BOTH attn_implementation="eager" and "sdpa".
flash_attention_2 is not supported for T5Gemma2ForConditionalGeneration.

Config relevant: decoder uses sliding window attention
(sliding_window=1024, _sliding_window_pattern=6) and merged self+cross
attention. Likely bug in mask construction for batched padded inputs
above a certain length threshold.

Related issues in the Gemma family (all closed/fixed — the pattern is
well-known, just not yet filed for T5Gemma 2):
  - huggingface/transformers#37219 — RecurrentGemma crashes past SWA width
  - huggingface/transformers#35290 — Custom 4D tensor shape mismatch
  - huggingface/transformers#31931 — Gemma 2 BF16 inference fails
  - vllm-project/vllm#14881   — Gemma 3 batch + SWA

Workaround: max_length <= ~2048 for batched inference. Works at 2K.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "google/t5gemma-2-4b-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL, dtype=torch.bfloat16,
    attn_implementation="eager",  # same error with sdpa
    device_map="auto",
).eval()

# Two samples of VERY different lengths, above the sliding_window (1024).
short = "Summarize this table: " + "| col | val |\n|---|---|\n| a | 1 |\n" * 5
long = ("Analyze this table: " + "| col | val |\n|---|---|\n| a | 1 |\n" * 1500)
texts = [short, long]

for max_len in [2048, 4096, 8192]:
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(model.device)
    print(f"\n--- max_length={max_len}  input shape={tuple(enc['input_ids'].shape)} ---")
    try:
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, do_sample=False)
        print(f"  OK — output shape {tuple(out.shape)}")
    except RuntimeError as e:
        print(f"  FAIL — {e}")
        break
