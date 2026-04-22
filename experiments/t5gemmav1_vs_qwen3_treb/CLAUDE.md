# Experiment: T5Gemma v1 2B-2B UL2-IT vs Qwen3-4B on TReB

**Goal**: Compare T5Gemma v1 2B-2B UL2-IT (encoder-decoder, Google-IT'd) vs
Qwen3-4B-Instruct-2507 (decoder-only, Qwen-IT'd) on TReB English / TCoT.

**Why this is a cleaner read than the sibling `t5gemma_vs_qwen_treb`**:
- Both models are natively instruction-tuned by their vendors — no DIY SFT
  confound.
- Both in the 4–6B class (not exact param-match, but same regime).
- The comparison isolates architecture + pretraining recipe, with IT-quality
  controlled by "whatever the vendor shipped."

**Gate before building**: pre-filter TReB to samples whose TCoT prompt fits
under the SWA-bug threshold when tokenized with the T5Gemma v1 tokenizer.
Same filtered set is used for both variants → apples-to-apples.
