# Progress

Fresh log. Prior history archived at `progress.md.bak` (2026-04-20).

## 2026-04-22 — new experiment: T5Gemma v1 2B-2B UL2-IT vs Qwen3-4B

Spinning up a second TReB experiment under
`experiments/t5gemmav1_vs_qwen3_treb/`, sibling to the existing
`t5gemma_vs_qwen_treb` (v2 4B-4B vs Qwen 2.5 7B). Motivation: Google shipped
a fully IT'd T5Gemma v1 (SFT + RLHF + model merging), so we don't need our
own FLAN-SFT in the loop — cleaner architecture read.

Variants:
- `t5gemma_v1_2b2b_ul2_it` — `google/t5gemma-2b-2b-ul2-it` (5.6B, enc-dec, 8K context, Gemma 2 base)
- `qwen3_4b_instruct` — `Qwen/Qwen3-4B-Instruct-2507` (~4B, decoder-only, 32K context)

Not strictly param-matched (5.6B vs 4B) — same "2025 small-instruct" regime.

### Phase 0: bug-safe subset

Cap on encoder-input tokens with the T5Gemma v1 tokenizer, same threshold as
the sibling's v2 runs (4000 tokens; bug threshold per HF transformers#45521
is 4094). **Single filtered set used for both variants** for apples-to-apples
comparison. Run:

```
python experiments/t5gemmav1_vs_qwen3_treb/build_dataset.py --max-tokens 4000
```

Result (written to `experiments/t5gemmav1_vs_qwen3_treb/dataset/stats.json`):

| | total | kept | kept % |
|---|---|---|---|
| all English TCoT samples | 7,789 | 7,018 | **90.1%** |

Token distribution: p50=247, p90=3893, p95=9816, p99=19606, max=28136.
The 4000-token cap sits right above p90 — most samples fit comfortably; a
long tail (Multi-step tasks) is filtered.

**Tasks with <50% retention (flagged in README):**
- Multi-step_Conditional_Calculation: 1/17 (5.9%)
- Multi-step_Correlation_Analysis: 4/49 (8.2%)
- Multi-step_Hypothesis_Testing: 6/61 (9.8%)
- Multi-step_Operations: 7/61 (11.5%)
- Multi-step_Retrieval: 6/48 (12.5%)
- Multi-step_Fact_Checking: 11/61 (18.0%)
- Table_Column_Naming: 214/500 (42.8%)

The Multi-step family is where long-table reasoning matters most, and it's
exactly where T5Gemma v1's 8K context hurts — we'll flag this as a known
limitation of the 2B-2B architecture for this comparison rather than as
an experimental confound. Possible follow-up: run Qwen3 on the dropped tail
as a supplementary "long-table" table.

### Decisions made so far

- **Param-matching**: Qwen3-4B over Qwen3-8B. Not an exact match to T5Gemma
  v1's 5.6B total, but same class + lets us run on 1× H100 in vLLM fast.
- **Same cap for both variants**: filter once with the T5Gemma tokenizer,
  reuse the kept_ids for Qwen. Alternative (run each on its own tokenized
  cap) would confound the comparison.
- **Order of operations**: verify T5Gemma v1 works end-to-end BEFORE spending
  GPU time on Qwen. If the v1 run doesn't complete, the Qwen run has nothing
  to compare against.

### Next

1. Write variant configs + new run.sh/run_all.sh in the experiment dir.
2. New Docker images or rebuild existing with our configs baked in.
3. SSH into a RunPod, smoke-test T5Gemma v1, probe whether the SWA bug
   actually fires on v1 (its SWA window is 4096 vs v2's 1024; the bug may
   or may not transfer).
4. Only after T5Gemma v1 full completes: launch Qwen3.
