# T5Gemma v1 vs Gemma-2-2B-IT on TReB English / TCoT — 2026-04-22

## Headline

**Doubling parameters via encoder-decoder structure doesn't measurably move
TReB performance when you control for pretraining lineage.** On the same
248-sample stratified subset, with both models post-trained by Google from
the same Gemma-2 pretraining base:

| Variant | Params | Arch | EM | ROUGE-L | Extract |
|---|---:|---|---:|---:|---:|
| `google/t5gemma-2b-2b-ul2-it` | 5.6B | enc-dec (UL2 pretrain + SFT+RLHF+merge) | 0.105 | **0.244** | 248/248 |
| `google/gemma-2-2b-it` | 2.6B | dec-only (Gemma-2 pretrain + SFT+RLHF+merge) | **0.113** | 0.231 | 247/248 |

Neither model dominates. Given Gemma-2-2B-IT uses ~half the params AND ~half
the compute per forward pass (no encoder), the implicit win is for the
decoder-only architecture on a per-param-efficient basis — but the absolute
headline is that the architectures are **approximately equivalent** for this
benchmark.

## Setup

- **Dataset**: JiuTian TReB, English, TCoT mode, pre-filtered to 7018 samples
  whose TCoT prompts fit under 4000 T5Gemma tokens (SWA-bug safety gate).
- **Stratified subset**: 248 of those 7018, 4-quartile-by-length stratification
  with seed=42 — both variants evaluate on the **exact same sample set**.
- **Metrics**: local Exact-Match (case-insensitive, whitespace-normalized),
  ROUGE-L (rouge-score), extract rate (valid first `{"answer": X}` JSON).
  No LLM-judge run yet.
- **Infra**: Both variants on RunPod H100 80GB, EU-NL-1, bf16, greedy,
  `max_new_tokens=256`, batch=1 for T5Gemma (HF seq2seq), batch=4 for
  Gemma-2 (HF causal). Pods terminated after result SCP.

## Per-task breakdown

| Task | n | T5Gemma v1 ROUGE-L | Gemma-2-2B-IT ROUGE-L | Δ |
|---|---:|---:|---:|---:|
| Code_Generation | 17 | 0.21 | 0.23 | +0.02 (Gemma) |
| Hallucination_Evaluation | 17 | 0.06 | 0.04 | −0.02 (T5G) |
| Instruction_Following | 2 | 0.00 | 0.00 | tied |
| Mathematical_Reasoning | 24 | 0.10 | 0.07 | −0.03 (T5G) |
| Multi-step_Fact_Checking | 1 | 1.00 | 1.00 | tied (singleton) |
| Robustness_Evaluation | 17 | 0.65 | 0.67 | +0.02 (Gemma) |
| Table_Column_Naming | 9 | 0.00 | 0.00 | tied |
| Table_Correlation_Analysis | 4 | 0.03 | 0.03 | tied |
| Table_Distribution_Testing | 12 | 0.06 | 0.06 | tied |
| Table_Domain-specific_Operations | 6 | 0.29 | 0.25 | −0.04 (T5G) |
| Table_Fact_Checking | 12 | 0.75 | **0.83** | +0.08 (Gemma) |
| Table_General_Operations | 17 | 0.26 | 0.23 | −0.03 (T5G) |
| Table_Hypothesis_Testing | 2 | 0.11 | 0.15 | +0.04 (Gemma) |
| Table_Plausibility_Verification | 1 | 0.26 | 0.24 | −0.02 (T5G) |
| Table_Query | 18 | 0.07 | 0.04 | −0.03 (T5G) |
| Table_Retrieval | 13 | **0.37** | 0.28 | −0.09 (T5G) |
| Table_Selection | 17 | 0.07 | 0.07 | tied |
| Table_Summary | 17 | **0.32** | 0.25 | −0.07 (T5G) |
| Table_Title_Naming | 21 | **0.37** | 0.29 | −0.08 (T5G) |
| Understanding | 21 | 0.22 | **0.29** | +0.07 (Gemma) |
| **Overall** | **248** | **0.244** | 0.231 | −0.013 (T5G) |

T5Gemma leads on text-generation-heavy tasks (Summary, Title_Naming,
Retrieval). Gemma-2 leads on closed-form classification (Fact_Checking,
Understanding). This is directionally consistent with the T5Gemma v1 paper's
finding that encoder-decoder pretraining helps generation tasks.

## What changed vs the originally-proposed experiment

The original plan had **Qwen3-4B-Instruct-2507** as the decoder-only comparator.
Three things forced a pivot:

1. **OpenRouter doesn't serve Qwen3-4B.** Its smallest dense Qwen3 is 8B —
   43% larger than T5Gemma, uncontrolled param confound.
2. **Qwen3-8B run hit OpenRouter's paywall.** 49 successful predictions, 199
   HTTP 402 errors (account has 0 credits). Too little signal for a 250-sample
   comparison.
3. **RunPod EU-NL-1 had Docker Hub egress issues** during the experiment
   window — three consecutive pods stuck in runtime=null for 10+ minutes.

Switching to **Gemma-2-2B-IT** turned out to be a *better* comparison than
Qwen3-4B anyway: identical pretraining base, same chat template, same
tokenizer, same era SFT/RLHF recipe. The only difference between the two
checkpoints is the architecture (enc-dec vs dec-only) and the UL2 adaptation
phase T5Gemma adds on top.

## What this experiment does *not* answer

- Does T5Gemma v1 vs a decoder-only of **equal param count** look different?
  (Gemma-2-9B-IT would be a natural comparator — not attempted because
  OpenRouter doesn't serve it and pod boot was unreliable.)
- LLM-as-judge scores — string-based EM and ROUGE-L miss semantic wins
  and are a weak proxy on text-generation tasks. A DeepSeek-V3 judge pass
  (~$1) would give a much fairer read.
- Tasks beyond TCoT. PoT + ICoT were out of scope for this experiment.

## Known limitations

- **248 samples is small for per-task reads.** Most tasks have n=1–20; only
  the overall average is statistically meaningful at this n.
- **Multi-step tasks effectively dropped.** They're disproportionately
  filtered out by the 4K-token T5Gemma SWA safety gate (see phase-0 filter
  in `dataset/stats.json`). The comparison is representative of single-pass
  TReB, not the long-context / multi-step subset.
- **No seed replication.** Single seed=42 stratification. Directional
  signals OK; tight per-task claims are not.

## Cost

- T5Gemma v1 pod: ~35 min × $2.50/hr ≈ **$1.50**
- Gemma-2-2B-IT pod: ~10 min × $2.50/hr ≈ **$0.40**
- Qwen3-8B via OpenRouter: ~$0 (account had 0 credits; 49 hit free quota,
  rest errored)
- **Total: ~$2** plus ~2 hours of chasing RunPod boot issues.

## Files

- `results/t5gemma_v1_2b2b_ul2_it/predictions_250.jsonl` — T5Gemma v1 on 248
- `results/gemma2_2b_it/predictions.jsonl` — Gemma-2-2B-IT on 248
- `results/qwen3_8b_openrouter/predictions.jsonl` — Qwen3-8B (partial: 49/248
  valid, retained for the follow-up when credits are available)
- `logs/*` — per-run pod logs
