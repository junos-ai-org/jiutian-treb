# T5Gemma v1 2B-2B UL2-IT vs Qwen3-4B on TReB

**Status:** `COMPLETE — 3-way 248-sample comparison, T5Gemma v1 / Gemma-2-2B / Qwen3-4B, see insights/three_way_2026-04-23.md`

## TL;DR

| Variant | Params | Arch | Mean | CORRECT% | TRUNCATED% |
|---|---:|---|---:|---:|---:|
| T5Gemma v1 2B-2B UL2-IT | 5.6B | enc-dec | 4.10 | 32.3% | 17.7% |
| Gemma-2-2B-IT | 2.6B | dec-only | 3.81 | 30.2% | 12.5% |
| **Qwen3-4B-Instruct-2507** | 4.0B | dec-only | **4.99** | **33.1%** | **45.6%** |

**Qwen3-4B wins** despite being ~30% smaller than T5Gemma — 27 sole wins
vs 15 (T5G) vs 13 (G2) on the 55 samples where exactly one model got it
right. Attribution: Qwen3's newer pretraining + Qwen team's post-training,
not architecture. Qwen3's 45.6% truncation rate (vs ~15% for the Gemmas)
is a chain-of-thought length artifact against the 256-token generation cap
— raising to 1024 would likely widen Qwen3's lead further.

Encoder-decoder does win Table_Retrieval/Query/Domain-Ops specifically
against Gemma-2 (same pretraining), but that architecture edge is swamped
by Qwen3's overall capability advantage.

### Scope (2026-04-22, after fast-fail verification)

The first pod run revealed T5Gemma v1 UL2-IT's EOS-never-fires repetition
pathology (see `insights/verification_smoke_2026-04-22.md`). Fix landed;
re-scoping to a tiered rollout rather than a full 7018-sample run:

1. **25-sample EOS-fix verification** on T5Gemma v1 — proves the chat
   template + `<end_of_turn>` stop token fires correctly.
2. **250-sample comparison set** on T5Gemma v1 (stratified from the 7018
   filtered-kept IDs by prompt length).
3. **250-sample comparison set** on Qwen3-4B on the *same* stratified
   250 — apples-to-apples read.
4. Score both + insights writeup.

English only. TCoT only. 4K-token T5Gemma-tokenizer cap. No Chinese.
Full 7018 de-scoped — 250 is plenty to see whether the models are in the
same ballpark, and costs ~$0.50 vs ~$15.

## Hypothesis

Does T5Gemma v1's **encoder-decoder architecture + UL2 pretraining + Google's
full IT pipeline** (SFT + RLHF + model merging) carry any table-reasoning
advantage over a similarly-sized, similarly-well-instruction-tuned
decoder-only model (Qwen3-4B-Instruct-2507)?

This is a cleaner read than the sibling `t5gemma_vs_qwen_treb`: both models
are natively IT'd by their vendors, so there's no DIY-SFT confound.

| Variant | Model | Params | Arch | Context | Instruction-tuning |
|---|---|---|---|---|---|
| `t5gemma_v1` | `google/t5gemma-2b-2b-ul2-it` | 5.6B | Enc-Dec | 8K in / 8K out | SFT + RLHF + merge |
| `qwen3` | `Qwen/Qwen3-4B-Instruct-2507` | ~4B | Dec-only | 32K | Qwen's own IT recipe |

Not strictly param-matched (5.6B vs 4B) — same "2025 small-instruct" regime.

## Setup

### Dataset

TReB English, TCoT mode only. Same scoping as the sibling experiment (PoT and
ICoT are follow-ups; bilingual evaluation is a separate axis).

### Phase 0: bug-safe subset (pre-filter)

Before running eval, `build_dataset.py` filters the 7,789 English TReB
samples to those whose TCoT prompt fits under a 4,000-token cap when
tokenized with the T5Gemma v1 tokenizer. Rationale:

- T5Gemma 2 exhibits an SWA-mask-shape bug above 4094 tokens
  ([HF transformers#45521](https://github.com/huggingface/transformers/issues/45521)).
- T5Gemma v1's SWA window is 4096 (vs v2's 1024); the bug may or may not
  transfer. Start conservatively at 4000; the pod-side verification in
  task 3 of `docs/progress.md` will tell us if we can relax.
- T5Gemma v1's context cap is 8K regardless — about half the dataset's
  long tail exceeds that, so we're truncation-bound even without the SWA bug.

**The same `kept_ids.json` is used by both variants** so the comparison is
on an identical sample subset.

```bash
python build_dataset.py --max-tokens 4000
```

Produces `dataset/{kept_ids,dropped_ids,stats}.json`.

#### Current retention

| | count |
|---|---|
| total English samples | 7,789 |
| **kept** (≤4K T5Gemma v1 tokens) | **7,018** (90.1%) |
| dropped | 771 |

**Tasks disproportionately affected** (<50% retention — cohort-wide
8K-context weakness rather than experimental confound):

| Task | Retention |
|---|---|
| Multi-step_Conditional_Calculation | 1/17 (5.9%) |
| Multi-step_Correlation_Analysis | 4/49 (8.2%) |
| Multi-step_Hypothesis_Testing | 6/61 (9.8%) |
| Multi-step_Operations | 7/61 (11.5%) |
| Multi-step_Retrieval | 6/48 (12.5%) |
| Multi-step_Fact_Checking | 11/61 (18.0%) |
| Table_Column_Naming | 214/500 (42.8%) |

All other 19 tasks retain ≥62%, most at 100%. The Multi-step family's drop
reflects T5Gemma v1's 8K context limit — flagged as a known limitation of
this model, not a methodological issue.

### Reasoning modes

TCoT only. PoT/ICoT are follow-up experiments.

### Metrics

- **ROUGE-L** — text tasks
- **Exact Match** (with numeric equivalence) — numeric tasks
- **LLM-as-Judge** — DeepSeek V3 via OpenRouter (~$5 expected)

### Infrastructure

Two pods, same template as the sibling experiment:

| Pod | Image | GPU | Variant | Est. wall-clock |
|---|---|---|---|---|
| t5gemma-v1 | *TBD — new or rebuild existing `treb-eval-t5gemma`* | 1× H100 | `t5gemma_v1` | 45–90 min (HF, batch=1) |
| qwen3 | *TBD — new or rebuild existing `treb-eval-qwen` w/ vLLM ≥ 0.8* | 1× H100 | `qwen3` | 15–30 min (vLLM) |

**Critical ordering**: T5Gemma v1 runs first. Qwen3 runs only after
T5Gemma v1 produces a complete `predictions.jsonl.done` sentinel — no
point spending GPU minutes on a baseline with nothing to compare to.

### Pinning

| Variant | Pin |
|---|---|
| T5Gemma v1 | `google/t5gemma-2b-2b-ul2-it` (latest main — pretrained reference) |
| Qwen3 | `Qwen/Qwen3-4B-Instruct-2507` (explicit snapshot tag) |

## How to run

### Local scaffolding (phase 0)

```bash
cd experiments/t5gemmav1_vs_qwen3_treb
python build_dataset.py --max-tokens 4000   # writes dataset/kept_ids.json
```

### On a pod

```bash
# smoke-then-full for the variant baked in via VARIANTS env (set per-image)
bash run_all.sh

# or override:
VARIANTS=t5gemma_v1 bash run_all.sh           # just T5Gemma v1
SKIP_FULL=1 VARIANTS=t5gemma_v1 bash run_all.sh  # just smoke
RUN_ALL=0 CONFIG=configs/qwen3_4b_instruct.yaml SMOKE=1 bash run.sh
```

### SCP outputs off the pod

Before killing a pod, pull predictions to local disk:

```bash
# from laptop, once pod is reachable
scp -r root@<pod>:/workspace/results/ experiments/t5gemmav1_vs_qwen3_treb/results/
```

## TL;DR of findings

Comparator pivoted from Qwen3-4B to **Gemma-2-2B-IT** (same Gemma-2
pretraining base as T5Gemma v1 → cleaner architecture-only read). Qwen3-4B
not available on any API path during the experiment; Qwen3-8B partial run
retained for later (49/248 valid, blocked by OpenRouter credits).

**On 248 stratified TReB English / TCoT samples:**

| Variant | Params | Arch | EM | ROUGE-L |
|---|---:|---|---:|---:|
| T5Gemma v1 2B-2B UL2-IT | 5.6B | enc-dec | 0.105 | **0.244** |
| Gemma-2-2B-IT | 2.6B | dec-only | **0.113** | 0.231 |

**Tie on overall performance.** Doubling params via encoder-decoder
structure does not measurably move overall TReB/TCoT when you control for
pretraining lineage.

**Claude Sonnet LLM-judge pass** (via parallel subagents) on the same 248
samples × 2 variants = 496 judgments:

| Metric | T5Gemma v1 | Gemma-2-2B-IT |
|---|---:|---:|
| Mean score (0-10) | **4.10** | 3.81 |
| CORRECT % | **32.3%** | 30.2% |
| TRUNCATED % | 17.7% | 12.5% |
| REFUSAL % | 1.2% | **4.4%** |

**Architecture signal in per-task deltas (≥1 pt gap):**

- T5Gemma wins the **retrieval family** — Table_Retrieval (+1.69),
  Table_Query (+1.50), Table_Domain-specific_Operations (+1.67). Cross-
  attention helps "find this specific thing in the table" problems.
- Gemma-2 wins **Robustness_Evaluation** (+1.76) — decoder-only commits
  to NLI labels more reliably.
- T5Gemma **truncates ~40% more** (longer rationales hit `max_new_tokens=256`).
- Gemma-2 **refuses ~4× more** ("I can't access that table" non-answers).

Writeups: [`insights/judge_sonnet_2026-04-23.md`](insights/judge_sonnet_2026-04-23.md)
(primary) · [`insights/comparison_2026-04-22.md`](insights/comparison_2026-04-22.md)
(string-metrics baseline).
