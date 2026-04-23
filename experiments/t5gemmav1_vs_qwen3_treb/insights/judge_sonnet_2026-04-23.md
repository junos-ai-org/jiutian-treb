# Claude Sonnet LLM-judge pass — 2026-04-23

## Setup

- **Judge**: `claude-sonnet-4-6` via parallel subagents (no API key needed —
  uses Claude Code's compute pool).
- **Method**: 496 predictions split into 20 batches of 25, launched in two
  parallel waves of 10 agents each. Each agent scored its batch on the
  12-mode failure taxonomy from `score.py`.
- **Taxonomy**: `CORRECT`, `PARTIAL`, `FORMAT`, `HALLUCINATION`,
  `WRONG_SELECTION`, `NUMERIC`, `REASONING`, `REFUSAL`, `EMPTY`,
  `MISUNDERSTAND`, `TRUNCATED`, `OTHER`.
- **Output**: per-sample {score 0-10, mode, rationale}. Merged into
  `results/judged.jsonl`.

## Headline

| Metric | T5Gemma v1 (5.6B, enc-dec) | Gemma-2-2B-IT (2.6B, dec-only) | Δ |
|---|---:|---:|---:|
| Mean score (0-10) | **4.10** | 3.81 | +0.29 (T5G) |
| CORRECT % | **32.3%** | 30.2% | +2.1 pp (T5G) |
| CORRECT+PARTIAL % | 42.7% | **43.1%** | +0.4 pp (G2) |

**Still effectively a tie.** T5Gemma edges Gemma-2 on strict CORRECT rate
(+2.1pp) and mean score, but Gemma-2 catches up when PARTIAL credit is
given. Directionally consistent with the earlier EM/ROUGE-L result.

## Failure-mode distribution

| Mode | T5Gemma v1 | Gemma-2-2B-IT |
|---|---:|---:|
| CORRECT | **32.3%** | 30.2% |
| WRONG_SELECTION | 25.0% | 22.6% |
| TRUNCATED | **17.7%** | 12.5% |
| PARTIAL | 10.5% | **12.9%** |
| NUMERIC | 7.3% | 6.9% |
| REFUSAL | 1.2% | **4.4%** |
| HALLUCINATION | 1.6% | 2.0% |
| MISUNDERSTAND | 1.2% | 2.4% |
| REASONING | 2.0% | 1.6% |
| OTHER | 0.8% | 2.8% |
| EMPTY | 0.0% | **1.2%** |
| FORMAT | 0.4% | 0.4% |

**Two architecture-correlated failure modes jump out:**

1. **T5Gemma truncates ~40% more often** (17.7% vs 12.5%). The encoder-
   decoder tends to output longer chains of reasoning, hitting
   `max_new_tokens=256` mid-thought. Raising the cap to 512 would likely
   convert ~5pp of those truncations into CORRECT or WRONG_SELECTION.
2. **Gemma-2-2B refuses ~4× more often** (4.4% vs 1.2%) + has 3 EMPTY
   responses. The decoder-only is more prone to "I cannot access that
   table" refusals or silent no-ops on structured-data tasks. Likely a
   difference in SFT/RLHF calibration rather than architecture.

Both models share **WRONG_SELECTION as the dominant failure** (~25% and
22.6%) — picking wrong labels on multi-choice, wrong cells on Table_Query,
flipped entailment labels on Robustness. This is a capability gap both
models have at the 2-5B scale, not an architecture issue.

## Per-task deltas (≥1.0 mean-score gap)

Tasks where one model meaningfully outperforms:

| Task | n | T5G | G2 | Δ | Winner |
|---|---:|---:|---:|---:|---|
| Table_Retrieval | 13 | **5.38** | 3.69 | +1.69 | T5Gemma |
| Table_Query | 18 | **3.89** | 2.39 | +1.50 | T5Gemma |
| Table_Domain-specific_Operations | 6 | **5.00** | 3.33 | +1.67 | T5Gemma |
| Table_Distribution_Testing | 12 | **1.75** | 0.42 | +1.33 | T5Gemma |
| Robustness_Evaluation | 17 | 5.29 | **7.06** | −1.76 | Gemma-2 |

**Pattern: T5Gemma wins the retrieval / targeted-lookup family.** Both
Table_Retrieval and Table_Query ask the model to pull a specific cell
or value from the table; T5Gemma is meaningfully better at both.
Mechanistically consistent with cross-attention — the decoder can
re-attend to the encoded table at every generation step, which is
exactly the shape of "find this specific thing" problems.

**Gemma-2 wins Robustness_Evaluation decisively.** These are NLI-style
labeled classifications (entailment / not_entailment / contradiction /
neutral). Gemma-2 commits to a label more reliably; T5Gemma is more
likely to hedge or flip the label.

## Tasks where both models fail together

Sub-1.0 mean score on both:

| Task | n | T5G | G2 | Note |
|---|---:|---:|---:|---|
| Table_Column_Naming | 9 | 0.56 | 0.89 | Needs column schema inference from data |
| Table_Correlation_Analysis | 4 | 0.50 | 0.00 | Needs stat computation + interpretation |
| Table_Selection | 17 | 1.06 | 1.00 | Multi-row filtering with counts |

These are harder tasks at this parameter class — both models struggle
regardless of architecture. The 2-5B tier is below the threshold for
reliable multi-step table reasoning of this kind.

## Tasks where both excel

- **Table_Fact_Checking** (n=12): T5G 7.5, G2 8.3 — both strong on
  binary entailment over table facts.
- **Table_Summary** (n=17): T5G 7.5, G2 7.0 — both write reasonable
  summaries (ROUGE-L naturally favors T5G here, judge is flatter).
- **Hallucination_Evaluation** (n=17): T5G 3.5, G2 3.5 — identical.

## What this changes vs the string-based comparison

The original string-metrics writeup reported:
- EM: T5G 10.5% vs G2 11.3%
- ROUGE-L: T5G 0.244 vs G2 0.231

The Sonnet judge reveals the string metrics were **understating both
models by ~3× on CORRECT rate.** Most of the "EM misses" were semantic
wins — answered "2.65" for gold "Final Answer: 2.65", wrote a plausible
different phrasing for Table_Summary, etc. When you let a capable judge
read the predictions, CORRECT rate goes from 10-11% → 30-32%.

The directional story (tie) is unchanged. The absolute numbers are much
higher once you drop strict string matching.

## Cost

- 20 subagent runs × ~25K tokens each ≈ 500K tokens total.
- Free from user's perspective (Claude Code's compute pool, no
  ANTHROPIC_API_KEY or OpenRouter credits needed).
- Wall-clock: ~2 min for the pilot batch + ~3 min for each wave (10
  agents parallel) = ~8 min end-to-end.

## Files

- `results/judged.jsonl` — predictions + per-sample judge output
- `insights/judge_sonnet_2026-04-23.md` — this writeup
- `insights/comparison_2026-04-22.md` — prior string-metrics comparison
