# Pod verification smoke — 2026-04-22

**Pod:** `oqmsie3n5n4x6s` (EU-NL-1, H100 80GB HBM3)
**Image:** `achithanar/treb-eval-t5gemma:latest`
**Variant:** `t5gemma_v1_2b2b_ul2_it`
**Samples:** 20 stratified (from the 7018-sample filtered kept set)
**Duration:** ~8 minutes of generation + ~2 minutes model load

## Went well

- Existing `treb-eval-t5gemma` image works for v1 out of the box. `model_type:
  t5gemma` + transformers 5.5.4 + torch 2.8.0+cu128 → model loads in ~2 s
  after weight shards land, occupies ~11.6 GB of 80 GB GPU memory.
- Phase-0 filter held: p95 prompt = 2554 tokens (below the 4 K SWA threshold),
  no bug triggered on any sample.
- All 20 predictions written, sentinel + SHA matched, SCP verified.

## Blocker found: EOS repetition loop

The model produces a *correct-looking* first `{"answer": X}` JSON block,
then repeats that same block until `max_new_tokens=1024` is exhausted.
Example:

```
{"answer": "11011000_4"}
```json
{"answer": "11011000_4"}
```json
{"answer": "11011000_4"}
... [x 40 repetitions]
```

Root cause is upstream of us: v1 UL2-IT inherits the same "never learned to
stop" behavior the sibling v2 experiment already documented
([`report.tex` §1](../../t5gemma_vs_qwen_treb/insights/report.tex)).
We're also sending the raw prompt text through `tokenizer(...)` without
`apply_chat_template(...)` — likely missing `<start_of_turn>` / `<end_of_turn>`
framing the IT checkpoint expects.

Impact:
- Scoring **unaffected**: `score.py::extract_json_answer` grabs the first
  `{"answer": ...}` via regex and ignores trailing repetitions (17/20
  extractable).
- Wall-clock **severely affected**: ~30 s per sample due to generating 1024
  tokens of slop. At this rate a full 7018-sample run would take ~60 GPU-hours.
  A proper EOS stop should drop this to ~3 s/sample (~6 hrs total).

## Generation quality — eyeballed on first 5

| Task | Gold | Predicted |
|---|---|---|
| `Table_Query` | `From the data presented in the table, the preferred eatType is…` | `{"answer": "eatType"}` — likely misread question |
| `Table_General_Operations` | `Final Answer: 2.65` | `{"answer": 2.65}` — **match** (gold has prefix) |
| `Understanding` | `(A)Kevin` | `{"answer": "Kevin"}` — **match** (gold has option letter) |
| `Table_Summary` | `The statistic shows gross domestic product…` | reasonable multi-sentence summary of GDP |
| `Table_Title_Naming` | `The title is wybe` | `"List of Channels Broadcasting Specific Programming Types"` — plausible but mismatch |

Naive string exact-match says 1/20. This is misleading — real TReB scoring
uses ROUGE-L for text tasks, numeric-equivalence EM for numeric tasks, and
LLM-judge for semantic correctness. We'd expect the actual numbers to be
substantially higher once the scoring step runs.

## Recommended fixes before the full run

1. **Apply chat template** in `eval.py::run_hf_seq2seq` — wrap the TCoT
   prompt with `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`
   the same way `run_vllm` does at line ~210. This is the single highest-leverage
   fix; IT models expect turn markers.
2. **Add stop-string fallback**: pass `stop_strings=['}\n```']` (or similar
   pattern that matches the close of the first JSON block) as a belt-and-
   suspenders even if (1) works.
3. **Cap `max_new_tokens` to 256** once (1) lands. Enough for TCoT rationale
   + JSON answer; cuts wall-clock ~4×.
4. **Benchmark SWA-bug threshold on v1**: dedicated probe at 4000, 4094,
   5000 tokens. If v1 doesn't share v2's bug, we can relax the phase-0
   filter to ~7500 and recover all the Multi-step tasks.

## Cost

~35 min of pod uptime × ~$2.50/hr ≈ **$1.50** for this verification.
