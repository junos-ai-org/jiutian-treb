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

1. Write variant configs + new run.sh/run_all.sh in the experiment dir. **done**
2. New Docker images or rebuild existing with our configs baked in. **deferred
   — existing `achithanar/treb-eval-t5gemma:latest` verified to support v1.**
3. SSH into a RunPod, smoke-test T5Gemma v1. **partial — see below.**
4. Only after T5Gemma v1 full completes: launch Qwen3.

### 2026-04-22 (cont'd) — pod verification on `oqmsie3n5n4x6s` (EU-NL-1, H100 80GB)

Launched existing `achithanar/treb-eval-t5gemma:latest` + tar-streamed our new
experiment dir to `/workspace/code/`. **Had to manually inject HF_TOKEN** —
RunPod's pod-launch env didn't propagate HF_TOKEN to the shell even though
we passed it in the payload. Wrote to `/root/.hf_token` + exported via
`.bashrc`; `whoami` confirmed `akumch`.

**Smoke — 20 stratified samples, max prompt = 2554 tokens (all under the
4K SWA threshold):**

| Observation | Status |
|---|---|
| Model loads (5.6B bf16) | ✓ ~11.6GB GPU, 2s load time after weight shard fetch |
| Generation runs | ✓ at batch=1, ~30s/sample due to EOS repetition loop |
| Output format | ✓ valid `{"answer": X}` JSON on the first iteration |
| **EOS / stop behavior** | ✗ **model never emits EOS** — repeats the same JSON block until `max_new_tokens=1024` |

**EOS is the critical blocker.** This matches exactly what the sibling v2
experiment documented: see `experiments/t5gemma_vs_qwen_treb/insights/report.tex`
("Base T5Gemma 2 never learned to stop"). Apparently v1 UL2-IT inherits
the same pathology even after Google's SFT+RLHF — or the tokenizer / chat
template isn't being applied correctly by our vanilla `AutoTokenizer` call
+ plain `model.generate()`.

**Mitigations (next session):**
1. **Decode-time fix (cheap)**: add `stop_strings=['}']` or equivalent
   truncation in eval.py so generation stops after the first JSON
   closes. `score.py` already extracts the first valid JSON via regex
   (see `extract_json_answer`), so this only saves compute — scoring
   itself is unaffected.
2. **Chat template inspection**: the IT checkpoint likely expects
   `apply_chat_template(...)` wrapping (like Qwen's `<|im_start|>`
   turn markers). Our current `run_hf_seq2seq` sends the raw prompt text
   without chat template — Google's t5gemma-2b-2b-ul2-it expects some
   `<start_of_turn>` / `<end_of_turn>` framing. Fix once and all IT runs
   will generate cleanly + short outputs → ~10× faster wall-clock.
3. If chat-template fix yields clean single-JSON outputs, revert to
   max_new_tokens ~256 (from 1024) to dramatically speed up full run.

**Pod handling**: 20-sample smoke completed. All 20 predictions SCP'd to
`experiments/t5gemmav1_vs_qwen3_treb/results/t5gemma_v1_2b2b_ul2_it/` with
sentinel + SHA verified. Pod terminated (HTTP 204). ~$1.50 spent.

**Generation quality eyeball (first 5)**: model produces *correct-looking*
answers but with format drift from gold (e.g. `{"answer": 2.65}` vs
gold `Final Answer: 2.65`). Naive EM is 1/20, but this dramatically
undercounts real quality — proper scoring (ROUGE / numeric-EM /
LLM-judge) will give a fairer read. 17/20 predictions extract a valid
`{"answer": ...}` JSON via `score.py::extract_json_answer`. Full
writeup: `experiments/t5gemmav1_vs_qwen3_treb/insights/verification_smoke_2026-04-22.md`.

**SWA bug status for v1**: not triggered on this smoke (max prompt was
2554 — below the 4094 threshold). Needs dedicated probe at >4000 tokens
to know if v1 shares the bug. Low priority since we're pre-filtering anyway.

### Blocked: do not launch full run or Qwen3 yet

Three fixes needed before the full T5Gemma v1 run is worth doing:

1. **Apply chat template** in `eval.py::run_hf_seq2seq`. Currently the
   HF-seq2seq path sends raw prompt text; the vLLM path already calls
   `apply_chat_template`. IT checkpoints expect turn markers
   (`<start_of_turn>` / `<end_of_turn>`). This is likely the fix that
   makes EOS fire correctly.
2. **Add `stop_strings=['}```']`** (or similar close-of-JSON pattern) as
   belt-and-suspenders.
3. **Drop `max_new_tokens` from 1024 → 256** after (1)(2) land. Enough for
   TCoT rationale + JSON; cuts wall-clock ~4×.

Once those three ship → smoke again → if clean, launch full T5Gemma v1
(~6 GPU-hours estimated post-fix, vs the 60-hour slog we'd have today) →
only then launch Qwen3. Same-pod lifecycle handled by `monitor_pod.sh`.

### 2026-04-23 — EOS fix lands, scope pivot, comparison done

**EOS fix verified** (commit `d5554bd`): 25-sample smoke went from
4500-char repetition slop → 503-char clean single JSON response. 10×
shorter, 10× faster, zero repetition on 248 samples. Scope dropped from
7018 full to 250 stratified per user direction.

**Pivot from Qwen3-4B to Gemma-2-2B-IT comparator** (commit `<next>`):

1. OpenRouter has no dense Qwen3 <7B. Qwen3-8B is too big for a param-match.
2. User's OpenRouter account had 0 credits → Qwen3-8B run errored at sample 49.
3. RunPod EU-NL-1 Docker Hub egress was broken all afternoon — 3 consecutive
   pods stuck for 10+ min at image pull.
4. Gemma-2-2B-IT is a **cleaner comparison** than Qwen3 anyway: same
   Gemma-2 pretraining base as T5Gemma v1, same chat template, same tokenizer,
   same era SFT/RLHF. Only axis differing is enc-dec vs dec-only.

**Comparison on 248 stratified samples (English TCoT, 4K-token gated):**

| Variant | Params | Arch | EM | ROUGE-L |
|---|---:|---|---:|---:|
| T5Gemma v1 2B-2B UL2-IT | 5.6B | enc-dec | 0.105 | **0.244** |
| Gemma-2-2B-IT | 2.6B | dec-only | **0.113** | 0.231 |

**Directional finding: tie.** Doubling params via enc-dec structure doesn't
measurably shift TReB/TCoT when you control for pretraining lineage.
Decoder-only is the per-param winner. Full per-task table in
`experiments/t5gemmav1_vs_qwen3_treb/insights/comparison_2026-04-22.md`.

Both pods SCP'd + terminated. Total pod spend ~$2.

### Open

- LLM-as-judge pass (DeepSeek-V3 via OpenRouter, ~$1) for a more sensible
  score than EM / ROUGE-L on text-generation tasks. Waiting on credits.
- Qwen3-8B retry when OpenRouter credits available (49/248 valid today).
- Gemma-2-9B-IT as a param-equal decoder-only if we want a second data point.

### 2026-04-23 — EOS fix verified + re-scoped run

Per user direction, **full 7018 run de-scoped**. New tiered plan: 25-sample
EOS-fix verification → 250-sample T5Gemma v1 → 250-sample Qwen3 (same
stratified subset) → compare.

Rationale: 250 stratified samples is enough to see whether the two models
are in the same ballpark; 7018 would be ~15× the cost to buy little extra
signal before we know scoring is calibrated.

**EOS fix landed** (commit `d5554bd`): `run_hf_seq2seq` in `eval.py` now
applies `tokenizer.apply_chat_template(...)` AND passes
`eos_token_id=[1, 107]` (the regular `<eos>` + the `<end_of_turn>` marker
that Gemma IT models actually use). Also drops `max_new_tokens` 1024→256.

**Pod 2 `4q5g2hyy50vrq7` (EU-NL-1, H100 80GB)** — relaunched image, streamed
new code, injected HF_TOKEN same as pod 1.

**25-sample EOS-fix smoke (actually 24 after 4-bin stratification) —
completed in ~3 min:**

| Metric | Pre-fix (pod 1) | Post-fix (pod 2) |
|---|---|---|
| Avg prediction char length | ~4500 | **503** (↓ 10×) |
| Wall-clock per sample (batch=1) | ~30 s | **~3 s** (↓ 10×) |
| Valid-JSON extraction rate | 17/20 (85%) | 20/24 (83%) |
| Repetition loop | yes, to `max_new_tokens` | **none** |
| Exact-match hits | 1/20 (`Kevin` vs `(A)Kevin`) | 1/24 (`Entailment` exact) |

Example post-fix output (Table_Title_Naming):
```
The table lists channels, their video aspect ratio, and the programming
they broadcast.

Therefore, a suitable title for the table would be: **"List of Channels
and Their Programming"**.

```json
{"answer": "List of Channels and Their Programming"}
```
```

Clean reasoning, single JSON block, model stopped on its own. This is
what the scoring layer (ROUGE/numeric/LLM-judge) was designed for.

**250-sample T5Gemma v1 run: in progress** on the same pod. p50 prompt
length 237 tokens, max 3791 — well under the 4K filter cap. At ~3 s/sample
× 248 samples ≈ 12 min wall-clock. Will SCP + terminate pod on sentinel.

**Next**: launch Qwen3-4B pod (may need `pip install --upgrade vllm>=0.8`
on pod; the baked image pins 0.6.3 which predates Qwen3 support). Same
stratified 250 samples — the stratifier uses char-length and a fixed
Random(42) seed, so both variants will see exactly the same subset.
