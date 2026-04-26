# T5Gemma vs Qwen on TReB

**Status:** `planning` (scaffolded, no runs yet)

## Hypothesis

Does T5Gemma 2 4B-4B's **encoder-decoder architecture** carry any table-reasoning
advantage over a similarly-sized decoder-only model (Qwen 2.5 7B Instruct),
once we've closed the instruction-tuning gap by SFT'ing T5Gemma on FLAN?

Three variants, same benchmark:

| Variant | Model | Instruction-tuned? | Expected role |
|---|---|---|---|
| `t5gemma_base` | `google/t5gemma-2-4b-4b` | No (Google only shipped pretrained) | Zero-floor baseline |
| `t5gemma_sft` | `t5gemma-2-4b-4b` + FLAN-SFT adapter | Yes (our QLoRA, 100K FLAN) | Our candidate |
| `qwen_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | Yes (heavily, by Qwen team) | Strong decoder-only baseline |

T5Gemma 2 has 7.58B total params (4B enc + 4B dec, tied embed) — so Qwen 7B is
the fair total-params comparison.

## Setup

### Dataset

`JT-LM/JIUTIAN-TReB` on HuggingFace. Structure: `{Chinese,English}/<Task>.json`
+ shared `CSV/CSV.zip` of tables. 7,790 samples total across 26 tasks.

**First pass: English only** (3,895 samples). Drops the language-coverage
confound that would otherwise benefit Qwen (bilingual) over T5Gemma 2 (unknown
Chinese coverage).

### Reasoning modes

TReB supports three: **TCoT** (textual chain-of-thought, JSON output), **PoT**
(program-of-thought, executes Python), **ICoT** (iterative, up to 4 rounds).

**First pass: TCoT only.** PoT needs a Python sandbox; ICoT needs a multi-turn
loop. Both are interesting follow-ups; neither is needed for the core
architecture comparison.

### Metrics

- **ROUGE-L** — text tasks
- **Exact Match** — numeric tasks (with numeric equivalence, e.g., 42.0 == 42)
- **LLM-as-Judge** — DeepSeek V3 via OpenRouter (≈$5–10 for 3,895 × 3 judgments).
  Cheaper and stronger than hosting Qwen2-72B locally.

### Infrastructure

Two pods, each runs smoke-then-full for its variants. vLLM pins
`transformers<5` while T5Gemma 2 needs `>=5.0` — version conflict forces
two images, which forces two pods.

| Pod | Image | GPUs | Variants | Duration (est.) |
|---|---|---|---|---|
| **qwen** | `achithanar/treb-eval-qwen` | 1× H100 | `qwen` | 15–30 min (vLLM fast) |
| **t5gemma** | `achithanar/treb-eval-t5gemma` | 2× H100 | `t5gemma_base`, `t5gemma_sft` (parallel, GPU 0 + GPU 1) | 60–120 min (HF + enc-dec) |

Both pods idle on success (`AUTO_TERMINATE=0`) so we can `scp` results off
the network volume. Each pod's `run.sh` invokes `run_all.sh`, which reads
`$VARIANTS` (set per-image in the Dockerfile) and runs smoke first, only
proceeds to full if smoke passes, and writes per-variant logs to
`/workspace/logs/{smoke,full}_<variant>.log`.

### Pinning

| Variant | Pin |
|---|---|
| Base | `google/t5gemma-2-4b-4b` (always-latest — it's a pretrained reference) |
| SFT | `DiffusionTableQA/t5gemma-2-4b-flan-sft-ddp` @ commit `abae90a8` (run-1 final from 2026-04-19 03:39:46) |
| Qwen | `Qwen/Qwen2.5-7B-Instruct` (latest revision — matches what anyone would download today) |

## How to run

On each pod (CMD calls `run.sh` → `run_all.sh`):

```bash
# Default: smoke-then-full for all variants in $VARIANTS (set per-image).
bash run_all.sh

# Skip smoke (if previously validated):
SKIP_SMOKE=1 bash run_all.sh

# Single variant (bypass orchestrator):
RUN_ALL=0 CONFIG=configs/qwen_7b_instruct.yaml SMOKE=1 bash run.sh
```

Retrieving results:

```bash
# scp from pod — /workspace persists on the network volume.
scp -i ~/.ssh/runpod_key -P <port> -r \
    root@<pod-ip>:/workspace/results ./experiments/t5gemma_vs_qwen_treb/results/
scp -i ~/.ssh/runpod_key -P <port> -r \
    root@<pod-ip>:/workspace/logs ./experiments/t5gemma_vs_qwen_treb/logs/
```

Scoring happens locally after retrieval (for all three variants):

```bash
for v in qwen_7b_instruct t5gemma_base t5gemma_sft; do
  python score.py \
    --predictions results/$v/predictions.jsonl \
    --judge deepseek-v3 \
    --wandb
done
# Reads OPENROUTER_API_KEY and WANDB_API_KEY from ~/.claude/.env.
```

The judge returns a structured failure mode per sample (one of `CORRECT`,
`PARTIAL`, `FORMAT`, `HALLUCINATION`, `WRONG_SELECTION`, `NUMERIC`, `REASONING`,
`REFUSAL`, `EMPTY`, `MISUNDERSTAND`, `TRUNCATED`, `OTHER`) so per-variant
failure-mode distributions are aggregated in `metrics.json` and logged as
scalars to wandb.
The `--wandb` flag also logs a `predictions` wandb.Table with per-sample rows
for filtering and cross-variant comparison in the wandb UI. Project:
`treb-encoder-vs-decoder-eval`.

## Findings

*(to be filled in after runs)*

## Plan status

- [x] Scaffold directory + configs
- [x] Fix SFT pod's restart-loop bug (`models/t5gemma-2-4b-sft/run.sh`)
- [ ] Write `eval.py` + `score.py`
- [ ] Write two Dockerfiles + GHA workflows
- [ ] Investigate flash-attn on T5Gemma 2
- [ ] Smoke test (100 samples, ~$1.35)
- [ ] Full run (3,895 samples × 3 models on 4× H100, ~$18–28)
- [ ] Analysis write-up in `insights/`
- [ ] Verify [transformers#45540](https://github.com/huggingface/transformers/pull/45540) unblocks long-context T5Gemma 2 (see `insights/verify_pr45540.{py,sh}`)

## Verifying transformers#45540 (T5Gemma 2 long-input bug fix)

We filed [transformers#45521](https://github.com/huggingface/transformers/issues/45521) — T5Gemma 2 decoder
self-attention crashes at batch=1 once inputs exceed ~4094 tokens because
the cross-attention cache was incorrectly built as a sliding-window cache.
PR [#45540](https://github.com/huggingface/transformers/pull/45540) (APPROVED, not yet merged
as of 2026-04-25) fixes it; we need to confirm before re-running the
T5Gemma variants without the 4K cap.

`insights/verify_pr45540.py` runs the exact failure-length sweep from the
bug report; `insights/verify_pr45540.sh` wraps it in a control/treatment
harness that pip-installs the PR branch in between.

```bash
# On a t5gemma-eval pod (H100/A100, ~40 GB VRAM, HF auth for the gated model):
ssh -i ~/.ssh/runpod_key -p <port> root@<pod-ip>
cd /opt/eval                         # baked-in path; or /workspace/code if CODE_REPO was set
bash insights/verify_pr45540.sh      # runs control + treatment, ~6 min on H100
# scp the log back:
scp -i ~/.ssh/runpod_key -P <port> \
    root@<pod-ip>:/workspace/logs/verify_pr45540_*.log \
    experiments/t5gemma_vs_qwen_treb/insights/
```

Pass criterion: in the treatment phase, **zero failures at length > 4094**
across all eight test lengths (3525, 5015, 6499, 7493, 9997, 14967, 19808,
25135). If it passes, post the log summary to issue #45521 and bump the
T5Gemma configs' `max_input_tokens` from `4094` back to `32768`.
