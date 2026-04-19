# Experiments

Evaluation experiments live here. This is **distinct from `models/`**, which holds
training/fine-tuning pipelines. Experiments *consume* trained models (ours or
third-party) and compare them on a benchmark.

## Layout

Each experiment is a self-contained directory under `experiments/<name>/` with
snake_case names:

```
experiments/<name>/
├── README.md        # Hypothesis, setup, how to run, status, TL;DR of findings
├── configs/         # One YAML per variant being compared
├── run.sh           # Entrypoint; accepts a config path arg to pick a variant
├── results/         # Raw eval outputs (predictions.jsonl, metrics.json) — gitignored
└── insights/        # Analysis markdown, plots, writeups — committed
```

### `README.md`

Lead with the hypothesis and end with the finding. Sections:

- **Hypothesis** — the one-sentence claim being tested.
- **Setup** — models compared, dataset, metrics, infra (pod type, wall-clock).
- **How to run** — copy-pasteable commands (`bash run.sh configs/<variant>.yaml`).
- **Status** — `planning` / `running` / `complete` / `abandoned`.
- **Findings** (fill in at end) — per-variant numbers + TL;DR + links to
  `insights/` for the deeper write-up.

### `configs/`

One YAML per variant. Keep the config flat and explicit — model id, dataset
slice, eval params, output path. Example:

```yaml
# configs/qwen_7b_instruct.yaml
model:
  name_or_path: Qwen/Qwen2.5-7B-Instruct
  dtype: bfloat16
dataset:
  name: JiuTian-AI/JIUTIAN-TReB
  split: test
eval:
  max_new_tokens: 512
  batch_size: 8
output:
  predictions_path: results/qwen_7b_instruct/predictions.jsonl
  metrics_path: results/qwen_7b_instruct/metrics.json
```

### `run.sh`

Entrypoint. Accept config as the first arg so the same script runs any variant:

```bash
#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/default.yaml}"
python eval.py --config "$CONFIG"
```

If the experiment runs on RunPod, mirror the `models/*/run.sh` pattern
(sshd + log tee + `CODE_REPO` git-pull + `DEV=1` to keep pod alive). See
`models/t5gemma-2-4b-sft/run.sh` for the reference implementation.

### `results/`

Raw machine-readable output. One subdir per variant:

```
results/
├── qwen_7b_instruct/
│   ├── predictions.jsonl
│   └── metrics.json
└── t5gemma_sft/
    ├── predictions.jsonl
    └── metrics.json
```

**Gitignored by default** — raw predictions can be GB-scale. Commit only the
`metrics.json` summary if it's small, or reference results from
`insights/` with per-task tables copied in.

### `insights/`

Human-readable analysis — **committed**. This is the durable output of the
experiment. What lives here:

- `findings.md` — the narrative: what we measured, what surprised us, what
  we're going to do next. Cross-link to specific predictions.jsonl rows
  when making a claim (e.g. "T5Gemma refused on 12% of SymbolicReasoning
  rows — see `results/t5gemma_sft/predictions.jsonl` lines 142, 387, …").
- `per_task.md` — per-TReB-task breakdown table.
- `plots/` — PNG/SVG plots; regenerate from `results/` via a script.
- `error_analysis.md` — failure-mode taxonomy with examples.

If you'd summarize the experiment to a colleague in 6 months using only these
files, you've written enough.

## Naming

- snake_case (matches existing `experiments/encoder_vs_decoder/`).
- Name by *what's being compared*, not by a date or a ticket id.
- Good: `encoder_vs_decoder`, `flan_vs_tableinstruct_sft`, `cot_prompt_ablation`.
- Bad: `experiment_1`, `2026_04_runs`, `jira_TRB_142`.

## Status lifecycle

Drive from the README.md `Status:` line:

- `planning` — README + configs stubbed, no runs yet. OK to commit.
- `running` — first run kicked off. Update `Findings` as numbers come in.
- `complete` — all variants run, findings written, insights committed.
- `abandoned` — leave the directory with a `Status: abandoned` and a one-line
  reason. Don't delete — future-you wants to know this was tried.

## Relationship to other parts of the repo

- **`models/`** trains/fine-tunes models → produces checkpoints on Hub.
  Experiments consume those.
- **`docs/progress.md`** is the project-wide narrative log. Log experiment
  launches/completions there; keep the *analysis* in `experiments/<name>/insights/`.
- **`docs/learnings.md`** captures generic gotchas. Experiment-specific
  findings stay inside the experiment; only lift something into `learnings.md`
  if it generalizes across experiments.
