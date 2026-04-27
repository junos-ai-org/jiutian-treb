# jiutian-treb

TReB (Table Reasoning Benchmark) — evaluates LLMs on 26 table reasoning tasks
across 6 capability categories. Paper: arxiv:2506.18421. Upstream: github.com/JT-LM/jiutian-treb.
Dataset: HuggingFace/ModelScope `JiuTian-AI/JIUTIAN-TReB`.

License: Apache 2.0. Python 3.9+.

## Remotes

- `origin` — our fork: git@github.com:junos-ai-org/jiutian-treb.git
- `upstream` — original repo: https://github.com/JT-LM/jiutian-treb.git

## Checkpoints

Session checkpoints capture conversation state, decisions, and next steps.
Load the latest checkpoint at the start of a new session to resume context.

- `~/.claude/projects/-Users-arunkumarchithanar-research-jiutian-treb/checkpoints/2026-04-16-session-02.json`
  Deep dive into T5Gemma v1 paper, PrefixLM vs UL2 analysis, cross-attention mechanics.
- `~/.claude/projects/-Users-arunkumarchithanar-research-jiutian-treb/checkpoints/2026-04-17-session-01.json`
  Initial research session: experiment planning, T5Gemma vs Qwen comparison, SFT setup.
- `~/.claude/projects/-Users-arunkumarchithanar-research-jiutian-treb/checkpoints/2026-04-25-session-01.json`
  Verified transformers PR #45540 (T5Gemma 2 long-input fix, 8/8 OK); stopped verification pod; recalled 1000-sample T5G v1 vs Gemma-2-2B-IT result (+4.9pp CORRECT).

## Knowledgebase

Detailed reference material lives in `~/.claude/knowledgebase/`:

- `treb/paper.kb` — TReB paper summary (26 tasks, 3 reasoning modes, results)
- `treb/t5gemma-v1-paper.kb` — T5Gemma v1 technical report (arxiv:2504.06225): SFT+RLHF pipeline, IT advantage findings
- `treb/t5gemma-sft.kb` — SFT training guide (Seq2SeqTrainer, QLoRA, RunPod setup)
- `mmtu/experiments.kb` — MMTU experiment framework reference

## Code layout

- `models/<name>/` — model training code (e.g. `models/t5gemma-2-4b-sft/` for SFT)
- `experiments/<name>/` — evaluation experiments (see convention below)

## Experiment convention

Evaluation experiments live under `experiments/<name>/` with snake_case names. Layout:

```
experiments/<name>/
├── README.md        # Hypothesis, setup, how to run, status, TL;DR of findings
├── configs/         # One YAML per variant (e.g. qwen_7b_instruct.yaml, t5gemma_sft.yaml)
├── run.sh           # Entrypoint; accepts a config path arg to pick the variant
├── results/         # Raw eval outputs (predictions.jsonl, metrics.json) — gitignored
└── insights/        # Analysis markdown, plots, writeups — committed
```

See `experiments/README.md` for the full spec and a minimal template.

## Logging

Bash commands and web queries are logged to `~/.claude/logs/treb/` via hooks in `.claude/settings.local.json`.
