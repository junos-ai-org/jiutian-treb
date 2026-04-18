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

## Knowledgebase

Detailed reference material lives in `~/.claude/knowledgebase/`:

- `treb/paper.kb` — TReB paper summary (26 tasks, 3 reasoning modes, results)
- `treb/t5gemma-v1-paper.kb` — T5Gemma v1 technical report (arxiv:2504.06225): SFT+RLHF pipeline, IT advantage findings
- `treb/t5gemma-sft.kb` — SFT training guide (Seq2SeqTrainer, QLoRA, RunPod setup)
- `mmtu/experiments.kb` — MMTU experiment framework reference

## Current Experiment: Encoder vs Decoder on TReB

**Goal**: Compare T5Gemma 2 4B-4B (encoder-decoder) vs Qwen-Instruct (decoder-only) on table reasoning.

**Status**: Research deep-dive complete. Next: scaffold code and run SFT.

**Key decisions**:
- SFT T5Gemma 2 on FLAN only (no TableInstruct — fair comparison)
- Use Seq2SeqTrainer + QLoRA (SFTTrainer doesn't support encoder-decoder)
- RunPod A100 80GB for training (~$8-9)
- `models/t5gemma-2-4b-sft/` for SFT code, `experiments/encoder_vs_decoder/` for benchmark

## Logging

Bash commands and web queries are logged to `~/.claude/logs/treb/` via hooks in `.claude/settings.local.json`.
