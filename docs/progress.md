# Progress Log

Narrative journal for the encoder-vs-decoder TReB experiment. Append newest entries at the top. One entry per working session or meaningful milestone.

**Entry format**
```
## YYYY-MM-DD — short title
**Status**: planning | in-progress | blocked | done
**What happened**: ...
**Decisions**: ...
**Next**: ...
**Artifacts**: paths, run IDs, wandb links, checkpoint files
```

---

## 2026-04-18 — SFT scaffold created
**Status**: in-progress
**What happened**: Scaffolded `models/t5gemma-2-4b-sft/`: `configs/sft_flan.yaml` (all hyperparams), `prepare_data.py` (FLAN subset → tokenized on `/workspace`), `train.py` (Seq2SeqTrainer + QLoRA, `push_to_hub` streaming to `DiffusionTableQA/t5gemma-2-4b-flan-sft`), `merge_adapter.py`, slim `Dockerfile` (code+deps only, no data), `requirements.txt`, `.env.example`, `README.md`. FLAN + base weights are cached to the RunPod network volume at runtime rather than baked into the image.
**Decisions**: 100K-sample FLAN subset (not full 15M). `eager` attn, `SEQ_2_SEQ_LM` LoRA, `hub_strategy=checkpoint` for resilient checkpointing. `.env` already in `.gitignore`.
**Next**: Build Docker image on AWS build server (git clone → `docker build -t achithanar/t5gemma-sft:latest .` → `docker push`), then launch smoke pod on RunPod, verify non-zero loss.
**Artifacts**: `models/t5gemma-2-4b-sft/{configs/sft_flan.yaml, prepare_data.py, train.py, merge_adapter.py, Dockerfile, requirements.txt, README.md, .env.example}`

---

## 2026-04-18 — HF push capability verified
**Status**: done
**What happened**: Tested HF Hub push with a fine-grained token. `whoami` → user `akumch`. Created a private test repo, uploaded a file, deleted it — all succeeded. HF orgs available: `bayarea-big-cats`, `DiffusionTableQA`. Note: `junos-ai-org` exists on GitHub but not on HF.
**Decisions**: Push destination = `DiffusionTableQA/t5gemma-2-4b-flan-sft` (org write scope verified with create/upload/delete test). Token will be rotated (was pasted in chat) and stored in a gitignored `.env` or pod env var.
**Next**: Rotate token, wire `DiffusionTableQA/t5gemma-2-4b-flan-sft` into training config, scaffold SFT code.

---

## 2026-04-18 — Progress log established
**Status**: planning
**What happened**: Loaded checkpoint `2026-04-17-session-01`. Confirmed SFT hyperparameters for T5Gemma 2 4B-4B (QLoRA, Seq2SeqTrainer, LR 2e-4, bs 32, bf16, eager attn). Noted that `CLAUDE.md` references bash/web hooks logging to `~/.claude/logs/treb/` but the hooks are not actually wired in `.claude/settings.local.json` (logs dir is empty).
**Decisions**: Track progress in-repo via this file rather than only in session checkpoints. Hooks-based command logging deferred.
**Next**: Scaffold `models/t5gemma-2-4b-sft/` (training script, config, Dockerfile) and `experiments/encoder_vs_decoder/`.
**Artifacts**:
- Checkpoint: `~/.claude/projects/-Users-arunkumarchithanar-research-jiutian-treb/checkpoints/2026-04-17-session-01.json`
- KB: `~/.claude/knowledgebase/treb/{paper,t5gemma-sft,t5gemma-v1-paper}.kb`
