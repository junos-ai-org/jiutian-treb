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

## 2026-04-19 — Smoke test PASSED; pipeline end-to-end works
**Status**: done
**What happened**: Smoke on the first real run (pod `byupl3n709221y`, image `c1de6d9`) finished all 50 steps and saved the adapter — we didn't know at the time because logs weren't tee'd to the volume. Confirmed by SSHing into a fresh inspect pod (`849tj18meu7xll`) and reading `/workspace/checkpoints/t5gemma-2-4b-flan-sft-smoke/checkpoint-50/trainer_state.json`.

Loss curve (50 steps, 200 FLAN samples, QLoRA r=16):
| step | train loss | eval loss |
|-----:|-----------:|----------:|
| 5    | 7.93       | —         |
| 25   | 0.96       | 0.78      |
| 50   | 0.49       | 0.32      |

Adapter `adapter_model.safetensors` is 273 MB with LoRA targets: `q/k/v/o_proj`, `fc1/fc2`, `gate/up/down_proj`, `self_attn.out_proj`. 68.3M trainable / 7.58B total = 0.90%.

Dev-loop improvements deployed in image `c1de6d9`:
- **run.sh** now generates sshd host keys (`ssh-keygen -A`), starts sshd, injects `$PUBLIC_KEY`, tees all output to `/workspace/logs/<timestamp>.log` (survives container death), supports `CODE_REPO`/`CODE_REF` for git-on-volume iteration (skip image rebuild for code-only changes), and `DEV=1` to keep the pod alive after training.
- Now able to SSH directly from this Claude container via `/root/.ssh/runpod_key` — no more "paste the logs" round-trips.
- Previous painful bugs captured: `datasets.load_dataset` pulls all 2157 shards unless `streaming=True`; `DataCollatorForSeq2Seq(model=...)` crashes on T5Gemma2 because `prepare_decoder_input_ids_from_labels` has a bad signature (drop `model=` to let the model shift labels internally); base image has no SSH host keys; `/workspace` mount shadows code, so code lives at `/opt/sft`.

**Decisions**: Smoke validated; moving to full FLAN SFT.
**Next**: Launch full run with `configs/sft_flan.yaml` (100K samples, 1 epoch, `push_to_hub=true` → `DiffusionTableQA/t5gemma-2-4b-flan-sft`). Estimated ~4–6 hours at $1.49/hr on A100 SXM 80GB.
**Artifacts**:
- Volume `l6pnvotcgk` holds: `hf-cache/` (T5Gemma 2 base weights cached ~17 GB), `flan-tokenized-smoke/` (3.7 MB), `checkpoints/t5gemma-2-4b-flan-sft-smoke/` with adapter + checkpoint-50, `logs/20260419-011045.log`.
- Image: `achithanar/t5gemma-sft:c1de6d9`

---

## 2026-04-18 — Smoke pod launched
**Status**: in-progress
**What happened**:
- Refactored image so code lives at `/opt/sft` (not `/workspace`) because the RunPod network volume mounts at `/workspace` and would shadow it. Added `run.sh` entrypoint that runs `prepare_data.py` once (skips if tokenized dir exists) then `train.py`. Added `configs/sft_flan_smoke.yaml` (200 samples, 50 steps, no hub push, no wandb).
- RunPod setup: created 200 GB network volume `l6pnvotcgk` in US-WA-1. A100 SXM 80GB available at $1.39/hr (Low stock; only US-KS-2, US-MO-1, US-WA-1 have it).
- Launched pod `gn2j7nxbog1kca` with image `achithanar/t5gemma-sft:14785c8`, volume mounted at `/workspace`, env `CONFIG=configs/sft_flan_smoke.yaml`, `HF_TOKEN` and `WANDB_API_KEY` set (wandb unused in smoke).
- Pod RUNNING, SSH on `195.26.233.55:32642`. GPU utilization 0% — data/model download phase.
**Decisions**: Ran smoke before full to validate pod boot, GPU access, gated-model download, and non-zero loss with minimal spend.
**Next**: Monitor GPU util; once training starts verify non-zero loss. Then launch full run (100K FLAN, 1 epoch, push_to_hub enabled).
**Artifacts**:
- Pod console: https://www.runpod.io/console/pods/gn2j7nxbog1kca
- Volume: `l6pnvotcgk` (200 GB, US-WA-1)
- Image: `achithanar/t5gemma-sft:14785c8`

---

## 2026-04-18 — First successful image build
**Status**: done
**What happened**: Installed `gh` CLI in the Claude container, authenticated via device flow as `arunkumarchithanar`. First workflow run failed because the Dockerfile's base image tag `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu` doesn't exist — RunPod moved to the pattern `<rev>-cu<cuda>-torch<torchver>-ubuntu<ubuntuver>`. Switched to `runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404` (CUDA 12.8.1, torch 2.8.0, Ubuntu 24.04). Push auto-triggered the workflow; build succeeded in ~3 min.
**Decisions**: Use RunPod's current tag scheme; pin `cu1281-torch280-ubuntu2404` for reproducibility.
**Next**: Launch RunPod A100 pod with `achithanar/t5gemma-sft:56e594c`, mount network volume, run `prepare_data.py` then smoke `train.py` (tiny subset) to verify non-zero loss before committing to full FLAN SFT.
**Artifacts**:
- Docker Hub: `achithanar/t5gemma-sft:latest` + `:56e594c` (10.3 GB)
- Workflow run: https://github.com/junos-ai-org/jiutian-treb/actions/runs/24615161282

---

## 2026-04-18 — GitHub Actions image build
**Status**: in-progress
**What happened**: Added `.github/workflows/build-sft-image.yml` — builds the SFT image on Ubuntu runners via `docker/build-push-action@v6` with GHA buildx cache, pushes to `achithanar/t5gemma-sft`. Triggers on pushes to `experiment-setup`/`main` touching `models/t5gemma-2-4b-sft/**`, plus `workflow_dispatch` with an optional `extra_tag` input. Tags: `<short-sha>` + `:latest` (on branch push) + optional extra. Kept `build.sh` as a local fallback.
**Decisions**: CI-based builds primary; AWS build server secondary. Secrets required: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
**Next**: Add Docker Hub secrets to the repo (Settings → Secrets → Actions). Confirm first workflow run succeeds and the image lands on Docker Hub.
**Artifacts**: `.github/workflows/build-sft-image.yml`

---

## 2026-04-18 — Code review + v5 correctness fixes
**Status**: in-progress
**What happened**: Audited the SFT scaffold against live HF Hub + transformers v5 docs. Found and fixed:
- **Wrong model ID**: `google/t5gemma-2-4b-4b-ul2` doesn't exist. v2 dropped the `-ul2` suffix — actual repo is `google/t5gemma-2-4b-4b` (gated, uploaded 2025-12-18). The `-ul2`/`-prefixlm`/`-it` suffixes only exist on v1.
- **v5 Trainer API**: `Trainer(..., tokenizer=...)` is a hard `TypeError` in v5 — renamed to `processing_class=`. Fixed.
- **v5 from_pretrained API**: `torch_dtype=` deprecated in favor of `dtype=`. Fixed in `train.py` and `merge_adapter.py`.
- **Missing `device_map`**: bnb 4-bit on a single GPU requires `device_map="auto"` — without it, weights stay on CPU and forward fails (bnb 4-bit kernels are CUDA-only). Added.
- **Encoder-decoder grad flow**: added defensive `model.enable_input_require_grads()` after `prepare_model_for_kbit_training` — peft handles it in the kbit+reentrant path, but the explicit call is idempotent and safer.

Verified (no change needed): `transformers>=5.0.0` pin is correct (t5gemma2 support landed in v5.0.0 via PR #41834, Dec 2025). `eval_strategy`, `hub_token`, `hub_private_repo`, `BitsAndBytesConfig(load_in_4bit, nf4, double_quant, compute_dtype=bf16)`, `Open-Orca/FLAN` columns `inputs`/`targets` — all OK.
**Decisions**: Kept `transformers>=5.0.0` pin. Kept `report_to: [wandb]`, `hub_strategy: checkpoint`.
**Next**: Commit + push fixes. Then build image on AWS build server and smoke test.
**Artifacts**: `models/t5gemma-2-4b-sft/{configs/sft_flan.yaml, train.py, merge_adapter.py}` updated.

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
