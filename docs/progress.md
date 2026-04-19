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

## 2026-04-19 — DDP restart-loop bug + eval harness scaffolded
**Status**: in-progress
**What happened**:
- Confirmed DDP run 1 finished cleanly and pushed final adapter to Hub (`DiffusionTableQA/t5gemma-2-4b-flan-sft-ddp` commit `abae90a8` at 03:39:46).
- **Restart-loop bug**: pod's container exited with 0 after `run.sh` finished; RunPod's default restart policy re-ran it twice more before I noticed. Runs 2 and 3 also completed (commits `429bcef8` at 05:16, `18075f97` at 06:51). User terminated pod `0rcddmeaunsklh` via RunPod API at ~07:46. Wasted GPU cost: ~$68 (2 × 1.5 h × $11.96/hr).
- Fix shipped: `models/t5gemma-2-4b-sft/run.sh` now defaults to `exec tail -f /dev/null` after training; set `AUTO_TERMINATE=1` to exit instead.
- Scaffolded `experiments/t5gemma_vs_qwen_treb/` (README + 3 configs + eval.py + score.py + run.sh) and two Docker images (`docker/t5gemma-eval/`, `docker/qwen-eval/`) with a matrix GHA workflow. Adapter pinned to run-1 `abae90a8` in `t5gemma_sft.yaml`.
- Persisted `OPENROUTER_API_KEY` + `HF_TOKEN` to `~/.claude/.env` (mode 600) alongside RUNPOD key. Score path uses DeepSeek V3 via OpenRouter for LLM-as-Judge (~$5–10 for full run vs ~$100 hosting Qwen2-72B).

**Decisions**:
- Eval first pass: English only (3,895 samples), TCoT mode only, 4× H100, three variants in parallel. Defer PoT/ICoT and Chinese.
- Two Dockerfiles (vLLM for Qwen / transformers v5 + peft for T5Gemma) — version conflict is real.
- Pin SFT adapter to `abae90a8` (run-1 final) for clean provenance even though run-3 is quality-equivalent.

**Next**:
- Push to trigger GHA image builds.
- Attach volume `l6pnvotcgk` (US-WA-1, has base model cached) to an eval pod.
- Flash-attn investigation on T5Gemma 2 (current learnings say eager required; retest under v5).
- Smoke test 100 samples per variant.

**Artifacts**:
- `experiments/t5gemma_vs_qwen_treb/{README.md,eval.py,score.py,run.sh,configs/*.yaml}`
- `docker/{t5gemma-eval,qwen-eval}/{Dockerfile,requirements.txt}`
- `.github/workflows/build-eval-images.yml`
- Hub: `DiffusionTableQA/t5gemma-2-4b-flan-sft-ddp@abae90a8` (SFT adapter)
- Secrets: `~/.claude/.env` now has RUNPOD_API_KEY, OPENROUTER_API_KEY, HF_TOKEN

---

## 2026-04-19 — Full runs launched (A100 + 4× H100 DDP in parallel)
**Status**: in-progress
**What happened**: Launched two parallel full-FLAN SFT runs — an A100 baseline and an aggressive 4× H100 DDP variant. Both writing to the same wandb project.

**A100 run** (pod `igd5cee414zi5c`, US-WA-1, $1.49/hr):
- Config `configs/sft_flan.yaml`: batch=8, grad_accum=4 (effective=32), max_in=1024, max_tgt=384, LR=2e-4, cosine, warmup=0.03, `predict_with_generate=false`.
- Revised wall-clock measurement: **12.3 s/step** — my original 3 h estimate was way off. Real ETA ~10.7 h, cost ~$15.95. Cause: T5Gemma 2 is encoder-decoder; each step runs both encoder AND decoder forwards, ~2× compute vs a same-size decoder-only.
- Hub target: `DiffusionTableQA/t5gemma-2-4b-flan-sft`
- Wandb: [l551vomo](https://wandb.ai/junos/treb-encoder-vs-decoder/runs/l551vomo)

**DDP run** (pod `0rcddmeaunsklh`, EU-NL-1, $11.96/hr for 4× H100):
- Config `configs/sft_flan_ddp.yaml`: batch=8 per-GPU, grad_accum=4, 4 GPUs → effective=128, LR=4e-4 (sqrt scaling), warmup=0.05.
- Launcher: `torchrun --standalone --nproc_per_node=4 train.py`. HF Trainer auto-detected DDP; no other code changes.
- Measured: **7.8 s/step** → ETA ~1.7 h, cost ~$20.33. ~6× faster than the A100 for comparable total spend.
- Hub target: `DiffusionTableQA/t5gemma-2-4b-flan-sft-ddp` (separate repo so the two runs don't step on each other).
- Wandb: [7s7i24cw](https://wandb.ai/junos/treb-encoder-vs-decoder/runs/7s7i24cw)

Path to launch the DDP: after deciding on effective=128, needed to create a 200 GB volume in an enum-accessible DC. US-MO-1 (Medium H100 stock) is rejected by `POST /pods` despite being valid for `POST /networkvolumes` — had to delete that volume and create one in EU-NL-1 instead. Captured this in `docs/learnings.md` and the `runpod-training-pipeline` skill.

**Decisions**: Keep both runs for safety + direct comparison of effective batch 32 vs 128 on this model/data.
**Next**: Wait ~1.7 h for DDP to finish. If clean, merge adapter and push merged model. A100 stays as an overnight backup.
**Artifacts**:
- US-WA-1 volume: `l6pnvotcgk` (has cached base model from earlier runs)
- EU-NL-1 volume: `8uj8ouz5es` (fresh; re-downloads base and tokenizes on first boot)
- Image: `achithanar/t5gemma-sft:6d01d94` (has NUM_GPUS/torchrun launcher + CODE_REPO + sshd + log-to-volume)

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
