#!/usr/bin/env bash
# Pod entrypoint for TReB eval runs. Mirrors models/t5gemma-2-4b-sft/run.sh.
#
# Env vars:
#   CONFIG          — path to YAML config (e.g. configs/qwen_7b_instruct.yaml)
#   SMOKE           — if set to 1, pass --smoke to eval.py (100 samples)
#   AUTO_TERMINATE  — default 0; set 1 to exit the container after eval.
#                     Default idles so RunPod doesn't restart-loop.
#   CODE_REPO / CODE_REF — optional; git-clone into /workspace/code and run
#                     from there instead of the baked-in image code.
set -euo pipefail

# --- SSH bring-up (single-tenant pod, fine to stash secrets in .bashrc) ---
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
if command -v sshd >/dev/null 2>&1; then
  mkdir -p /run/sshd
  ssh-keygen -A >/dev/null 2>&1 || true
  /usr/sbin/sshd
  echo "[run.sh] sshd started"
fi

# --- Persist pod env to .bashrc so SSH sessions inherit it ---------------
{
  echo "# Injected by run.sh at $(date -Iseconds)"
  for var in HF_TOKEN WANDB_API_KEY WANDB_PROJECT OPENROUTER_API_KEY \
             HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE \
             HF_HUB_ENABLE_HF_TRANSFER HF_XET_HIGH_PERFORMANCE; do
    val="${!var:-}"
    if [[ -n "$val" ]]; then
      printf 'export %s=%q\n' "$var" "$val"
    fi
  done
} >> /root/.bashrc
echo "[run.sh] pod env persisted to /root/.bashrc"

# --- Log to volume so it survives container exits -------------------------
mkdir -p /workspace/logs
LOG=/workspace/logs/$(date +%Y%m%d-%H%M%S).log
echo "[run.sh] logging to $LOG"
exec > >(tee -a "$LOG") 2>&1

# --- Optional: run from a git checkout on the volume ---------------------
CODE_DIR=/opt/eval
if [[ -n "${CODE_REPO:-}" ]]; then
  CODE_DIR=/workspace/code
  if [[ ! -d "$CODE_DIR/.git" ]]; then
    echo "[run.sh] cloning $CODE_REPO -> $CODE_DIR"
    git clone "$CODE_REPO" "$CODE_DIR"
  fi
  cd "$CODE_DIR"
  git fetch --all --quiet
  git checkout "${CODE_REF:-main}"
  git pull --ff-only
  CODE_DIR="$CODE_DIR/experiments/t5gemma_vs_qwen_treb"
fi
cd "$CODE_DIR"
echo "[run.sh] code dir: $(pwd)  ($(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"

# --- Eval -----------------------------------------------------------------
# Default: orchestrate all three variants (smoke-then-full) on the same pod.
# Override with RUN_ALL=0 + CONFIG to run a single variant.
RUN_ALL="${RUN_ALL:-1}"
if [[ "$RUN_ALL" == "1" ]]; then
  echo "[run.sh] running run_all.sh (all three variants)"
  set +e
  bash run_all.sh
  EVAL_EXIT=$?
  set -e
else
  CONFIG="${CONFIG:-configs/qwen_7b_instruct.yaml}"
  CMD=(python eval.py --config "$CONFIG")
  [[ "${SMOKE:-}" == "1" ]] && CMD+=(--smoke)
  echo "[run.sh] single variant: ${CMD[*]}"
  set +e
  "${CMD[@]}"
  EVAL_EXIT=$?
  set -e
fi
echo "[run.sh] eval phase exited with code $EVAL_EXIT"

# --- Idle by default so RunPod doesn't restart-loop ----------------------
if [[ "${AUTO_TERMINATE:-}" == "1" ]]; then
  echo "[run.sh] AUTO_TERMINATE=1 — exiting with code $EVAL_EXIT"
  exit "$EVAL_EXIT"
fi
echo "[run.sh] eval done; pod is idling. Set AUTO_TERMINATE=1 to exit instead."
exec tail -f /dev/null
