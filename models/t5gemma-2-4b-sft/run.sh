#!/usr/bin/env bash
# Pod entrypoint.
#
# Env vars:
#   CONFIG   — path to YAML config (default: configs/sft_flan.yaml)
#   RESUME   — if set, pass --resume to train.py
#   DEV      — if set to 1, sleep forever after training (keep pod alive
#              for SSH/debugging). Costs $$ while idle.
#   CODE_REPO / CODE_REF — if set, git-clone into /workspace/code and run
#              from there instead of the baked-in /opt/sft code. Lets us
#              iterate on train.py without rebuilding the image.
set -euo pipefail

# --- SSH bring-up ---------------------------------------------------------
# The RunPod base image's /start.sh normally starts sshd, but our custom
# CMD bypasses it. Replicate the essentials so we can SSH in.
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
if command -v sshd >/dev/null 2>&1; then
  mkdir -p /run/sshd
  # Base image ships without host keys; generate them so sshd doesn't
  # bail with "no hostkeys available -- exiting".
  ssh-keygen -A >/dev/null 2>&1 || true
  /usr/sbin/sshd
  echo "[run.sh] sshd started"
fi

# --- Log to volume so it survives container exits ------------------------
mkdir -p /workspace/logs
LOG=/workspace/logs/$(date +%Y%m%d-%H%M%S).log
echo "[run.sh] logging to $LOG"
exec > >(tee -a "$LOG") 2>&1

# --- Optional: run from a git checkout on the volume ---------------------
# If CODE_REPO is set, pull latest from there and cd in. This eliminates
# the "rebuild-image-for-every-train.py-change" loop.
CODE_DIR=/opt/sft
if [[ -n "${CODE_REPO:-}" ]]; then
  CODE_DIR=/workspace/code
  if [[ ! -d "$CODE_DIR/.git" ]]; then
    echo "[run.sh] cloning $CODE_REPO -> $CODE_DIR"
    git clone "$CODE_REPO" "$CODE_DIR"
  fi
  cd "$CODE_DIR"
  git fetch --all --quiet
  git checkout "${CODE_REF:-experiment-setup}"
  git pull --ff-only
  # Re-enter the actual SFT subdir of the repo.
  CODE_DIR="$CODE_DIR/models/t5gemma-2-4b-sft"
fi
cd "$CODE_DIR"
echo "[run.sh] code dir: $(pwd)  ($(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"

# --- Training -----------------------------------------------------------
CONFIG="${CONFIG:-configs/sft_flan.yaml}"
TOKENIZED=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['tokenized_dir'])" "$CONFIG")

if [[ ! -d "$TOKENIZED" ]]; then
  echo "[run.sh] preparing data at $TOKENIZED"
  python prepare_data.py --config "$CONFIG"
else
  echo "[run.sh] reusing tokenized data at $TOKENIZED"
fi

TRAIN_CMD=(python train.py --config "$CONFIG")
[[ -n "${RESUME:-}" ]] && TRAIN_CMD+=(--resume)
echo "[run.sh] ${TRAIN_CMD[*]}"

set +e
"${TRAIN_CMD[@]}"
TRAIN_EXIT=$?
set -e
echo "[run.sh] train.py exited with code $TRAIN_EXIT"

# --- Keep pod alive in dev mode for post-mortem -------------------------
if [[ "${DEV:-}" == "1" ]]; then
  echo "[run.sh] DEV=1 — sleeping forever so the pod stays reachable via SSH"
  exec tail -f /dev/null
fi

exit "$TRAIN_EXIT"
