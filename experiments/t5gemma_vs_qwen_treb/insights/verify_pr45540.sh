#!/usr/bin/env bash
# Before/after harness for huggingface/transformers PR #45540.
#
# Phase 1 (control)   : test the currently-installed transformers.
# Phase 2 (treatment) : install the PR branch from a git checkout (so we
#                       capture the SHA), then test again.
#
# Designed to run on a t5gemma-eval pod (or any GPU box with the gated
# `google/t5gemma-2-4b-4b` cached). Outputs go to stdout + a log under
# /workspace/logs by default.
#
# Requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the environment for
# the gated google/t5gemma-2-4b-4b model. Pass it explicitly:
#   HF_TOKEN=hf_xxx bash verify_pr45540.sh
#
# Usage:
#   bash verify_pr45540.sh                    # both phases
#   PHASE=control   bash verify_pr45540.sh    # only the control run
#   PHASE=treatment bash verify_pr45540.sh    # only the treatment run
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [[ -z "$HUGGING_FACE_HUB_TOKEN" ]]; then
  echo "[verify] WARNING: HF_TOKEN/HUGGING_FACE_HUB_TOKEN unset — gated model will 401" >&2
fi

PHASE="${PHASE:-both}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/verify_pr45540_$(date +%Y%m%d-%H%M%S).log"
echo "[verify] logging to $LOG"
exec > >(tee -a "$LOG") 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_PY="$SCRIPT_DIR/verify_pr45540.py"

run_phase() {
  local label="$1"
  echo
  echo "=================================================================="
  echo "  PHASE: $label"
  echo "=================================================================="
  python "$VERIFY_PY"
}

if [[ "$PHASE" == "control" || "$PHASE" == "both" ]]; then
  run_phase "control (stock transformers)" || echo "[verify] control phase exited non-zero (expected — bug present)"
fi

if [[ "$PHASE" == "treatment" || "$PHASE" == "both" ]]; then
  echo
  echo "[verify] installing transformers from PR #45540 branch..."
  # Beichen-Ma's fork, branch fix-cross-attention-cache-not-sliding.
  # --break-system-packages: the runpod/pytorch base image is Ubuntu 24.04 (PEP 668).
  pip install --upgrade --force-reinstall --no-deps --break-system-packages \
    "git+https://github.com/Beichen-Ma/transformers.git@fix-cross-attention-cache-not-sliding"
  python -c "import transformers; print('[verify] reinstalled transformers', transformers.__version__)"
  run_phase "treatment (PR #45540 branch)"
fi

echo
echo "[verify] done. Pass criterion: zero failures at length > 4094 in the treatment phase."
