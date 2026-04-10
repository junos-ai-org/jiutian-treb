#!/bin/bash
set -euo pipefail

# --- Logging & error handling ---------------------------------------------------
log()  { echo "[entrypoint] $(date '+%H:%M:%S') $*"; }
warn() { echo "[entrypoint] $(date '+%H:%M:%S') WARNING: $*" >&2; }
die()  { echo "[entrypoint] $(date '+%H:%M:%S') FATAL: $*" >&2; exit 1; }

trap 'rc=$?; echo ""; die "entrypoint failed at line $LINENO (exit code $rc). Check logs above."' ERR

# --- Deploy key (optional — only needed for private forks) ----------------------
_setup_deploy_key() {
    log "Setting up GitHub deploy key..."
    mkdir -p ~/.ssh
    python3 -c "
import base64, os, sys

raw = os.environ['DEPLOY_KEY'].strip()
raw = raw.replace(' ', '').replace('\n', '').replace('\r', '')
raw += '=' * (-len(raw) % 4)

n_data = len(raw.rstrip('='))
if n_data % 4 == 1:
    print(f'ERROR: DEPLOY_KEY has {n_data} base64 data chars (remainder 1 mod 4).', file=sys.stderr)
    sys.exit(1)

try:
    decoded = base64.b64decode(raw)
except Exception as e:
    print(f'ERROR: Failed to base64-decode DEPLOY_KEY: {e}', file=sys.stderr)
    sys.exit(1)

open('/root/.ssh/github_deploy_key', 'wb').write(decoded)
print(f'  Decoded deploy key ({len(decoded)} bytes).')
"
    chmod 600 ~/.ssh/github_deploy_key
    cat > ~/.ssh/config << 'SSHEOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_deploy_key
    IdentitiesOnly yes
SSHEOF
    chmod 600 ~/.ssh/config
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
    log "  Deploy key ready."
}

if [ -n "${DEPLOY_KEY:-}" ]; then
    if ! _setup_deploy_key; then
        warn "Deploy key setup failed. Continuing without it."
    fi
fi

# --- Clone or update TReB code -------------------------------------------------
TREB_REPO="${TREB_GIT_URL:-git@github.com:junos-ai-org/jiutian-treb.git}"
TREB_REF="${TREB_GIT_REF:-bidir-attn-experiment}"

if [ -d /workspace/jiutian-treb/.git ]; then
    log "Updating TReB code (ref: ${TREB_REF})..."
    (cd /workspace/jiutian-treb && git fetch origin && git checkout "$TREB_REF" && git pull origin "$TREB_REF") || warn "Git update failed — continuing with existing code."
else
    log "Cloning TReB code (ref: ${TREB_REF}) from ${TREB_REPO}..."
    git clone --branch "$TREB_REF" "$TREB_REPO" /workspace/jiutian-treb || warn "Git clone failed — SSH in to debug. Container will stay alive."
fi

(cd /workspace/jiutian-treb && git config --replace-all remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*") 2>/dev/null || true

# TReB's Sample class writes temp CSVs to ./tmp/
mkdir -p /workspace/jiutian-treb/src/tmp

# --- Model weights --------------------------------------------------------------
export HF_HOME="/workspace/.cache/huggingface"

T5GEMMA_MODEL="${T5GEMMA_MODEL_PATH:-google/t5gemma-9b-9b-ul2-it}"
log "Downloading model weights for ${T5GEMMA_MODEL}..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('${T5GEMMA_MODEL}')"
log "  T5Gemma model weights ready."

# --- Ready banner ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  TReB — T5Gemma image ready"
echo "============================================================"
echo ""
echo "  cd /workspace/jiutian-treb"
echo ""
echo "  # 1. Sample datasets (if not already done)"
echo "  python scripts/sample_dataset.py --size smoke --output experiments/bidir_attn/data/smoke/"
echo "  python scripts/sample_dataset.py --size large --output experiments/bidir_attn/data/large/"
echo ""
echo "  # 2. Generate configs (if not already done)"
echo "  python scripts/generate_configs.py"
echo ""
echo "  # 3. Run inference"
echo "  cd src"
echo "  python run_eval.py --config ../experiments/bidir_attn/configs/config_t5gemma_smoke.json --run_step reason"
echo "  python run_eval.py --config ../experiments/bidir_attn/configs/config_t5gemma_large.json --run_step reason"
echo ""
echo "  # 4. Run judge (needs OPENAI_API_KEY)"
echo "  export OPENAI_API_KEY=..."
echo "  python run_eval.py --config ../experiments/bidir_attn/configs/config_t5gemma_smoke.json --run_step judge"
echo "  python run_eval.py --config ../experiments/bidir_attn/configs/config_t5gemma_large.json --run_step judge"
echo ""
echo "  # Update code mid-session"
echo "  treb-update"
echo ""

exec sleep infinity
