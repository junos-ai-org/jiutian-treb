#!/usr/bin/env bash
# Pod entrypoint: prepare data once, then train.
#   CONFIG   — path to YAML config (default: configs/sft_flan.yaml)
#   RESUME   — if set, pass --resume to train.py
set -euo pipefail

cd "$(dirname "$0")"
CONFIG="${CONFIG:-configs/sft_flan.yaml}"

TOKENIZED=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['tokenized_dir'])" "$CONFIG")

if [[ ! -d "$TOKENIZED" ]]; then
  echo "[run.sh] tokenized data missing at $TOKENIZED — running prepare_data.py"
  python prepare_data.py --config "$CONFIG"
else
  echo "[run.sh] reusing tokenized data at $TOKENIZED"
fi

CMD=(python train.py --config "$CONFIG")
if [[ -n "${RESUME:-}" ]]; then CMD+=(--resume); fi
echo "[run.sh] exec: ${CMD[*]}"
exec "${CMD[@]}"
