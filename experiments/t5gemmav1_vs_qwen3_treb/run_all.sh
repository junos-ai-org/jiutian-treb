#!/usr/bin/env bash
# Smoke-then-full orchestrator for this experiment's pods.
#
# Each image runs a single variant; VARIANTS env picks which:
#   t5gemma-v1-eval image (transformers) → VARIANTS="t5gemma_v1"  (1 GPU)
#   qwen3-eval image      (vllm)         → VARIANTS="qwen3"       (1 GPU)
#
# Override VARIANTS to run something else.
#
# Env vars:
#   VARIANTS     space-separated names from: t5gemma_v1 | qwen3
#   SKIP_SMOKE   if 1, skip smoke phase
#   SKIP_FULL    if 1, skip full phase
#   PARALLEL     if 0, run variants sequentially (default: 1, parallel)
#
# Logs:    /workspace/logs/{smoke,full}_<variant>.log
# Results: /workspace/results/<variant>/{predictions.jsonl,metrics.json}
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p /workspace/logs /workspace/results

declare -A CONFIG_OF=(
  [t5gemma_v1]="configs/t5gemma_v1_2b2b_ul2_it.yaml"
  [qwen3]="configs/qwen3_4b_instruct.yaml"
)

VARIANTS="${VARIANTS:-t5gemma_v1}"
PARALLEL="${PARALLEL:-1}"

for v in $VARIANTS; do
  [[ -n "${CONFIG_OF[$v]:-}" ]] || { echo "[run_all] unknown variant: $v"; exit 2; }
  [[ -f "${CONFIG_OF[$v]}" ]]    || { echo "[run_all] missing config: ${CONFIG_OF[$v]}"; exit 2; }
done

run_phase() {
  local phase="$1"
  local extra_args=()
  [[ "$phase" == "smoke" ]] && extra_args+=(--smoke)

  echo "========================================"
  echo "[run_all] phase=$phase variants=($VARIANTS) parallel=$PARALLEL"
  echo "========================================"

  declare -A PIDS
  local gpu=0
  for name in $VARIANTS; do
    local cfg="${CONFIG_OF[$name]}"
    local log="/workspace/logs/${phase}_${name}.log"
    echo "[run_all]   $name (GPU $gpu) → $log"
    if [[ "$PARALLEL" == "1" ]]; then
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python eval.py --config "$cfg" "${extra_args[@]}"
      ) > "$log" 2>&1 &
      PIDS[$name]=$!
    else
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python eval.py --config "$cfg" "${extra_args[@]}"
      ) > "$log" 2>&1 || {
        echo "[run_all]   ✗ $phase $name FAILED — tail $log"
        return 1
      }
      echo "[run_all]   ✓ $phase $name"
    fi
    gpu=$((gpu + 1))
  done

  if [[ "$PARALLEL" == "1" ]]; then
    local fail=0
    for name in "${!PIDS[@]}"; do
      if wait "${PIDS[$name]}"; then
        echo "[run_all]   ✓ $phase $name"
      else
        echo "[run_all]   ✗ $phase $name FAILED — tail /workspace/logs/${phase}_${name}.log"
        fail=1
      fi
    done
    return $fail
  fi
  return 0
}

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  if ! run_phase smoke; then
    echo "[run_all] smoke failed — aborting before full run"
    exit 1
  fi
  echo "[run_all] smoke OK — proceeding to full run"
fi

if [[ "${SKIP_FULL:-0}" != "1" ]]; then
  if ! run_phase full; then
    echo "[run_all] some full runs failed — check /workspace/logs/"
    exit 1
  fi
fi

echo "[run_all] done — results under /workspace/results/"
ls -la /workspace/results/ 2>/dev/null
