#!/usr/bin/env bash
# Background pod monitor. Polls every N seconds for completion sentinels,
# SCPs predictions on success, auto-terminates the pod. Failures (no
# sentinel) leave the pod alive for manual diagnosis.
#
# Start this AFTER a pod launch, e.g.:
#   bash monitor_pod.sh <pod_id> "t5gemma_base t5gemma_sft" &
#
# Writes:
#   /tmp/pod_monitor.log   — every poll, timestamped state
#   /tmp/pod_alerts.log    — major events only (completions, terminations, errors)
#
# Reads from $HOME/.claude/.env for RUNPOD_API_KEY.
set -eo pipefail

POD_ID="${1:?usage: monitor_pod.sh POD_ID VARIANTS [LOCAL_DIR] [POLL_SEC]}"
VARIANTS="${2:?variants required, e.g. 't5gemma_base' or 't5gemma_base t5gemma_sft'}"
LOCAL_DIR="${3:-$(pwd)/results}"
POLL_SEC="${4:-180}"

set -a; . "$HOME/.claude/.env"; set +a

MON_LOG="/tmp/pod_monitor.log"
ALERT_LOG="/tmp/pod_alerts.log"
KEY="$HOME/.ssh/runpod_key"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log()   { echo "[$(ts)] [$POD_ID] $*" | tee -a "$MON_LOG"; }
alert() { echo "[$(ts)] [$POD_ID] $*" | tee -a "$ALERT_LOG" "$MON_LOG"; }

get_pod_state() {
  curl -sS -X POST "https://api.runpod.io/graphql" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"query { pod(input:{podId:\\\"$POD_ID\\\"}) { desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } } }\"}" \
  | python3 -c "
import sys, json
p = (json.load(sys.stdin).get('data') or {}).get('pod')
if not p:
    print('MISSING||')
    sys.exit()
rt = p.get('runtime') or {}
ssh = ''
for port in (rt.get('ports') or []):
    if port.get('privatePort') == 22 and port.get('isIpPublic'):
        ssh = f\"{port['ip']}:{port['publicPort']}\"
print(f\"{p.get('desiredStatus')}||{ssh}\")
"
}

check_sentinels() {
  # Return 0 if every variant has a .done sentinel on the pod
  local ip="${1%:*}"
  local port="${1#*:}"
  local count=0
  local needed=0
  for v in $VARIANTS; do
    needed=$((needed + 1))
    if ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
         -p "$port" "root@$ip" \
         "test -s /workspace/results/$v/predictions.jsonl.done" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  echo "$count/$needed"
}

scp_and_verify() {
  local ip="${1%:*}"
  local port="${1#*:}"
  mkdir -p "$LOCAL_DIR"
  for v in $VARIANTS; do
    local dst="$LOCAL_DIR/$v"
    mkdir -p "$dst"
    scp -i "$KEY" -o StrictHostKeyChecking=no -P "$port" \
        "root@$ip:/workspace/results/$v/predictions.jsonl" "$dst/" || return 1
    scp -i "$KEY" -o StrictHostKeyChecking=no -P "$port" \
        "root@$ip:/workspace/results/$v/predictions.jsonl.done" "$dst/" || return 1
    # Verify line count matches sentinel's n_predictions
    local expected=$(python3 -c "import json; print(json.load(open('$dst/predictions.jsonl.done'))['n_predictions'])")
    local actual=$(wc -l < "$dst/predictions.jsonl")
    if [[ "$expected" != "$actual" ]]; then
      alert "VERIFY-FAILED $v: sentinel says $expected, actual $actual"
      return 1
    fi
    log "verified $v: $actual predictions ($dst)"
  done
  return 0
}

terminate_pod() {
  local code=$(curl -sS -o /dev/null -w "%{http_code}" \
    -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY")
  if [[ "$code" == "204" ]]; then
    alert "TERMINATED $POD_ID (HTTP 204)"
    return 0
  fi
  alert "TERMINATE-FAILED $POD_ID (HTTP $code)"
  return 1
}

alert "MONITOR-START  variants=[$VARIANTS]  local=$LOCAL_DIR  poll=${POLL_SEC}s"

while true; do
  state=$(get_pod_state)
  status="${state%%||*}"
  ssh_ep="${state#*||}"
  log "state=$status  ssh=$ssh_ep"

  case "$status" in
    MISSING)
      alert "POD-MISSING — already terminated elsewhere. exiting monitor."
      exit 0
      ;;
    EXITED)
      alert "POD-EXITED-BY-RUNPOD — likely spend-cap or eviction. NOT terminating (already down). exiting monitor."
      exit 0
      ;;
    RUNNING)
      if [[ -n "$ssh_ep" && "$ssh_ep" != "||" ]]; then
        progress=$(check_sentinels "$ssh_ep")
        log "sentinels: $progress"
        if [[ "${progress%/*}" == "${progress#*/}" ]]; then
          alert "ALL-SENTINELS-PRESENT ($progress) — scp + verify"
          if scp_and_verify "$ssh_ep"; then
            alert "SCP-VERIFIED — terminating pod"
            terminate_pod
            exit 0
          else
            alert "SCP-OR-VERIFY-FAILED — leaving pod alive for manual"
            exit 1
          fi
        fi
      fi
      ;;
    *)
      log "unexpected status: $status"
      ;;
  esac

  sleep "$POLL_SEC"
done
