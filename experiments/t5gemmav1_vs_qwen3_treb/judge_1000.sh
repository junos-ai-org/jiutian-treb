#!/usr/bin/env bash
# Score T5Gemma v1 + Gemma-2-2B predictions from the 1000-sample rerun.
# Uses DeepSeek v3.2 via OpenRouter as judge.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

T5G=${1:-results/t5gemma_v1_1k/predictions.jsonl}
G2=${2:-results/gemma2_2b_1k/predictions.jsonl}
WORKERS=${WORKERS:-12}

[[ -f "$T5G" ]] || { echo "missing $T5G"; exit 2; }
[[ -f "$G2" ]] || { echo "missing $G2"; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY not set"; exit 2; }

echo "[judge_1000] T5Gemma v1 → $(wc -l < "$T5G") samples"
python3 score.py --predictions "$T5G" --judge deepseek-v3.2 --judge-workers "$WORKERS" \
    --output results/t5gemma_v1_1k/metrics.json

echo "[judge_1000] Gemma-2-2B → $(wc -l < "$G2") samples"
python3 score.py --predictions "$G2" --judge deepseek-v3.2 --judge-workers "$WORKERS" \
    --output results/gemma2_2b_1k/metrics.json

echo "[judge_1000] merging into results/judged_1k_2way.jsonl"
python3 -c "
import json
from pathlib import Path

merged = []
for variant, path in [('t5gemma_v1', '$T5G'), ('gemma2_2b', '$G2')]:
    scored = Path(path).with_name(Path(path).stem + '.scored.jsonl')
    for line in scored.read_text().splitlines():
        if not line.strip(): continue
        d = json.loads(line)
        d.setdefault('variant', variant)
        merged.append(d)

out = Path('results/judged_1k_2way.jsonl')
out.write_text('\n'.join(json.dumps(m, ensure_ascii=False) for m in merged) + '\n')
print(f'wrote {len(merged)} rows to {out}')
"
echo "[judge_1000] done"
