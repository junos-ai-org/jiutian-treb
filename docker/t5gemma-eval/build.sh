#!/usr/bin/env bash
# Build + push the T5Gemma eval image.
# Run from the REPO ROOT on the AWS build server, since the Dockerfile COPYs
# from `experiments/t5gemma_vs_qwen_treb/` (paths are relative to the build context).
#
# Defaults to tagging only ${IMAGE}:${SHA} + any extra tags you pass.
# Does NOT touch `:latest` by default — opt in with LATEST=1 — so a PR-pinned
# build can't silently replace the image used by existing pods.
#
# Usage (from repo root):
#   docker/t5gemma-eval/build.sh pr45540-b8c3dff   # tag = <repo-sha> + pr45540-b8c3dff
#   PUSH=0 docker/t5gemma-eval/build.sh pr45540-b8c3dff   # build only
#   LATEST=1 docker/t5gemma-eval/build.sh           # also bump :latest (use with care)

set -euo pipefail

IMAGE="${IMAGE:-achithanar/treb-eval-t5gemma}"
PUSH="${PUSH:-1}"
LATEST="${LATEST:-0}"

# Resolve repo root from the script location, then build from there so the
# Dockerfile's `COPY docker/t5gemma-eval/...` and `COPY experiments/...` paths resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SHA="$(git rev-parse --short HEAD)"

TAGS=( "${IMAGE}:${SHA}" )
for extra in "$@"; do
  TAGS+=( "${IMAGE}:${extra}" )
done
if [[ "$LATEST" == "1" ]]; then
  TAGS+=( "${IMAGE}:latest" )
fi

TAG_ARGS=()
for t in "${TAGS[@]}"; do TAG_ARGS+=( -t "$t" ); done

echo "building: ${TAGS[*]}"
docker build "${TAG_ARGS[@]}" -f docker/t5gemma-eval/Dockerfile .

if [[ "$PUSH" == "1" ]]; then
  for t in "${TAGS[@]}"; do
    echo "pushing: $t"
    docker push "$t"
  done
fi

echo
echo "done. pinned tag: ${IMAGE}:${SHA}"
