#!/usr/bin/env bash
# Build + push the SFT training image, tagged with the git short SHA and :latest.
# Run from this directory on the AWS build server after `git pull`.
#
# Usage:
#   ./build.sh                 # tag = <short-sha> + latest
#   ./build.sh v0.2            # extra tag on top (e.g., ./build.sh flan-100k)
#   PUSH=0 ./build.sh          # build only, skip push
#   IMAGE=myuser/foo ./build.sh

set -euo pipefail

IMAGE="${IMAGE:-achithanar/t5gemma-sft}"
PUSH="${PUSH:-1}"

# Must run from the dir containing the Dockerfile.
cd "$(dirname "$0")"

# Resolve SHA from the repo root; fail loudly if not a git checkout.
SHA="$(git rev-parse --short HEAD)"

TAGS=( "${IMAGE}:${SHA}" "${IMAGE}:latest" )
for extra in "$@"; do
  TAGS+=( "${IMAGE}:${extra}" )
done

TAG_ARGS=()
for t in "${TAGS[@]}"; do TAG_ARGS+=( -t "$t" ); done

echo "building: ${TAGS[*]}"
docker build "${TAG_ARGS[@]}" .

if [[ "$PUSH" == "1" ]]; then
  for t in "${TAGS[@]}"; do
    echo "pushing: $t"
    docker push "$t"
  done
fi

echo
echo "done. pinned tag: ${IMAGE}:${SHA}"
