#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_REPO="achithanar"
DEFAULT_TAG="latest"

usage() {
    cat <<EOF
Build and push Docker images for TReB bidir_attn experiments.

Usage: $(basename "$0") [OPTIONS]

Options:
  --repo REPO    Docker Hub repo prefix (default: ${DEFAULT_REPO})
  --tag TAG      Image tag (default: ${DEFAULT_TAG})
  --no-push      Build only, don't push
  -h, --help     Show this help

Images built:
  REPO/treb-base:TAG       (shared base — not pushed)
  REPO/treb-t5gemma:TAG
  REPO/treb-qwen:TAG

Examples:
  $(basename "$0")                          # ${DEFAULT_REPO}/treb-qwen:latest
  $(basename "$0") --tag v1                 # ${DEFAULT_REPO}/treb-qwen:v1
  $(basename "$0") --no-push               # build only
EOF
}

REPO="$DEFAULT_REPO"
TAG="$DEFAULT_TAG"
PUSH=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)  REPO="$2"; shift 2 ;;
        --tag)   TAG="$2"; shift 2 ;;
        --no-push) PUSH=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

BASE_IMAGE="${REPO}/treb-base:${TAG}"
IMAGES=()

echo "Building base image (${BASE_IMAGE})..."
docker build -f "$SCRIPT_DIR/Dockerfile.base" \
    -t "${BASE_IMAGE}" "$REPO_ROOT"

echo ""
echo "Building T5Gemma image..."
docker build -f "$SCRIPT_DIR/Dockerfile.t5gemma" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${REPO}/treb-t5gemma:${TAG}" "$REPO_ROOT"
IMAGES+=("${REPO}/treb-t5gemma:${TAG}")

echo ""
echo "Building Qwen image..."
docker build -f "$SCRIPT_DIR/Dockerfile.qwen" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${REPO}/treb-qwen:${TAG}" "$REPO_ROOT"
IMAGES+=("${REPO}/treb-qwen:${TAG}")

if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing images..."
    for img in "${IMAGES[@]}"; do
        docker push "$img"
    done
fi

echo ""
echo "Done:"
for img in "${IMAGES[@]}"; do
    echo "  $img"
done
