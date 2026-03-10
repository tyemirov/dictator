#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/test-docker-image.sh [options]

Build or reuse a Docker image and run the in-image blackbox probe.
The probe generates a spoken WAV sample inside the image and verifies that the
service returns a synthesized Qwen3-TTS WAV back over gRPC.

Options:
  --image IMAGE       Probe an existing image instead of building one
  --tag TAG           Local tag to use when building (default: dictator:blackbox)
  --dockerfile PATH   Dockerfile to build (default: Dockerfile.gpu)
  --context PATH      Docker build context (default: repo root)
  -h, --help          Show this help
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME=""
LOCAL_TAG="dictator:blackbox"
DOCKERFILE_PATH="Dockerfile.gpu"
BUILD_CONTEXT="${REPO_ROOT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      shift
      [[ $# -gt 0 ]] || die "--image requires a value."
      IMAGE_NAME="$1"
      ;;
    --tag)
      shift
      [[ $# -gt 0 ]] || die "--tag requires a value."
      LOCAL_TAG="$1"
      ;;
    --dockerfile)
      shift
      [[ $# -gt 0 ]] || die "--dockerfile requires a value."
      DOCKERFILE_PATH="$1"
      ;;
    --context)
      shift
      [[ $# -gt 0 ]] || die "--context requires a value."
      BUILD_CONTEXT="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
  shift
done

require_cmd docker

cd "${REPO_ROOT}"

if [[ -z "${IMAGE_NAME}" ]]; then
  printf 'Building Docker image %s from %s\n' "${LOCAL_TAG}" "${DOCKERFILE_PATH}"
  docker build --file "${DOCKERFILE_PATH}" --tag "${LOCAL_TAG}" "${BUILD_CONTEXT}"
  IMAGE_NAME="${LOCAL_TAG}"
fi

printf 'Running blackbox probe inside %s\n' "${IMAGE_NAME}"
docker run \
  --rm \
  --entrypoint python \
  "${IMAGE_NAME}" \
  /app/scripts/docker_image_blackbox_probe.py
