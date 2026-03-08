#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/voice-clone-demo-down.sh [options] [docker compose rm args...]

Stops the browser voice-clone demo stack.

Options:
  --with-dictator  Also stop the GHCR-backed GPU Dictator service
  -h, --help       Show this help

Examples:
  ./scripts/voice-clone-demo-down.sh
  ./scripts/voice-clone-demo-down.sh --with-dictator
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WITH_DICTATOR=0
DOWN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-dictator)
      WITH_DICTATOR=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      DOWN_ARGS+=("$1")
      ;;
    *)
      DOWN_ARGS+=("$1")
      ;;
  esac
  shift
done

if ! command -v docker >/dev/null 2>&1; then
  die "docker is required."
fi

COMPOSE_ARGS=(--profile voice-clone-demo)
SERVICES=(voice-clone-web voice-clone-bridge)
if [[ "$WITH_DICTATOR" -eq 1 ]]; then
  COMPOSE_ARGS=(--profile ghcr-gpu "${COMPOSE_ARGS[@]}")
  SERVICES=(dictator-ghcr "${SERVICES[@]}")
fi

cd "${REPO_ROOT}"

if [[ "$WITH_DICTATOR" -eq 0 ]]; then
  export DICTATOR_GRPC_AUTH_TOKEN="${DICTATOR_GRPC_AUTH_TOKEN:-voice-clone-demo-placeholder-token}"
  export HF_TOKEN="${HF_TOKEN:-voice-clone-demo-placeholder-hf-token}"
fi

docker compose "${COMPOSE_ARGS[@]}" rm -f -s "${DOWN_ARGS[@]}" "${SERVICES[@]}"
