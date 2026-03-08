#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/up.sh [docker compose up args...]

Pulls the GHCR GPU image and starts the local Compose stack.

Environment:
  DICTATOR_IMAGE  Override the default image, e.g. ghcr.io/tyemirov/dictator-gpu:1.2.3
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/compose.ghcr.gpu.yml"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required." >&2
  exit 1
fi

cd "${REPO_ROOT}"

UP_ARGS=("$@")
if [[ "${#UP_ARGS[@]}" -eq 0 ]]; then
  UP_ARGS=(-d)
fi

docker compose -f "${COMPOSE_FILE}" pull
docker compose -f "${COMPOSE_FILE}" up "${UP_ARGS[@]}"
