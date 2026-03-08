#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/down.sh [docker compose down args...]

Stops the GHCR-backed GPU Compose stack started by scripts/up.sh.
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

docker compose -f "${COMPOSE_FILE}" down "$@"
