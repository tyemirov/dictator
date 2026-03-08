#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/up.sh [docker compose up args...]

Pulls the GHCR GPU image and starts the GHCR GPU service profile from docker-compose.yml.

Environment:
  DICTATOR_IMAGE  Override the default image, e.g. ghcr.io/tyemirov/dictator-gpu:1.2.3
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_ARGS=(--profile ghcr-gpu)
SERVICES=(dictator-ghcr)

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

docker compose "${COMPOSE_ARGS[@]}" pull "${SERVICES[@]}"
docker compose "${COMPOSE_ARGS[@]}" up "${UP_ARGS[@]}" "${SERVICES[@]}"
