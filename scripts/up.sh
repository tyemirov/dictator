#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/up.sh [docker compose up args...]

Builds the local GPU image from the current checkout and starts Dictator from
docker-compose.yml.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_ARGS=()
SERVICES=(dictator)

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

docker compose "${COMPOSE_ARGS[@]}" up --build "${UP_ARGS[@]}" "${SERVICES[@]}"
