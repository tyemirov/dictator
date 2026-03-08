#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/voice-clone-demo.sh [options] [docker compose up args...]

Starts the browser voice-clone demo stack that serves the test page through ghttp.

By default this starts only the demo frontend + HTTP bridge. Use --with-dictator
to also start the GHCR-backed GPU Dictator service profile in the same Compose project.

Options:
  --with-dictator  Also start the GHCR-backed GPU Dictator service
  --host HOST      Public hostname to print after startup
  --port PORT      Publish the demo page on this host port
  -h, --help       Show this help

Environment:
  VOICE_CLONE_DEMO_HOST Public hostname to print (default: computercat.tyemirov.net)
  VOICE_CLONE_WEB_PORT  Host port for the demo page (default: 8001)
  DICTATOR_IMAGE        Override the Dictator image, e.g. ghcr.io/tyemirov/dictator-gpu:1.2.3

Examples:
  ./scripts/voice-clone-demo.sh
  ./scripts/voice-clone-demo.sh --host computercat.tyemirov.net
  ./scripts/voice-clone-demo.sh --with-dictator --port 8001
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WITH_DICTATOR=0
PUBLIC_HOST="${VOICE_CLONE_DEMO_HOST:-computercat.tyemirov.net}"
PORT_VALUE="${VOICE_CLONE_WEB_PORT:-8001}"
UP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-dictator)
      WITH_DICTATOR=1
      ;;
    --host)
      shift
      [[ $# -gt 0 ]] || die "--host requires a value."
      PUBLIC_HOST="$1"
      ;;
    --port)
      shift
      [[ $# -gt 0 ]] || die "--port requires a value."
      PORT_VALUE="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      UP_ARGS+=("$1")
      ;;
  esac
  shift
done

if ! command -v docker >/dev/null 2>&1; then
  die "docker is required."
fi

if [[ -n "$PORT_VALUE" ]]; then
  [[ "$PORT_VALUE" =~ ^[0-9]+$ ]] || die "--port must be numeric."
fi

if [[ "${#UP_ARGS[@]}" -eq 0 ]]; then
  UP_ARGS=(-d)
fi

COMPOSE_ARGS=(--profile voice-clone-demo)
SERVICES=(voice-clone-web voice-clone-bridge)
if [[ "$WITH_DICTATOR" -eq 1 ]]; then
  COMPOSE_ARGS=(--profile ghcr-gpu "${COMPOSE_ARGS[@]}")
  SERVICES=(dictator-ghcr "${SERVICES[@]}")
fi

cd "${REPO_ROOT}"

if [[ -n "$PORT_VALUE" ]]; then
  export VOICE_CLONE_WEB_PORT="$PORT_VALUE"
fi

docker compose "${COMPOSE_ARGS[@]}" pull "${SERVICES[@]}"
docker compose "${COMPOSE_ARGS[@]}" up "${UP_ARGS[@]}" "${SERVICES[@]}"

printf 'Voice clone demo runs on http://%s:%s/\n' "$PUBLIC_HOST" "$VOICE_CLONE_WEB_PORT"
printf 'Local fallback: http://127.0.0.1:%s/\n' "$VOICE_CLONE_WEB_PORT"
