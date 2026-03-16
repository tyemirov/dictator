#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/demo-up.sh [voice-clone-demo args...]

Builds and starts the local Dictator service from the current checkout, then
starts the browser voice-clone demo stack against that local service.

This is the local-development path. It intentionally does not use the published
Dictator image for the backend service.

By default it overrides `DICTATOR_HOST_PORT` to `50003` so the local demo can
coexist with the deployed computercat stack on `50002`.

Examples:
  ./scripts/demo-up.sh --tls-cert /path/to/computercat-cert.pem --tls-key /path/to/computercat-key.pem
  ./scripts/demo-up.sh --host computercat.tyemirov.net --port 8001 --tls-cert /path/to/computercat-cert.pem --tls-key /path/to/computercat-key.pem
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for arg in "$@"; do
  if [[ "$arg" == "--with-dictator" ]]; then
    echo "Error: demo-up.sh always uses the local Dictator build; do not pass --with-dictator." >&2
    exit 1
  fi
done

export DICTATOR_HOST_PORT="${DICTATOR_HOST_PORT:-50003}"
export VOICE_CLONE_BRIDGE_IMAGE="${VOICE_CLONE_BRIDGE_IMAGE:-dictator:gpu}"
export VOICE_CLONE_DEMO_SKIP_BRIDGE_PULL=1

"${REPO_ROOT}/scripts/up.sh"
"${REPO_ROOT}/scripts/voice-clone-demo.sh" --pull never --force-recreate "$@"
