#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/voice-clone-demo.sh [options] [docker compose up args...]

Starts the browser voice-clone demo stack and serves the test page over HTTPS through ghttp.

By default this starts only the demo frontend + HTTP bridge. It expects a Dictator
service that is already running on the same Compose network under the shared alias
`dictator-grpc`, and it expects `DICTATOR_GRPC_AUTH_TOKEN` to be available in the
Compose environment. Use --with-dictator to also start the GHCR-backed GPU Dictator
service profile in the same Compose project.

Options:
  --with-dictator  Also start the GHCR-backed GPU Dictator service
  --host HOST      Public hostname to print after startup
  --port PORT      Publish the demo page on this host port
  --tls-cert PATH  Host path to the TLS certificate for ghttp
  --tls-key PATH   Host path to the TLS private key for ghttp
  -h, --help       Show this help

Environment:
  VOICE_CLONE_DEMO_HOST Public hostname to print (default: computercat.tyemirov.net)
  VOICE_CLONE_WEB_PORT  Host port for the demo page (default: 8001)
  TLS_CERT_HOST_PATH    Host path to the TLS certificate for ghttp
  TLS_KEY_HOST_PATH     Host path to the TLS private key for ghttp
  DICTATOR_GRPC_AUTH_TOKEN Dictator auth token shared with the backend bridge
  DICTATOR_IMAGE        Override the Dictator image, e.g. ghcr.io/tyemirov/dictator-gpu:1.2.3

If the TLS paths are not passed explicitly, the wrapper falls back to the repo-root .env file.

Examples:
  ./scripts/voice-clone-demo.sh --tls-cert /path/to/computercat-cert.pem --tls-key /path/to/computercat-key.pem
  ./scripts/voice-clone-demo.sh --host computercat.tyemirov.net --tls-cert /path/to/computercat-cert.pem --tls-key /path/to/computercat-key.pem
  ./scripts/voice-clone-demo.sh --with-dictator --port 8001 --tls-cert /path/to/computercat-cert.pem --tls-key /path/to/computercat-key.pem
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

trim_quotes() {
  local value="$1"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "${value}"
}

read_env_file_value() {
  local env_file_path="$1"
  local key="$2"

  [[ -f "$env_file_path" ]] || return 0
  awk -F '=' -v target_key="${key}" '$1 == target_key {sub($1 "=", ""); print; exit}' "${env_file_path}"
}

validate_file_path() {
  local path_label="$1"
  local path_value="$2"

  [[ -n "$path_value" ]] || die "${path_label} is required."
  [[ -f "$path_value" ]] || die "${path_label} is not a file: ${path_value}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE_PATH="${REPO_ROOT}/.env"

WITH_DICTATOR=0
PUBLIC_HOST="${VOICE_CLONE_DEMO_HOST:-computercat.tyemirov.net}"
PORT_VALUE="${VOICE_CLONE_WEB_PORT:-8001}"
TLS_CERT_PATH="${TLS_CERT_HOST_PATH:-}"
TLS_KEY_PATH="${TLS_KEY_HOST_PATH:-}"
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
    --tls-cert)
      shift
      [[ $# -gt 0 ]] || die "--tls-cert requires a value."
      TLS_CERT_PATH="$1"
      ;;
    --tls-key)
      shift
      [[ $# -gt 0 ]] || die "--tls-key requires a value."
      TLS_KEY_PATH="$1"
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

if [[ -z "$TLS_CERT_PATH" ]]; then
  TLS_CERT_PATH="$(trim_quotes "$(read_env_file_value "${ENV_FILE_PATH}" "TLS_CERT_HOST_PATH")")"
fi

if [[ -z "$TLS_KEY_PATH" ]]; then
  TLS_KEY_PATH="$(trim_quotes "$(read_env_file_value "${ENV_FILE_PATH}" "TLS_KEY_HOST_PATH")")"
fi

validate_file_path "--tls-cert / TLS_CERT_HOST_PATH" "$TLS_CERT_PATH"
validate_file_path "--tls-key / TLS_KEY_HOST_PATH" "$TLS_KEY_PATH"

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
export TLS_CERT_HOST_PATH="$TLS_CERT_PATH"
export TLS_KEY_HOST_PATH="$TLS_KEY_PATH"

if [[ "$WITH_DICTATOR" -eq 0 ]]; then
  export HF_TOKEN="${HF_TOKEN:-voice-clone-demo-placeholder-hf-token}"
fi

docker compose "${COMPOSE_ARGS[@]}" pull "${SERVICES[@]}"
docker compose "${COMPOSE_ARGS[@]}" up "${UP_ARGS[@]}" "${SERVICES[@]}"

printf 'Voice clone demo runs on https://%s:%s/\n' "$PUBLIC_HOST" "$VOICE_CLONE_WEB_PORT"
