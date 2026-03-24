#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_ROOT="${REPO_ROOT}/proto"
OUTPUT_ROOT="${REPO_ROOT}/sdk/go/dictatorspeechv1"
GO_MODULE="github.com/tyemirov/dictator/sdk/go/dictatorspeechv1"
PROTOC_VERSION="34.0"
PROTOC_ROOT="${REPO_ROOT}/tools/protoc/${PROTOC_VERSION}"
PROTOC_BIN="${PROTOC_ROOT}/bin/protoc"
PROTO_INCLUDE_ARGS=(--proto_path="${PROTO_ROOT}")

if command -v go >/dev/null 2>&1; then
  export PATH="$(go env GOPATH)/bin:${PATH}"
fi

bootstrap_protoc() {
  local os arch asset url tmpdir

  case "$(uname -s)" in
    Linux) os="linux" ;;
    Darwin) os="osx" ;;
    *)
      echo "Error: unsupported OS for protoc bootstrap: $(uname -s)" >&2
      exit 1
      ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64) arch="x86_64" ;;
    arm64|aarch64) arch="aarch_64" ;;
    *)
      echo "Error: unsupported architecture for protoc bootstrap: $(uname -m)" >&2
      exit 1
      ;;
  esac

  if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required to bootstrap protoc ${PROTOC_VERSION}." >&2
    exit 1
  fi

  if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: unzip is required to bootstrap protoc ${PROTOC_VERSION}." >&2
    exit 1
  fi

  asset="protoc-${PROTOC_VERSION}-${os}-${arch}.zip"
  url="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${asset}"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  echo "Bootstrapping protoc ${PROTOC_VERSION}" >&2
  rm -rf "${PROTOC_ROOT}"
  mkdir -p "${PROTOC_ROOT}"
  curl -fsSL "${url}" -o "${tmpdir}/protoc.zip"
  unzip -q "${tmpdir}/protoc.zip" -d "${PROTOC_ROOT}"
}

if [ ! -x "${PROTOC_BIN}" ]; then
  bootstrap_protoc
fi

for candidate in \
  "${PROTOBUF_INCLUDE_DIR:-}" \
  "${PROTOC_ROOT}/include" \
  /opt/homebrew/include \
  /usr/local/include \
  /usr/include
do
  if [ -n "${candidate}" ] && [ -f "${candidate}/google/protobuf/struct.proto" ]; then
    PROTO_INCLUDE_ARGS+=(--proto_path="${candidate}")
    break
  fi
done

if grep -R -q 'import "google/protobuf/' "${PROTO_ROOT}" && [ "${#PROTO_INCLUDE_ARGS[@]}" -lt 2 ]; then
  echo "Error: google/protobuf/struct.proto was not found for protoc imports." >&2
  echo "Set PROTOBUF_INCLUDE_DIR or install the protobuf development headers." >&2
  exit 1
fi

if ! command -v protoc-gen-go >/dev/null 2>&1; then
  if ! command -v go >/dev/null 2>&1; then
    echo "Error: protoc-gen-go is required to generate Go protobuf stubs." >&2
    echo "Install Go and rerun this command." >&2
    exit 1
  fi
  echo "Installing protoc-gen-go@v1.36.11" >&2
  go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.11
fi

if ! command -v protoc-gen-go-grpc >/dev/null 2>&1; then
  if ! command -v go >/dev/null 2>&1; then
    echo "Error: protoc-gen-go-grpc is required to generate Go gRPC stubs." >&2
    echo "Install Go and rerun this command." >&2
    exit 1
  fi
  echo "Installing protoc-gen-go-grpc@v1.5.1" >&2
  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1
fi

mkdir -p "${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name '*.pb.go' -delete

"${PROTOC_BIN}" \
  "${PROTO_INCLUDE_ARGS[@]}" \
  --go_out="${OUTPUT_ROOT}" \
  --go_opt="module=${GO_MODULE}" \
  --go-grpc_out="${OUTPUT_ROOT}" \
  --go-grpc_opt="module=${GO_MODULE}" \
  "${PROTO_ROOT}"/dictator/speech/v1/*.proto
