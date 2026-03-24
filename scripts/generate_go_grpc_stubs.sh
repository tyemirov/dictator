#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_ROOT="${REPO_ROOT}/proto"
OUTPUT_ROOT="${REPO_ROOT}/sdk/go/dictatorspeechv1"
GO_MODULE="github.com/tyemirov/dictator/sdk/go/dictatorspeechv1"

if command -v go >/dev/null 2>&1; then
  export PATH="$(go env GOPATH)/bin:${PATH}"
fi

if ! command -v protoc >/dev/null 2>&1; then
  echo "Error: protoc is required to generate Go gRPC stubs." >&2
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

protoc \
  --proto_path="${PROTO_ROOT}" \
  --go_out="${OUTPUT_ROOT}" \
  --go_opt="module=${GO_MODULE}" \
  --go-grpc_out="${OUTPUT_ROOT}" \
  --go-grpc_opt="module=${GO_MODULE}" \
  "${PROTO_ROOT}"/dictator/speech/v1/*.proto
