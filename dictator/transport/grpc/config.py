"""Configuration for the Dictator gRPC server."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50051
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_INFLIGHT = 4
DEFAULT_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_ARTIFACT_ROOT = ".dictator-artifacts"

HOST_ENV = "DICTATOR_GRPC_HOST"
PORT_ENV = "DICTATOR_GRPC_PORT"
MAX_WORKERS_ENV = "DICTATOR_GRPC_MAX_WORKERS"
MAX_MESSAGE_BYTES_ENV = "DICTATOR_GRPC_MAX_MESSAGE_BYTES"
MAX_INFLIGHT_ENV = "DICTATOR_GRPC_MAX_INFLIGHT"
DOWNLOAD_CHUNK_BYTES_ENV = "DICTATOR_GRPC_DOWNLOAD_CHUNK_BYTES"
ARTIFACT_ROOT_ENV = "DICTATOR_GRPC_ARTIFACT_ROOT"
AUTH_TOKEN_ENV = "DICTATOR_GRPC_AUTH_TOKEN"


def _parse_positive_int(value: str, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


@dataclass(frozen=True)
class ServerConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_workers: int = DEFAULT_MAX_WORKERS
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    download_chunk_bytes: int = DEFAULT_DOWNLOAD_CHUNK_BYTES
    artifact_root: Path = Path(DEFAULT_ARTIFACT_ROOT)
    auth_token: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ServerConfig":
        source = dict(os.environ if env is None else env)
        host = source.get(HOST_ENV, DEFAULT_HOST).strip() or DEFAULT_HOST
        port = _parse_positive_int(source.get(PORT_ENV, str(DEFAULT_PORT)), PORT_ENV)
        max_workers = _parse_positive_int(
            source.get(MAX_WORKERS_ENV, str(DEFAULT_MAX_WORKERS)),
            MAX_WORKERS_ENV,
        )
        max_message_bytes = _parse_positive_int(
            source.get(MAX_MESSAGE_BYTES_ENV, str(DEFAULT_MAX_MESSAGE_BYTES)),
            MAX_MESSAGE_BYTES_ENV,
        )
        max_inflight = _parse_positive_int(
            source.get(MAX_INFLIGHT_ENV, str(DEFAULT_MAX_INFLIGHT)),
            MAX_INFLIGHT_ENV,
        )
        download_chunk_bytes = _parse_positive_int(
            source.get(DOWNLOAD_CHUNK_BYTES_ENV, str(DEFAULT_DOWNLOAD_CHUNK_BYTES)),
            DOWNLOAD_CHUNK_BYTES_ENV,
        )
        artifact_root = Path(source.get(ARTIFACT_ROOT_ENV, DEFAULT_ARTIFACT_ROOT)).expanduser()
        auth_token = source.get(AUTH_TOKEN_ENV, "").strip() or None
        return cls(
            host=host,
            port=port,
            max_workers=max_workers,
            max_message_bytes=max_message_bytes,
            max_inflight=max_inflight,
            download_chunk_bytes=download_chunk_bytes,
            artifact_root=artifact_root,
            auth_token=auth_token,
        )
