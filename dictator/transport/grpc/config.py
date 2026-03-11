"""Configuration for the Dictator gRPC server."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50051
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_INFLIGHT = 4
DEFAULT_SYNTHESIS_JOB_WORKERS = 1
DEFAULT_MAX_PENDING_SYNTHESIS_JOBS = 32
DEFAULT_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_ARTIFACT_ROOT = ".dictator-artifacts"

HOST_ENV = "DICTATOR_GRPC_HOST"
PORT_ENV = "DICTATOR_GRPC_PORT"
MAX_WORKERS_ENV = "DICTATOR_GRPC_MAX_WORKERS"
MAX_MESSAGE_BYTES_ENV = "DICTATOR_GRPC_MAX_MESSAGE_BYTES"
MAX_INFLIGHT_ENV = "DICTATOR_GRPC_MAX_INFLIGHT"
SYNTHESIS_JOB_WORKERS_ENV = "DICTATOR_GRPC_SYNTHESIS_JOB_WORKERS"
MAX_PENDING_SYNTHESIS_JOBS_ENV = "DICTATOR_GRPC_MAX_PENDING_SYNTHESIS_JOBS"
DOWNLOAD_CHUNK_BYTES_ENV = "DICTATOR_GRPC_DOWNLOAD_CHUNK_BYTES"
ARTIFACT_ROOT_ENV = "DICTATOR_GRPC_ARTIFACT_ROOT"
AUTH_TOKEN_ENV = "DICTATOR_GRPC_AUTH_TOKEN"
DEFAULT_CONFIG_PATH = Path("config.yml")

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_FIELD_NAMES = {
    "host",
    "port",
    "max_workers",
    "max_message_bytes",
    "max_inflight",
    "synthesis_job_workers",
    "max_pending_synthesis_jobs",
    "download_chunk_bytes",
    "artifact_root",
    "auth_token",
}


def _parse_positive_int(value: str, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _parse_scalar(value: str) -> object:
    stripped = value.strip()
    if not stripped:
        return ""
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _resolve_env_placeholders(value: object, env: Mapping[str, str]) -> object:
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        if env_name not in env or env[env_name] == "":
            raise ValueError(f"missing required env var {env_name}")
        return env[env_name]

    return _ENV_VAR_PATTERN.sub(replace, value)


def _load_config_mapping(config_file: Path | None, env: Mapping[str, str]) -> dict[str, object]:
    if config_file is None or not config_file.exists():
        return {}

    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]
    for line_number, raw_line in enumerate(
        config_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"invalid indentation in {config_file}:{line_number}")
        stripped = raw_line.strip()
        key, sep, raw_value = stripped.partition(":")
        if not sep:
            raise ValueError(f"invalid config line in {config_file}:{line_number}")
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            child: dict[str, object] = {}
            current[key] = child
            stack.append((indent, child))
            continue
        current[key] = _resolve_env_placeholders(_parse_scalar(raw_value), env)

    grpc_section = root.get("grpc")
    if isinstance(grpc_section, dict):
        mapping = grpc_section
    else:
        mapping = root
    return {key: value for key, value in mapping.items() if key in _FIELD_NAMES}


@dataclass(frozen=True)
class ServerConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_workers: int = DEFAULT_MAX_WORKERS
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    synthesis_job_workers: int = DEFAULT_SYNTHESIS_JOB_WORKERS
    max_pending_synthesis_jobs: int = DEFAULT_MAX_PENDING_SYNTHESIS_JOBS
    download_chunk_bytes: int = DEFAULT_DOWNLOAD_CHUNK_BYTES
    artifact_root: Path = Path(DEFAULT_ARTIFACT_ROOT)
    auth_token: str | None = None

    def overlay(self, **overrides: object) -> "ServerConfig":
        host = str(overrides.get("host", self.host)).strip() or self.host
        port = _parse_positive_int(str(overrides.get("port", self.port)), "port")
        max_workers = _parse_positive_int(
            str(overrides.get("max_workers", self.max_workers)),
            "max_workers",
        )
        max_message_bytes = _parse_positive_int(
            str(overrides.get("max_message_bytes", self.max_message_bytes)),
            "max_message_bytes",
        )
        max_inflight = _parse_positive_int(
            str(overrides.get("max_inflight", self.max_inflight)),
            "max_inflight",
        )
        synthesis_job_workers = _parse_positive_int(
            str(overrides.get("synthesis_job_workers", self.synthesis_job_workers)),
            "synthesis_job_workers",
        )
        max_pending_synthesis_jobs = _parse_positive_int(
            str(overrides.get("max_pending_synthesis_jobs", self.max_pending_synthesis_jobs)),
            "max_pending_synthesis_jobs",
        )
        download_chunk_bytes = _parse_positive_int(
            str(overrides.get("download_chunk_bytes", self.download_chunk_bytes)),
            "download_chunk_bytes",
        )
        artifact_root = Path(str(overrides.get("artifact_root", self.artifact_root))).expanduser()
        auth_override = overrides.get("auth_token", self.auth_token)
        auth_token = None if auth_override is None else str(auth_override).strip() or None
        return ServerConfig(
            host=host,
            port=port,
            max_workers=max_workers,
            max_message_bytes=max_message_bytes,
            max_inflight=max_inflight,
            synthesis_job_workers=synthesis_job_workers,
            max_pending_synthesis_jobs=max_pending_synthesis_jobs,
            download_chunk_bytes=download_chunk_bytes,
            artifact_root=artifact_root,
            auth_token=auth_token,
        )

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
        synthesis_job_workers = _parse_positive_int(
            source.get(SYNTHESIS_JOB_WORKERS_ENV, str(DEFAULT_SYNTHESIS_JOB_WORKERS)),
            SYNTHESIS_JOB_WORKERS_ENV,
        )
        max_pending_synthesis_jobs = _parse_positive_int(
            source.get(MAX_PENDING_SYNTHESIS_JOBS_ENV, str(DEFAULT_MAX_PENDING_SYNTHESIS_JOBS)),
            MAX_PENDING_SYNTHESIS_JOBS_ENV,
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
            synthesis_job_workers=synthesis_job_workers,
            max_pending_synthesis_jobs=max_pending_synthesis_jobs,
            download_chunk_bytes=download_chunk_bytes,
            artifact_root=artifact_root,
            auth_token=auth_token,
        )

    @classmethod
    def from_sources(
        cls,
        *,
        config_file: Path | None = DEFAULT_CONFIG_PATH,
        env: Mapping[str, str] | None = None,
    ) -> "ServerConfig":
        merged_env = dict(os.environ if env is None else env)
        config = cls.from_env(merged_env)
        file_overrides = _load_config_mapping(config_file, merged_env)
        if file_overrides:
            config = config.overlay(**file_overrides)
        return config
