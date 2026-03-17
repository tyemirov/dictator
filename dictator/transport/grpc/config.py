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
DEFAULT_ALIGNMENT_JOB_WORKERS = 1
DEFAULT_MAX_PENDING_ALIGNMENT_JOBS = 32
DEFAULT_TRANSCRIPTION_JOB_WORKERS = 1
DEFAULT_MAX_PENDING_TRANSCRIPTION_JOBS = 32
DEFAULT_DIARIZATION_JOB_WORKERS = 1
DEFAULT_MAX_PENDING_DIARIZATION_JOBS = 32
DEFAULT_SUBTITLE_JOB_WORKERS = 1
DEFAULT_MAX_PENDING_SUBTITLE_JOBS = 32
DEFAULT_REFERENCE_EXTRACTION_JOB_WORKERS = 1
DEFAULT_MAX_PENDING_REFERENCE_EXTRACTION_JOBS = 32
DEFAULT_JOB_WAIT_TIMEOUT_SECONDS = 300.0
DEFAULT_JOB_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_ARTIFACT_ROOT = ".dictator-artifacts"

HOST_ENV = "DICTATOR_GRPC_HOST"
PORT_ENV = "DICTATOR_GRPC_PORT"
MAX_WORKERS_ENV = "DICTATOR_GRPC_MAX_WORKERS"
MAX_MESSAGE_BYTES_ENV = "DICTATOR_GRPC_MAX_MESSAGE_BYTES"
MAX_INFLIGHT_ENV = "DICTATOR_GRPC_MAX_INFLIGHT"
SYNTHESIS_JOB_WORKERS_ENV = "DICTATOR_GRPC_SYNTHESIS_JOB_WORKERS"
MAX_PENDING_SYNTHESIS_JOBS_ENV = "DICTATOR_GRPC_MAX_PENDING_SYNTHESIS_JOBS"
ALIGNMENT_JOB_WORKERS_ENV = "DICTATOR_GRPC_ALIGNMENT_JOB_WORKERS"
MAX_PENDING_ALIGNMENT_JOBS_ENV = "DICTATOR_GRPC_MAX_PENDING_ALIGNMENT_JOBS"
TRANSCRIPTION_JOB_WORKERS_ENV = "DICTATOR_GRPC_TRANSCRIPTION_JOB_WORKERS"
MAX_PENDING_TRANSCRIPTION_JOBS_ENV = "DICTATOR_GRPC_MAX_PENDING_TRANSCRIPTION_JOBS"
DIARIZATION_JOB_WORKERS_ENV = "DICTATOR_GRPC_DIARIZATION_JOB_WORKERS"
MAX_PENDING_DIARIZATION_JOBS_ENV = "DICTATOR_GRPC_MAX_PENDING_DIARIZATION_JOBS"
SUBTITLE_JOB_WORKERS_ENV = "DICTATOR_GRPC_SUBTITLE_JOB_WORKERS"
MAX_PENDING_SUBTITLE_JOBS_ENV = "DICTATOR_GRPC_MAX_PENDING_SUBTITLE_JOBS"
REFERENCE_EXTRACTION_JOB_WORKERS_ENV = "DICTATOR_GRPC_REFERENCE_EXTRACTION_JOB_WORKERS"
MAX_PENDING_REFERENCE_EXTRACTION_JOBS_ENV = "DICTATOR_GRPC_MAX_PENDING_REFERENCE_EXTRACTION_JOBS"
JOB_WAIT_TIMEOUT_SECONDS_ENV = "DICTATOR_GRPC_JOB_WAIT_TIMEOUT_SECONDS"
JOB_POLL_INTERVAL_SECONDS_ENV = "DICTATOR_GRPC_JOB_POLL_INTERVAL_SECONDS"
DOWNLOAD_CHUNK_BYTES_ENV = "DICTATOR_GRPC_DOWNLOAD_CHUNK_BYTES"
ARTIFACT_ROOT_ENV = "DICTATOR_GRPC_ARTIFACT_ROOT"
AUTH_TOKEN_ENV = "DICTATOR_GRPC_AUTH_TOKEN"
DEFAULT_CONFIG_PATH = Path("config.yml")

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CONFIG_SCHEMA = {
    "server": {
        "listen": {
            "host": None,
            "port": None,
        },
        "grpc": {
            "max_message_bytes": None,
            "auth_token": None,
        },
    },
    "execution": {
        "concurrency": {
            "workers": None,
            "inflight": None,
        },
        "jobs": {
            "wait_timeout_seconds": None,
            "poll_interval_seconds": None,
            "synthesis": {
                "workers": None,
                "max_pending": None,
            },
            "alignment": {
                "workers": None,
                "max_pending": None,
            },
            "transcription": {
                "workers": None,
                "max_pending": None,
            },
            "diarization": {
                "workers": None,
                "max_pending": None,
            },
            "subtitle": {
                "workers": None,
                "max_pending": None,
            },
            "reference_extraction": {
                "workers": None,
                "max_pending": None,
            },
        },
    },
    "downloads": {
        "chunk_bytes": None,
    },
    "storage": {
        "artifacts": {
            "root": None,
        },
    },
}
_FIELD_PATHS = {
    ("server", "listen", "host"): "host",
    ("server", "listen", "port"): "port",
    ("server", "grpc", "max_message_bytes"): "max_message_bytes",
    ("server", "grpc", "auth_token"): "auth_token",
    ("execution", "concurrency", "workers"): "max_workers",
    ("execution", "concurrency", "inflight"): "max_inflight",
    ("execution", "jobs", "wait_timeout_seconds"): "job_wait_timeout_seconds",
    ("execution", "jobs", "poll_interval_seconds"): "job_poll_interval_seconds",
    ("execution", "jobs", "synthesis", "workers"): "synthesis_job_workers",
    ("execution", "jobs", "synthesis", "max_pending"): "max_pending_synthesis_jobs",
    ("execution", "jobs", "alignment", "workers"): "alignment_job_workers",
    ("execution", "jobs", "alignment", "max_pending"): "max_pending_alignment_jobs",
    ("execution", "jobs", "transcription", "workers"): "transcription_job_workers",
    ("execution", "jobs", "transcription", "max_pending"): "max_pending_transcription_jobs",
    ("execution", "jobs", "diarization", "workers"): "diarization_job_workers",
    ("execution", "jobs", "diarization", "max_pending"): "max_pending_diarization_jobs",
    ("execution", "jobs", "subtitle", "workers"): "subtitle_job_workers",
    ("execution", "jobs", "subtitle", "max_pending"): "max_pending_subtitle_jobs",
    ("execution", "jobs", "reference_extraction", "workers"): "reference_extraction_job_workers",
    ("execution", "jobs", "reference_extraction", "max_pending"): "max_pending_reference_extraction_jobs",
    ("downloads", "chunk_bytes"): "download_chunk_bytes",
    ("storage", "artifacts", "root"): "artifact_root",
}


def _parse_positive_int(value: str, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _parse_positive_float(value: str, label: str) -> float:
    parsed = float(value)
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


def _path_label(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _validate_config_mapping(
    mapping: dict[str, object],
    *,
    schema: dict[str, object],
    path: tuple[str, ...] = (),
) -> None:
    for key, value in mapping.items():
        if key not in schema:
            raise ValueError(f"unknown config key {_path_label(path + (key,))}")
        child_schema = schema[key]
        child_path = path + (key,)
        if child_schema is None:
            if isinstance(value, dict):
                raise ValueError(f"config key {_path_label(child_path)} must be a scalar")
            continue
        if not isinstance(value, dict):
            raise ValueError(f"config key {_path_label(child_path)} must be a mapping")
        _validate_config_mapping(value, schema=child_schema, path=child_path)


def _lookup_mapping_value(mapping: dict[str, object], path: tuple[str, ...]) -> tuple[object, bool]:
    current: object = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None, False
        current = current[key]
    return current, True


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

    if not root:
        return {}

    _validate_config_mapping(root, schema=_CONFIG_SCHEMA)

    flattened: dict[str, object] = {}
    for path, field_name in _FIELD_PATHS.items():
        value, found = _lookup_mapping_value(root, path)
        if found:
            flattened[field_name] = value
    return flattened


@dataclass(frozen=True)
class ServerConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_workers: int = DEFAULT_MAX_WORKERS
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    synthesis_job_workers: int = DEFAULT_SYNTHESIS_JOB_WORKERS
    max_pending_synthesis_jobs: int = DEFAULT_MAX_PENDING_SYNTHESIS_JOBS
    alignment_job_workers: int = DEFAULT_ALIGNMENT_JOB_WORKERS
    max_pending_alignment_jobs: int = DEFAULT_MAX_PENDING_ALIGNMENT_JOBS
    transcription_job_workers: int = DEFAULT_TRANSCRIPTION_JOB_WORKERS
    max_pending_transcription_jobs: int = DEFAULT_MAX_PENDING_TRANSCRIPTION_JOBS
    diarization_job_workers: int = DEFAULT_DIARIZATION_JOB_WORKERS
    max_pending_diarization_jobs: int = DEFAULT_MAX_PENDING_DIARIZATION_JOBS
    subtitle_job_workers: int = DEFAULT_SUBTITLE_JOB_WORKERS
    max_pending_subtitle_jobs: int = DEFAULT_MAX_PENDING_SUBTITLE_JOBS
    reference_extraction_job_workers: int = DEFAULT_REFERENCE_EXTRACTION_JOB_WORKERS
    max_pending_reference_extraction_jobs: int = DEFAULT_MAX_PENDING_REFERENCE_EXTRACTION_JOBS
    job_wait_timeout_seconds: float = DEFAULT_JOB_WAIT_TIMEOUT_SECONDS
    job_poll_interval_seconds: float = DEFAULT_JOB_POLL_INTERVAL_SECONDS
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
        alignment_job_workers = _parse_positive_int(
            str(overrides.get("alignment_job_workers", self.alignment_job_workers)),
            "alignment_job_workers",
        )
        max_pending_alignment_jobs = _parse_positive_int(
            str(overrides.get("max_pending_alignment_jobs", self.max_pending_alignment_jobs)),
            "max_pending_alignment_jobs",
        )
        transcription_job_workers = _parse_positive_int(
            str(overrides.get("transcription_job_workers", self.transcription_job_workers)),
            "transcription_job_workers",
        )
        max_pending_transcription_jobs = _parse_positive_int(
            str(overrides.get("max_pending_transcription_jobs", self.max_pending_transcription_jobs)),
            "max_pending_transcription_jobs",
        )
        diarization_job_workers = _parse_positive_int(
            str(overrides.get("diarization_job_workers", self.diarization_job_workers)),
            "diarization_job_workers",
        )
        max_pending_diarization_jobs = _parse_positive_int(
            str(overrides.get("max_pending_diarization_jobs", self.max_pending_diarization_jobs)),
            "max_pending_diarization_jobs",
        )
        subtitle_job_workers = _parse_positive_int(
            str(overrides.get("subtitle_job_workers", self.subtitle_job_workers)),
            "subtitle_job_workers",
        )
        max_pending_subtitle_jobs = _parse_positive_int(
            str(overrides.get("max_pending_subtitle_jobs", self.max_pending_subtitle_jobs)),
            "max_pending_subtitle_jobs",
        )
        reference_extraction_job_workers = _parse_positive_int(
            str(
                overrides.get(
                    "reference_extraction_job_workers",
                    self.reference_extraction_job_workers,
                )
            ),
            "reference_extraction_job_workers",
        )
        max_pending_reference_extraction_jobs = _parse_positive_int(
            str(
                overrides.get(
                    "max_pending_reference_extraction_jobs",
                    self.max_pending_reference_extraction_jobs,
                )
            ),
            "max_pending_reference_extraction_jobs",
        )
        job_wait_timeout_seconds = _parse_positive_float(
            str(
                overrides.get(
                    "job_wait_timeout_seconds",
                    self.job_wait_timeout_seconds,
                )
            ),
            "job_wait_timeout_seconds",
        )
        job_poll_interval_seconds = _parse_positive_float(
            str(
                overrides.get(
                    "job_poll_interval_seconds",
                    self.job_poll_interval_seconds,
                )
            ),
            "job_poll_interval_seconds",
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
            alignment_job_workers=alignment_job_workers,
            max_pending_alignment_jobs=max_pending_alignment_jobs,
            transcription_job_workers=transcription_job_workers,
            max_pending_transcription_jobs=max_pending_transcription_jobs,
            diarization_job_workers=diarization_job_workers,
            max_pending_diarization_jobs=max_pending_diarization_jobs,
            subtitle_job_workers=subtitle_job_workers,
            max_pending_subtitle_jobs=max_pending_subtitle_jobs,
            reference_extraction_job_workers=reference_extraction_job_workers,
            max_pending_reference_extraction_jobs=max_pending_reference_extraction_jobs,
            job_wait_timeout_seconds=job_wait_timeout_seconds,
            job_poll_interval_seconds=job_poll_interval_seconds,
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
        alignment_job_workers = _parse_positive_int(
            source.get(ALIGNMENT_JOB_WORKERS_ENV, str(DEFAULT_ALIGNMENT_JOB_WORKERS)),
            ALIGNMENT_JOB_WORKERS_ENV,
        )
        max_pending_alignment_jobs = _parse_positive_int(
            source.get(MAX_PENDING_ALIGNMENT_JOBS_ENV, str(DEFAULT_MAX_PENDING_ALIGNMENT_JOBS)),
            MAX_PENDING_ALIGNMENT_JOBS_ENV,
        )
        transcription_job_workers = _parse_positive_int(
            source.get(TRANSCRIPTION_JOB_WORKERS_ENV, str(DEFAULT_TRANSCRIPTION_JOB_WORKERS)),
            TRANSCRIPTION_JOB_WORKERS_ENV,
        )
        max_pending_transcription_jobs = _parse_positive_int(
            source.get(
                MAX_PENDING_TRANSCRIPTION_JOBS_ENV,
                str(DEFAULT_MAX_PENDING_TRANSCRIPTION_JOBS),
            ),
            MAX_PENDING_TRANSCRIPTION_JOBS_ENV,
        )
        diarization_job_workers = _parse_positive_int(
            source.get(DIARIZATION_JOB_WORKERS_ENV, str(DEFAULT_DIARIZATION_JOB_WORKERS)),
            DIARIZATION_JOB_WORKERS_ENV,
        )
        max_pending_diarization_jobs = _parse_positive_int(
            source.get(MAX_PENDING_DIARIZATION_JOBS_ENV, str(DEFAULT_MAX_PENDING_DIARIZATION_JOBS)),
            MAX_PENDING_DIARIZATION_JOBS_ENV,
        )
        subtitle_job_workers = _parse_positive_int(
            source.get(SUBTITLE_JOB_WORKERS_ENV, str(DEFAULT_SUBTITLE_JOB_WORKERS)),
            SUBTITLE_JOB_WORKERS_ENV,
        )
        max_pending_subtitle_jobs = _parse_positive_int(
            source.get(MAX_PENDING_SUBTITLE_JOBS_ENV, str(DEFAULT_MAX_PENDING_SUBTITLE_JOBS)),
            MAX_PENDING_SUBTITLE_JOBS_ENV,
        )
        reference_extraction_job_workers = _parse_positive_int(
            source.get(
                REFERENCE_EXTRACTION_JOB_WORKERS_ENV,
                str(DEFAULT_REFERENCE_EXTRACTION_JOB_WORKERS),
            ),
            REFERENCE_EXTRACTION_JOB_WORKERS_ENV,
        )
        max_pending_reference_extraction_jobs = _parse_positive_int(
            source.get(
                MAX_PENDING_REFERENCE_EXTRACTION_JOBS_ENV,
                str(DEFAULT_MAX_PENDING_REFERENCE_EXTRACTION_JOBS),
            ),
            MAX_PENDING_REFERENCE_EXTRACTION_JOBS_ENV,
        )
        job_wait_timeout_seconds = _parse_positive_float(
            source.get(
                JOB_WAIT_TIMEOUT_SECONDS_ENV,
                str(DEFAULT_JOB_WAIT_TIMEOUT_SECONDS),
            ),
            JOB_WAIT_TIMEOUT_SECONDS_ENV,
        )
        job_poll_interval_seconds = _parse_positive_float(
            source.get(
                JOB_POLL_INTERVAL_SECONDS_ENV,
                str(DEFAULT_JOB_POLL_INTERVAL_SECONDS),
            ),
            JOB_POLL_INTERVAL_SECONDS_ENV,
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
            alignment_job_workers=alignment_job_workers,
            max_pending_alignment_jobs=max_pending_alignment_jobs,
            transcription_job_workers=transcription_job_workers,
            max_pending_transcription_jobs=max_pending_transcription_jobs,
            diarization_job_workers=diarization_job_workers,
            max_pending_diarization_jobs=max_pending_diarization_jobs,
            subtitle_job_workers=subtitle_job_workers,
            max_pending_subtitle_jobs=max_pending_subtitle_jobs,
            reference_extraction_job_workers=reference_extraction_job_workers,
            max_pending_reference_extraction_jobs=max_pending_reference_extraction_jobs,
            job_wait_timeout_seconds=job_wait_timeout_seconds,
            job_poll_interval_seconds=job_poll_interval_seconds,
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
        config = cls()
        file_overrides = _load_config_mapping(config_file, merged_env)
        if file_overrides:
            config = config.overlay(**file_overrides)
        return config
