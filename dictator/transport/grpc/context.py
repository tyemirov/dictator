"""Shared gRPC transport context."""

from __future__ import annotations

from dataclasses import dataclass

from dictator.runtime import InflightLimiter, MetricsRegistry, SpeechExecutionRuntime
from dictator.storage import LocalArtifactStore


@dataclass(frozen=True)
class ServiceContext:
    artifact_store: LocalArtifactStore
    execution_runtime: SpeechExecutionRuntime
    metrics: MetricsRegistry
    limiter: InflightLimiter
    auth_token: str | None
    download_chunk_bytes: int
    synthesis_job_manager: object | None = None
