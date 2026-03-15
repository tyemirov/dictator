"""Server construction for the Dictator gRPC transport."""

from __future__ import annotations

from concurrent import futures
import logging

import grpc
from grpc_health.v1 import health as grpc_health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from dictator.runtime import InflightLimiter, MetricsRegistry, SpeechExecutionRuntime
from dictator.runtime.jobs import (
    AlignmentJobManager,
    DiarizationJobManager,
    ExtractReferenceSampleJobManager,
    LocalAlignmentJobStore,
    LocalDiarizationJobStore,
    LocalExtractReferenceSampleJobStore,
    LocalSynthesisJobStore,
    LocalSubtitleJobStore,
    LocalTranscriptionJobStore,
    SynthesisJobManager,
    SubtitleJobManager,
    TranscriptionJobManager,
)
from dictator.storage import LocalArtifactStore

from .config import ServerConfig
from .services import ServiceContext, register_services

_SERVICE_NAMES = (
    "dictator.speech.v1.ArtifactService",
    "dictator.speech.v1.TranscriptionService",
    "dictator.speech.v1.AlignmentService",
    "dictator.speech.v1.SubtitleService",
    "dictator.speech.v1.VoiceService",
    "dictator.speech.v1.RuntimeService",
)


def build_server(
    config: ServerConfig,
    service_context: ServiceContext | None = None,
) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        options=(
            ("grpc.max_send_message_length", config.max_message_bytes),
            ("grpc.max_receive_message_length", config.max_message_bytes),
        ),
    )
    if service_context is None:
        execution_runtime = SpeechExecutionRuntime()
        artifact_store = LocalArtifactStore(config.artifact_root)
        service_context = ServiceContext(
            artifact_store=artifact_store,
            execution_runtime=execution_runtime,
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(config.max_inflight),
            auth_token=config.auth_token,
            download_chunk_bytes=config.download_chunk_bytes,
            synthesis_job_manager=SynthesisJobManager(
                job_store=LocalSynthesisJobStore(config.artifact_root / ".dictator-jobs"),
                artifact_store=artifact_store,
                execution_runtime=execution_runtime,
                max_workers=config.synthesis_job_workers,
                max_pending_jobs=config.max_pending_synthesis_jobs,
            ),
            alignment_job_manager=AlignmentJobManager(
                job_store=LocalAlignmentJobStore(config.artifact_root / ".dictator-alignment-jobs"),
                artifact_store=artifact_store,
                execution_runtime=execution_runtime,
                max_workers=config.alignment_job_workers,
                max_pending_jobs=config.max_pending_alignment_jobs,
            ),
            transcription_job_manager=TranscriptionJobManager(
                job_store=LocalTranscriptionJobStore(config.artifact_root / ".dictator-transcription-jobs"),
                artifact_store=artifact_store,
                execution_runtime=execution_runtime,
                max_workers=config.transcription_job_workers,
                max_pending_jobs=config.max_pending_transcription_jobs,
            ),
            diarization_job_manager=DiarizationJobManager(
                job_store=LocalDiarizationJobStore(config.artifact_root / ".dictator-diarization-jobs"),
                artifact_store=artifact_store,
                execution_runtime=execution_runtime,
                max_workers=config.diarization_job_workers,
                max_pending_jobs=config.max_pending_diarization_jobs,
            ),
            subtitle_job_manager=SubtitleJobManager(
                job_store=LocalSubtitleJobStore(config.artifact_root / ".dictator-subtitle-jobs"),
                artifact_store=artifact_store,
                execution_runtime=execution_runtime,
                max_workers=config.subtitle_job_workers,
                max_pending_jobs=config.max_pending_subtitle_jobs,
            ),
            reference_extraction_job_manager=ExtractReferenceSampleJobManager(
                job_store=LocalExtractReferenceSampleJobStore(
                    config.artifact_root / ".dictator-reference-extraction-jobs"
                ),
                artifact_store=artifact_store,
                execution_runtime=execution_runtime,
                max_workers=config.reference_extraction_job_workers,
                max_pending_jobs=config.max_pending_reference_extraction_jobs,
            ),
        )
    register_services(server, service_context)
    health_service = grpc_health.HealthServicer()
    health_service.set("", health_pb2.HealthCheckResponse.SERVING)
    for service_name in _SERVICE_NAMES:
        health_service.set(service_name, health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
    return server


def serve(config: ServerConfig) -> None:
    server = build_server(config)
    address = f"{config.host}:{config.port}"
    server.add_insecure_port(address)
    server.start()
    logging.info("dictator gRPC server listening on %s", address)
    server.wait_for_termination()
