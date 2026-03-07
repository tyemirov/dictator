"""Server construction for the Dictator gRPC transport."""

from __future__ import annotations

from concurrent import futures
import logging

import grpc
from grpc_health.v1 import health as grpc_health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from dictator.runtime import InflightLimiter, MetricsRegistry, SpeechExecutionRuntime
from dictator.storage import LocalArtifactStore

from .config import ServerConfig
from .services import ServiceContext, register_services

_SERVICE_NAMES = (
    "dictator.speech.v1.ArtifactService",
    "dictator.speech.v1.TranscriptionService",
    "dictator.speech.v1.AlignmentService",
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
        service_context = ServiceContext(
            artifact_store=LocalArtifactStore(config.artifact_root),
            execution_runtime=SpeechExecutionRuntime(),
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(config.max_inflight),
            auth_token=config.auth_token,
            download_chunk_bytes=config.download_chunk_bytes,
        )
    register_services(server, service_context)
    health_service = grpc_health.HealthServicer()
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
