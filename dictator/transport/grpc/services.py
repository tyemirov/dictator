"""Compatibility exports and service registration for gRPC servicers."""

from __future__ import annotations

import grpc

from dictator.speech.v1 import (
    alignment_pb2_grpc,
    artifacts_pb2_grpc,
    runtime_pb2_grpc,
    subtitle_pb2_grpc,
    transcription_pb2_grpc,
    voice_pb2_grpc,
)

from .alignment_service import AlignmentServiceServicer
from .artifact_service import ArtifactServiceServicer
from .base import BaseServicer
from .context import ServiceContext
from .runtime_service import RuntimeServiceServicer
from .subtitle_service import SubtitleServiceServicer
from .transcription_service import TranscriptionServiceServicer
from .voice_service import VoiceServiceServicer


def register_services(server: grpc.Server, service_context: ServiceContext) -> None:
    artifacts_pb2_grpc.add_ArtifactServiceServicer_to_server(
        ArtifactServiceServicer(service_context),
        server,
    )
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServiceServicer(service_context),
        server,
    )
    alignment_pb2_grpc.add_AlignmentServiceServicer_to_server(
        AlignmentServiceServicer(service_context),
        server,
    )
    subtitle_pb2_grpc.add_SubtitleServiceServicer_to_server(
        SubtitleServiceServicer(service_context),
        server,
    )
    voice_pb2_grpc.add_VoiceServiceServicer_to_server(
        VoiceServiceServicer(service_context),
        server,
    )
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(service_context),
        server,
    )


__all__ = [
    "AlignmentServiceServicer",
    "ArtifactServiceServicer",
    "BaseServicer",
    "RuntimeServiceServicer",
    "ServiceContext",
    "SubtitleServiceServicer",
    "TranscriptionServiceServicer",
    "VoiceServiceServicer",
    "register_services",
]
