"""Shared artifact upload helpers for Dictator gRPC clients."""

from __future__ import annotations

from typing import Iterable, Sequence

from dictator.speech.v1 import artifacts_pb2, artifacts_pb2_grpc

DEFAULT_MEDIA_TYPE = "application/octet-stream"
DEFAULT_CHUNK_BYTES = 1024 * 1024


def upload_audio_artifact(
    artifact_stub: artifacts_pb2_grpc.ArtifactServiceStub,
    *,
    metadata: Sequence[tuple[str, str]],
    chunk_bytes: int,
    payload: bytes,
    filename: str,
    media_type: str,
):
    """Upload audio bytes and return the resulting artifact reference."""

    def request_iter() -> Iterable[artifacts_pb2.UploadArtifactChunk]:
        yield artifacts_pb2.UploadArtifactChunk(
            metadata=artifacts_pb2.UploadArtifactMetadata(
                filename=filename,
                media_type=media_type,
            )
        )
        for index in range(0, len(payload), chunk_bytes):
            yield artifacts_pb2.UploadArtifactChunk(
                content=payload[index : index + chunk_bytes]
            )

    response = artifact_stub.UploadArtifact(
        request_iter(),
        metadata=metadata,
    )
    return response.artifact
