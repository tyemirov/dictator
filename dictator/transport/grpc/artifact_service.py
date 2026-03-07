"""Artifact gRPC servicer."""

from __future__ import annotations

from dictator.runtime import ValidationError
from dictator.speech.v1 import artifacts_pb2, artifacts_pb2_grpc

from .base import BaseServicer


class ArtifactServiceServicer(BaseServicer, artifacts_pb2_grpc.ArtifactServiceServicer):
    def UploadArtifact(self, request_iterator, context):
        with self._request_scope(context):
            payload_size = 0
            reservation = None
            finalised = False
            iterator = iter(request_iterator)
            try:
                first_chunk = next(iterator)
            except StopIteration as exc:
                raise ValidationError(
                    "dictator.grpc.artifact.empty_upload",
                    "upload stream is empty",
                ) from exc
            if first_chunk.WhichOneof("payload") != "metadata":
                raise ValidationError(
                    "dictator.grpc.artifact.missing_metadata",
                    "first upload chunk must contain metadata",
                )
            metadata_message = first_chunk.metadata
            reservation = self.service_context.artifact_store.reserve_artifact(
                metadata_message.filename,
                media_type=metadata_message.media_type,
            )
            try:
                with reservation.path.open("wb") as handle:
                    for chunk in iterator:
                        payload_type = chunk.WhichOneof("payload")
                        if payload_type != "content":
                            raise ValidationError(
                                "dictator.grpc.artifact.invalid_chunk",
                                "upload content chunks cannot contain metadata",
                            )
                        handle.write(chunk.content)
                        payload_size += len(chunk.content)
                if payload_size:
                    self.service_context.metrics.record_bytes(payload_size)
                record = self.service_context.artifact_store.finalize_artifact(reservation)
                finalised = True
                return artifacts_pb2.UploadArtifactResponse(artifact=self._artifact_ref(record))
            finally:
                if reservation is not None and not finalised:
                    self.service_context.artifact_store.discard_reservation(reservation)

    def DownloadArtifact(self, request, context):
        with self._request_scope(context):
            chunk_size = request.chunk_size or self.service_context.download_chunk_bytes
            if chunk_size <= 0:
                raise ValidationError(
                    "dictator.grpc.artifact.invalid_chunk_size",
                    "chunk_size must be positive",
                )
            for record, offset, payload, eof in self.service_context.artifact_store.iter_artifact_chunks(
                request.artifact_id,
                chunk_size=chunk_size,
            ):
                yield artifacts_pb2.DownloadArtifactChunk(
                    artifact=self._artifact_ref(record),
                    content=payload,
                    offset=offset,
                    eof=eof,
                )
