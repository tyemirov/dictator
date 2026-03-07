"""Dictation-focused gRPC client helper for compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import grpc

from dictator.speech.v1 import artifacts_pb2, artifacts_pb2_grpc, transcription_pb2, transcription_pb2_grpc

_DEFAULT_MEDIA_TYPE = "application/octet-stream"
_DEFAULT_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DictationResult:
    text: str
    language_code: str
    artifact_id: str
    words: tuple[dict[str, float | str], ...] = ()

    def to_http_payload(self) -> dict[str, str]:
        return {"text": self.text}


class DictationClient:
    """Upload audio and call TranscriptionService in one step."""

    def __init__(
        self,
        channel: grpc.Channel,
        metadata: Sequence[tuple[str, str]] = (),
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    ) -> None:
        self._artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
        self._transcription_stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)
        self._metadata = tuple(metadata)
        self._chunk_bytes = chunk_bytes

    def dictate_file(
        self,
        audio_path: Path,
        model_size: str = "base",
        language_code: str = "",
        include_word_segments: bool = False,
        media_type: str | None = None,
    ) -> DictationResult:
        payload = audio_path.read_bytes()
        return self.dictate_bytes(
            payload,
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            include_word_segments=include_word_segments,
        )

    def dictate_bytes(
        self,
        payload: bytes,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "base",
        language_code: str = "",
        include_word_segments: bool = False,
    ) -> DictationResult:
        artifact = self._upload_audio(
            payload,
            filename=filename,
            media_type=media_type or _DEFAULT_MEDIA_TYPE,
        )
        response = self._transcription_stub.Transcribe(
            transcription_pb2.TranscribeRequest(
                audio_artifact_id=artifact.artifact_id,
                language_code=language_code,
                model_size=model_size,
                include_word_segments=include_word_segments,
            ),
            metadata=self._metadata,
        )
        words = tuple(
            {
                "content": word.content,
                "start": word.start_seconds,
                "end": word.end_seconds,
            }
            for word in response.words
        )
        return DictationResult(
            text=response.text,
            language_code=response.language_code,
            artifact_id=artifact.artifact_id,
            words=words,
        )

    def _upload_audio(
        self,
        payload: bytes,
        filename: str,
        media_type: str,
    ):
        def request_iter() -> Iterable[artifacts_pb2.UploadArtifactChunk]:
            yield artifacts_pb2.UploadArtifactChunk(
                metadata=artifacts_pb2.UploadArtifactMetadata(
                    filename=filename,
                    media_type=media_type,
                )
            )
            for index in range(0, len(payload), self._chunk_bytes):
                yield artifacts_pb2.UploadArtifactChunk(
                    content=payload[index : index + self._chunk_bytes]
                )

        response = self._artifact_stub.UploadArtifact(
            request_iter(),
            metadata=self._metadata,
        )
        return response.artifact
