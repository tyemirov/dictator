"""Dictation-focused gRPC client helper for compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import grpc

from dictator.speech.v1 import artifacts_pb2_grpc, transcription_pb2, transcription_pb2_grpc

from ._uploads import DEFAULT_CHUNK_BYTES, DEFAULT_MEDIA_TYPE, upload_audio_artifact


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
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
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
        autodetect_language: bool | None = None,
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
            autodetect_language=autodetect_language,
            include_word_segments=include_word_segments,
        )

    def dictate_bytes(
        self,
        payload: bytes,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        include_word_segments: bool = False,
    ) -> DictationResult:
        resolved_autodetect = self._resolve_autodetect(
            language_code=language_code,
            autodetect_language=autodetect_language,
        )
        artifact = upload_audio_artifact(
            self._artifact_stub,
            metadata=self._metadata,
            chunk_bytes=self._chunk_bytes,
            payload=payload,
            filename=filename,
            media_type=media_type or DEFAULT_MEDIA_TYPE,
        )
        response = self._transcription_stub.Transcribe(
            transcription_pb2.TranscribeRequest(
                audio_artifact_id=artifact.artifact_id,
                language_code=language_code,
                model_size=model_size,
                include_word_segments=include_word_segments,
                autodetect_language=resolved_autodetect,
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

    @staticmethod
    def _resolve_autodetect(
        *,
        language_code: str,
        autodetect_language: bool | None,
    ) -> bool:
        normalized = language_code.strip()
        if autodetect_language is None:
            autodetect_language = not bool(normalized)
        if normalized and autodetect_language:
            raise ValueError("language_code and autodetect_language cannot both be set")
        if not normalized and not autodetect_language:
            raise ValueError("set language_code or autodetect_language=True")
        return autodetect_language
