"""Dictation-focused gRPC client helper for compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import grpc

from dictator.speech.v1 import artifacts_pb2_grpc, transcription_pb2, transcription_pb2_grpc

from ._jobs import wait_for_job
from ._uploads import DEFAULT_CHUNK_BYTES, DEFAULT_MEDIA_TYPE, upload_audio_artifact


@dataclass(frozen=True)
class DictationResult:
    text: str
    language_code: str
    artifact_id: str
    words: tuple[dict[str, float | str], ...] = ()

    def to_http_payload(self) -> dict[str, str]:
        return {"text": self.text}


@dataclass(frozen=True)
class DictationJob:
    job_id: str
    state: str
    source_artifact_id: str = ""
    error_code: str = ""
    error_message: str = ""
    result: DictationResult | None = None
    created_at_unix_seconds: float = 0.0
    started_at_unix_seconds: float = 0.0
    finished_at_unix_seconds: float = 0.0


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
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
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
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
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
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> DictationResult:
        try:
            submitted = self.submit_dictate_bytes_job(
                payload,
                filename=filename,
                media_type=media_type,
                model_size=model_size,
                language_code=language_code,
                autodetect_language=autodetect_language,
                include_word_segments=include_word_segments,
            )
            finished = self.wait_for_dictation_job(
                submitted.job_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if finished.result is None:
                raise RuntimeError("dictation job succeeded without a result payload")
            return DictationResult(
                text=finished.result.text,
                language_code=finished.result.language_code,
                artifact_id=submitted.source_artifact_id,
                words=finished.result.words,
            )
        except grpc.RpcError as error:
            if not self._should_fallback_to_sync(error):
                raise

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
        return DictationResult(
            text=response.text,
            language_code=response.language_code,
            artifact_id=artifact.artifact_id,
            words=tuple(
                {
                    "content": word.content,
                    "start": word.start_seconds,
                    "end": word.end_seconds,
                }
                for word in response.words
            ),
        )

    def submit_dictate_file_job(
        self,
        audio_path: Path,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        include_word_segments: bool = False,
        media_type: str | None = None,
    ) -> DictationJob:
        return self.submit_dictate_bytes_job(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            autodetect_language=autodetect_language,
            include_word_segments=include_word_segments,
        )

    def submit_dictate_bytes_job(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        include_word_segments: bool = False,
    ) -> DictationJob:
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
        response = self._transcription_stub.SubmitTranscribeJob(
            transcription_pb2.TranscribeRequest(
                audio_artifact_id=artifact.artifact_id,
                language_code=language_code,
                model_size=model_size,
                include_word_segments=include_word_segments,
                autodetect_language=resolved_autodetect,
            ),
            metadata=self._metadata,
        )
        return DictationJob(
            job_id=response.job_id,
            state=transcription_pb2.TranscriptionJobState.Name(response.state),
            source_artifact_id=artifact.artifact_id,
        )

    def get_dictation_job(self, job_id: str) -> DictationJob:
        response = self._transcription_stub.GetTranscribeJob(
            transcription_pb2.GetTranscribeJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        result = None
        if response.state == transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED:
            result = DictationResult(
                text=response.text,
                language_code=response.language_code,
                artifact_id="",
                words=tuple(
                    {
                        "content": word.content,
                        "start": word.start_seconds,
                        "end": word.end_seconds,
                    }
                    for word in response.words
                ),
            )
        return DictationJob(
            job_id=response.job_id,
            state=transcription_pb2.TranscriptionJobState.Name(response.state),
            error_code=response.error_code,
            error_message=response.error_message,
            result=result,
            created_at_unix_seconds=response.created_at_unix_seconds,
            started_at_unix_seconds=response.started_at_unix_seconds,
            finished_at_unix_seconds=response.finished_at_unix_seconds,
        )

    def wait_for_dictation_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> DictationJob:
        return wait_for_job(
            lambda: self.get_dictation_job(job_id),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
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

    @staticmethod
    def _should_fallback_to_sync(error: grpc.RpcError) -> bool:
        if error.code() == grpc.StatusCode.UNIMPLEMENTED:
            return True
        if error.code() != grpc.StatusCode.INVALID_ARGUMENT:
            return False
        return (error.details() or "").endswith("job manager is not configured")
