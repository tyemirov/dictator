"""Diarization-focused gRPC client helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import grpc
from google.protobuf.json_format import MessageToDict

from dictator.speech.v1 import artifacts_pb2_grpc, transcription_pb2, transcription_pb2_grpc

from ._jobs import wait_for_job
from ._uploads import DEFAULT_CHUNK_BYTES, DEFAULT_MEDIA_TYPE, upload_audio_artifact
from .dictation import DictationClient


@dataclass(frozen=True)
class DiarizationResult:
    text: str
    language_code: str
    source_artifact_id: str
    diarization: dict[str, Any]
    diarization_artifact_id: str = ""


@dataclass(frozen=True)
class DiarizationJob:
    job_id: str
    state: str
    source_artifact_id: str = ""
    error_code: str = ""
    error_message: str = ""
    result: DiarizationResult | None = None
    created_at_unix_seconds: float = 0.0
    started_at_unix_seconds: float = 0.0
    finished_at_unix_seconds: float = 0.0


class DiarizationClient:
    """Upload audio and call DiarizeAudio in one step."""

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

    def diarize_file(
        self,
        audio_path: Path,
        *,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        include_words: bool = True,
        include_utterances: bool = True,
        include_speakers: bool = True,
        include_speaker_segments: bool = False,
        utterance_gap_seconds: float | None = None,
        persist_json_artifact: bool = False,
        media_type: str | None = None,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> DiarizationResult:
        payload = audio_path.read_bytes()
        return self.diarize_bytes(
            payload,
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            autodetect_language=autodetect_language,
            include_words=include_words,
            include_utterances=include_utterances,
            include_speakers=include_speakers,
            include_speaker_segments=include_speaker_segments,
            utterance_gap_seconds=utterance_gap_seconds,
            persist_json_artifact=persist_json_artifact,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def diarize_bytes(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        include_words: bool = True,
        include_utterances: bool = True,
        include_speakers: bool = True,
        include_speaker_segments: bool = False,
        utterance_gap_seconds: float | None = None,
        persist_json_artifact: bool = False,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> DiarizationResult:
        try:
            submitted = self.submit_diarize_bytes_job(
                payload,
                filename=filename,
                media_type=media_type,
                model_size=model_size,
                language_code=language_code,
                autodetect_language=autodetect_language,
                include_words=include_words,
                include_utterances=include_utterances,
                include_speakers=include_speakers,
                include_speaker_segments=include_speaker_segments,
                utterance_gap_seconds=utterance_gap_seconds,
                persist_json_artifact=persist_json_artifact,
            )
            finished = self.wait_for_diarization_job(
                submitted.job_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if finished.result is None:
                raise RuntimeError("diarization job succeeded without a result payload")
            return DiarizationResult(
                text=finished.result.text,
                language_code=finished.result.language_code,
                source_artifact_id=submitted.source_artifact_id,
                diarization=finished.result.diarization,
                diarization_artifact_id=finished.result.diarization_artifact_id,
            )
        except grpc.RpcError as error:
            if not self._should_fallback_to_sync(error):
                raise

        resolved_autodetect = DictationClient._resolve_autodetect(
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
        request = transcription_pb2.DiarizeAudioRequest(
            audio_artifact_id=artifact.artifact_id,
            language_code=language_code,
            model_size=model_size,
            include_words=include_words,
            include_utterances=include_utterances,
            include_speakers=include_speakers,
            include_speaker_segments=include_speaker_segments,
            persist_json_artifact=persist_json_artifact,
            autodetect_language=resolved_autodetect,
        )
        if utterance_gap_seconds is not None:
            request.utterance_gap_seconds = utterance_gap_seconds
        response = self._transcription_stub.DiarizeAudio(
            request,
            metadata=self._metadata,
        )
        return DiarizationResult(
            text=response.text,
            language_code=response.language_code,
            source_artifact_id=artifact.artifact_id,
            diarization=MessageToDict(
                response.diarization,
                preserving_proto_field_name=True,
            ),
            diarization_artifact_id=response.diarization_artifact_id,
        )

    def submit_diarize_file_job(
        self,
        audio_path: Path,
        *,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        include_words: bool = True,
        include_utterances: bool = True,
        include_speakers: bool = True,
        include_speaker_segments: bool = False,
        utterance_gap_seconds: float | None = None,
        persist_json_artifact: bool = False,
        media_type: str | None = None,
    ) -> DiarizationJob:
        return self.submit_diarize_bytes_job(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            autodetect_language=autodetect_language,
            include_words=include_words,
            include_utterances=include_utterances,
            include_speakers=include_speakers,
            include_speaker_segments=include_speaker_segments,
            utterance_gap_seconds=utterance_gap_seconds,
            persist_json_artifact=persist_json_artifact,
        )

    def submit_diarize_bytes_job(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        include_words: bool = True,
        include_utterances: bool = True,
        include_speakers: bool = True,
        include_speaker_segments: bool = False,
        utterance_gap_seconds: float | None = None,
        persist_json_artifact: bool = False,
    ) -> DiarizationJob:
        resolved_autodetect = DictationClient._resolve_autodetect(
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
        request = transcription_pb2.DiarizeAudioRequest(
            audio_artifact_id=artifact.artifact_id,
            language_code=language_code,
            model_size=model_size,
            include_words=include_words,
            include_utterances=include_utterances,
            include_speakers=include_speakers,
            include_speaker_segments=include_speaker_segments,
            persist_json_artifact=persist_json_artifact,
            autodetect_language=resolved_autodetect,
        )
        if utterance_gap_seconds is not None:
            request.utterance_gap_seconds = utterance_gap_seconds
        response = self._transcription_stub.SubmitDiarizeAudioJob(
            request,
            metadata=self._metadata,
        )
        return DiarizationJob(
            job_id=response.job_id,
            state=transcription_pb2.DiarizationJobState.Name(response.state),
            source_artifact_id=artifact.artifact_id,
        )

    def get_diarization_job(self, job_id: str) -> DiarizationJob:
        response = self._transcription_stub.GetDiarizeAudioJob(
            transcription_pb2.GetDiarizeAudioJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        result = None
        if response.state == transcription_pb2.DIARIZATION_JOB_STATE_SUCCEEDED:
            result = DiarizationResult(
                text=response.text,
                language_code=response.language_code,
                source_artifact_id="",
                diarization=MessageToDict(
                    response.diarization,
                    preserving_proto_field_name=True,
                ),
                diarization_artifact_id=response.diarization_artifact_id,
            )
        return DiarizationJob(
            job_id=response.job_id,
            state=transcription_pb2.DiarizationJobState.Name(response.state),
            error_code=response.error_code,
            error_message=response.error_message,
            result=result,
            created_at_unix_seconds=response.created_at_unix_seconds,
            started_at_unix_seconds=response.started_at_unix_seconds,
            finished_at_unix_seconds=response.finished_at_unix_seconds,
        )

    def wait_for_diarization_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> DiarizationJob:
        return wait_for_job(
            lambda: self.get_diarization_job(job_id),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    @staticmethod
    def _should_fallback_to_sync(error: grpc.RpcError) -> bool:
        if error.code() == grpc.StatusCode.UNIMPLEMENTED:
            return True
        if error.code() != grpc.StatusCode.INVALID_ARGUMENT:
            return False
        return (error.details() or "").endswith("job manager is not configured")
