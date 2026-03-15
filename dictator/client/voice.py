"""Voice-reference extraction gRPC client helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import grpc

from dictator.speech.v1 import artifacts_pb2_grpc, voice_pb2, voice_pb2_grpc

from ._jobs import wait_for_job
from ._uploads import DEFAULT_CHUNK_BYTES, DEFAULT_MEDIA_TYPE, upload_audio_artifact


@dataclass(frozen=True)
class ReferenceSampleResult:
    sample_artifact_id: str
    trim_start_seconds: float
    trim_end_seconds: float
    window_start_seconds: float
    window_end_seconds: float
    dominant_speaker_word_count: int


@dataclass(frozen=True)
class ReferenceSampleJob:
    job_id: str
    state: str
    source_artifact_id: str = ""
    error_code: str = ""
    error_message: str = ""
    result: ReferenceSampleResult | None = None
    created_at_unix_seconds: float = 0.0
    started_at_unix_seconds: float = 0.0
    finished_at_unix_seconds: float = 0.0


class ReferenceSampleClient:
    """Upload audio and call voice reference extraction jobs."""

    def __init__(
        self,
        channel: grpc.Channel,
        metadata: Sequence[tuple[str, str]] = (),
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self._artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
        self._voice_stub = voice_pb2_grpc.VoiceServiceStub(channel)
        self._metadata = tuple(metadata)
        self._chunk_bytes = chunk_bytes

    def extract_file(
        self,
        audio_path: Path,
        *,
        model_size: str = "medium",
        language_code: str = "",
        duration_seconds: float = 20.0,
        max_speech_rate: float = 4.0,
        min_centroid_hz: float = 500.0,
        max_centroid_hz: float = 4000.0,
        media_type: str | None = None,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> ReferenceSampleResult:
        return self.extract_bytes(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            duration_seconds=duration_seconds,
            max_speech_rate=max_speech_rate,
            min_centroid_hz=min_centroid_hz,
            max_centroid_hz=max_centroid_hz,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def extract_bytes(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "medium",
        language_code: str = "",
        duration_seconds: float = 20.0,
        max_speech_rate: float = 4.0,
        min_centroid_hz: float = 500.0,
        max_centroid_hz: float = 4000.0,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> ReferenceSampleResult:
        submitted = self.submit_extract_bytes_job(
            payload,
            filename=filename,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            duration_seconds=duration_seconds,
            max_speech_rate=max_speech_rate,
            min_centroid_hz=min_centroid_hz,
            max_centroid_hz=max_centroid_hz,
        )
        finished = self.wait_for_reference_sample_job(
            submitted.job_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        if finished.result is None:
            raise RuntimeError("reference sample job succeeded without a result payload")
        return finished.result

    def submit_extract_file_job(
        self,
        audio_path: Path,
        *,
        model_size: str = "medium",
        language_code: str = "",
        duration_seconds: float = 20.0,
        max_speech_rate: float = 4.0,
        min_centroid_hz: float = 500.0,
        max_centroid_hz: float = 4000.0,
        media_type: str | None = None,
    ) -> ReferenceSampleJob:
        return self.submit_extract_bytes_job(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            duration_seconds=duration_seconds,
            max_speech_rate=max_speech_rate,
            min_centroid_hz=min_centroid_hz,
            max_centroid_hz=max_centroid_hz,
        )

    def submit_extract_bytes_job(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "medium",
        language_code: str = "",
        duration_seconds: float = 20.0,
        max_speech_rate: float = 4.0,
        min_centroid_hz: float = 500.0,
        max_centroid_hz: float = 4000.0,
    ) -> ReferenceSampleJob:
        artifact = upload_audio_artifact(
            self._artifact_stub,
            metadata=self._metadata,
            chunk_bytes=self._chunk_bytes,
            payload=payload,
            filename=filename,
            media_type=media_type or DEFAULT_MEDIA_TYPE,
        )
        response = self._voice_stub.SubmitExtractReferenceSampleJob(
            voice_pb2.ExtractReferenceSampleRequest(
                source_artifact_id=artifact.artifact_id,
                model_size=model_size,
                language_code=language_code,
                duration_seconds=duration_seconds,
                max_speech_rate=max_speech_rate,
                min_centroid_hz=min_centroid_hz,
                max_centroid_hz=max_centroid_hz,
            ),
            metadata=self._metadata,
        )
        return ReferenceSampleJob(
            job_id=response.job_id,
            state=voice_pb2.ExtractReferenceSampleJobState.Name(response.state),
            source_artifact_id=artifact.artifact_id,
        )

    def get_reference_sample_job(self, job_id: str) -> ReferenceSampleJob:
        response = self._voice_stub.GetExtractReferenceSampleJob(
            voice_pb2.GetExtractReferenceSampleJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        result = None
        if response.state == voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_SUCCEEDED:
            result = ReferenceSampleResult(
                sample_artifact_id=response.sample_artifact.artifact_id,
                trim_start_seconds=response.trim_start_seconds,
                trim_end_seconds=response.trim_end_seconds,
                window_start_seconds=response.window_start_seconds,
                window_end_seconds=response.window_end_seconds,
                dominant_speaker_word_count=response.dominant_speaker_word_count,
            )
        return ReferenceSampleJob(
            job_id=response.job_id,
            state=voice_pb2.ExtractReferenceSampleJobState.Name(response.state),
            error_code=response.error_code,
            error_message=response.error_message,
            result=result,
            created_at_unix_seconds=response.created_at_unix_seconds,
            started_at_unix_seconds=response.started_at_unix_seconds,
            finished_at_unix_seconds=response.finished_at_unix_seconds,
        )

    def wait_for_reference_sample_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> ReferenceSampleJob:
        return wait_for_job(
            lambda: self.get_reference_sample_job(job_id),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
