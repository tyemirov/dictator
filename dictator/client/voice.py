"""Voice-reference extraction gRPC client helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import grpc

from dictator.speech.v1 import artifacts_pb2_grpc, common_pb2, voice_pb2, voice_pb2_grpc

from ._jobs import wait_for_job
from ._uploads import DEFAULT_CHUNK_BYTES, DEFAULT_MEDIA_TYPE, upload_audio_artifact


@dataclass(frozen=True)
class AudioFormat:
    container: str
    codec: str
    sample_rate_hz: int
    channel_count: int
    bit_depth: int


@dataclass(frozen=True)
class SynthesisResult:
    audio_artifact_id: str
    audio_duration_seconds: float
    timeline_artifact_id: str = ""
    chunk_count: int = 0
    resolved_audio_format: AudioFormat | None = None


@dataclass(frozen=True)
class SynthesisJob:
    job_id: str
    state: str
    error_code: str = ""
    error_message: str = ""
    result: SynthesisResult | None = None
    created_at_unix_seconds: float = 0.0
    started_at_unix_seconds: float = 0.0
    finished_at_unix_seconds: float = 0.0
    estimated_total_chunks: int = 0
    completed_chunks: int = 0


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


class SynthesisClient:
    """Synthesize speech through Dictator voice jobs."""

    def __init__(
        self,
        channel: grpc.Channel,
        metadata: Sequence[tuple[str, str]] = (),
    ) -> None:
        self._voice_stub = voice_pb2_grpc.VoiceServiceStub(channel)
        self._metadata = tuple(metadata)

    def synthesize(
        self,
        *,
        speaker_artifact_id: str = "",
        text: str = "",
        text_artifact_id: str = "",
        language_code: str = "",
        max_duration_seconds: float = 0.0,
        include_timeline: bool = False,
        synthesis_engine: int = voice_pb2.SYNTHESIS_ENGINE_UNSPECIFIED,
        speaker_transcript_text: str = "",
        preset_speaker: str = "",
        audio_format: common_pb2.AudioFormat | None = None,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> SynthesisResult:
        submitted = self.submit_synthesize_job(
            speaker_artifact_id=speaker_artifact_id,
            text=text,
            text_artifact_id=text_artifact_id,
            language_code=language_code,
            max_duration_seconds=max_duration_seconds,
            include_timeline=include_timeline,
            synthesis_engine=synthesis_engine,
            speaker_transcript_text=speaker_transcript_text,
            preset_speaker=preset_speaker,
            audio_format=audio_format,
        )
        finished = self.wait_for_synthesis_job(
            submitted.job_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        if finished.result is None:
            raise RuntimeError("synthesis job succeeded without a result payload")
        return finished.result

    def submit_synthesize_job(
        self,
        *,
        speaker_artifact_id: str = "",
        text: str = "",
        text_artifact_id: str = "",
        language_code: str = "",
        max_duration_seconds: float = 0.0,
        include_timeline: bool = False,
        synthesis_engine: int = voice_pb2.SYNTHESIS_ENGINE_UNSPECIFIED,
        speaker_transcript_text: str = "",
        preset_speaker: str = "",
        audio_format: common_pb2.AudioFormat | None = None,
    ) -> SynthesisJob:
        request = voice_pb2.SynthesizeSpeechRequest(
            speaker_artifact_id=speaker_artifact_id,
            language_code=language_code,
            max_duration_seconds=max_duration_seconds,
            include_timeline=include_timeline,
            synthesis_engine=synthesis_engine,
            speaker_transcript_text=speaker_transcript_text,
            preset_speaker=preset_speaker,
        )
        if audio_format is not None:
            request.audio_format.CopyFrom(audio_format)
        if text_artifact_id:
            request.text_artifact_id = text_artifact_id
        else:
            request.text = text
        response = self._voice_stub.SubmitSynthesizeSpeechJob(
            request,
            metadata=self._metadata,
        )
        return SynthesisJob(
            job_id=response.job_id,
            state=voice_pb2.SynthesisJobState.Name(response.state),
        )

    def get_synthesis_job(self, job_id: str) -> SynthesisJob:
        response = self._voice_stub.GetSynthesizeSpeechJob(
            voice_pb2.GetSynthesizeSpeechJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        result = None
        if response.state == voice_pb2.SYNTHESIS_JOB_STATE_SUCCEEDED:
            result = SynthesisResult(
                audio_artifact_id=response.audio_artifact.artifact_id,
                audio_duration_seconds=response.audio_duration_seconds,
                resolved_audio_format=_audio_format_from_pb(response.resolved_audio_format),
                timeline_artifact_id=response.timeline_artifact_id,
                chunk_count=response.chunk_count,
            )
        return SynthesisJob(
            job_id=response.job_id,
            state=voice_pb2.SynthesisJobState.Name(response.state),
            error_code=response.error_code,
            error_message=response.error_message,
            result=result,
            created_at_unix_seconds=response.created_at_unix_seconds,
            started_at_unix_seconds=response.started_at_unix_seconds,
            finished_at_unix_seconds=response.finished_at_unix_seconds,
            estimated_total_chunks=response.estimated_total_chunks,
            completed_chunks=response.completed_chunks,
        )

    def cancel_synthesis_job(self, job_id: str) -> SynthesisJob:
        response = self._voice_stub.CancelSynthesizeSpeechJob(
            voice_pb2.CancelSynthesizeSpeechJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        return SynthesisJob(
            job_id=response.job_id,
            state=voice_pb2.SynthesisJobState.Name(response.state),
        )

    def wait_for_synthesis_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> SynthesisJob:
        return wait_for_job(
            lambda: self.get_synthesis_job(job_id),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )


def _audio_format_from_pb(payload: common_pb2.AudioFormat) -> AudioFormat | None:
    if (
        payload.container == common_pb2.AUDIO_CONTAINER_UNSPECIFIED
        and payload.codec == common_pb2.AUDIO_CODEC_UNSPECIFIED
        and payload.sample_rate_hz == 0
        and payload.channel_count == 0
        and payload.bit_depth == 0
    ):
        return None
    container = {
        common_pb2.AUDIO_CONTAINER_WAV: "wav",
    }.get(payload.container, _unknown_enum_name("audio_container", payload.container))
    codec = {
        common_pb2.AUDIO_CODEC_PCM_S16LE: "pcm_s16le",
    }.get(payload.codec, _unknown_enum_name("audio_codec", payload.codec))
    return AudioFormat(
        container=container,
        codec=codec,
        sample_rate_hz=payload.sample_rate_hz,
        channel_count=payload.channel_count,
        bit_depth=payload.bit_depth,
    )


def _unknown_enum_name(prefix: str, value: int) -> str:
    return f"unknown_{prefix}_{value}"


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
        timeout_seconds: float | None = None,
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
        timeout_seconds: float | None = None,
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
        source_artifact_id = getattr(response, "source_artifact_id", "")
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
            source_artifact_id=source_artifact_id,
            error_code=response.error_code,
            error_message=response.error_message,
            result=result,
            created_at_unix_seconds=response.created_at_unix_seconds,
            started_at_unix_seconds=response.started_at_unix_seconds,
            finished_at_unix_seconds=response.finished_at_unix_seconds,
        )

    def cancel_reference_sample_job(self, job_id: str) -> ReferenceSampleJob:
        response = self._voice_stub.CancelExtractReferenceSampleJob(
            voice_pb2.CancelExtractReferenceSampleJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        return ReferenceSampleJob(
            job_id=response.job_id,
            state=voice_pb2.ExtractReferenceSampleJobState.Name(response.state),
        )

    def wait_for_reference_sample_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> ReferenceSampleJob:
        return wait_for_job(
            lambda: self.get_reference_sample_job(job_id),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
