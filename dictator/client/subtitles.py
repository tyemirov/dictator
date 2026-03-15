"""Subtitle-focused gRPC client helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import grpc

from dictator.speech.v1 import artifacts_pb2_grpc, subtitle_pb2, subtitle_pb2_grpc

from ._jobs import wait_for_job
from ._uploads import DEFAULT_CHUNK_BYTES, DEFAULT_MEDIA_TYPE, upload_audio_artifact
from .dictation import DictationClient


@dataclass(frozen=True)
class SubtitleResult:
    language_code: str
    mode: str
    granularity: str
    group_size: int
    source_artifact_id: str
    srt_artifact_id: str
    srt_text: str
    cues: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class SubtitleJob:
    job_id: str
    state: str
    source_artifact_id: str = ""
    error_code: str = ""
    error_message: str = ""
    result: SubtitleResult | None = None
    created_at_unix_seconds: float = 0.0
    started_at_unix_seconds: float = 0.0
    finished_at_unix_seconds: float = 0.0


class SubtitleClient:
    """Upload audio and call SubtitleService in one step."""

    def __init__(
        self,
        channel: grpc.Channel,
        metadata: Sequence[tuple[str, str]] = (),
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self._artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
        self._subtitle_stub = subtitle_pb2_grpc.SubtitleServiceStub(channel)
        self._metadata = tuple(metadata)
        self._chunk_bytes = chunk_bytes

    def render_file(
        self,
        audio_path: Path,
        *,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        granularity: str = "words",
        group_size: int = 1,
        source_text: str | None = None,
        source_text_file: Path | None = None,
        source_text_name: str = "",
        include_srt_text: bool = True,
        media_type: str | None = None,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> SubtitleResult:
        inline_source_text, resolved_source_name = self._resolve_source_text(
            source_text=source_text,
            source_text_file=source_text_file,
            source_text_name=source_text_name,
        )
        return self.render_bytes(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            autodetect_language=autodetect_language,
            granularity=granularity,
            group_size=group_size,
            source_text=inline_source_text,
            source_text_name=resolved_source_name,
            include_srt_text=include_srt_text,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def render_bytes(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        granularity: str = "words",
        group_size: int = 1,
        source_text: str | None = None,
        source_text_name: str = "",
        include_srt_text: bool = True,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> SubtitleResult:
        try:
            submitted = self.submit_render_bytes_job(
                payload,
                filename=filename,
                media_type=media_type,
                model_size=model_size,
                language_code=language_code,
                autodetect_language=autodetect_language,
                granularity=granularity,
                group_size=group_size,
                source_text=source_text,
                source_text_name=source_text_name,
                include_srt_text=include_srt_text,
            )
            finished = self.wait_for_subtitle_job(
                submitted.job_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if finished.result is None:
                raise RuntimeError("subtitle job succeeded without a result payload")
            return SubtitleResult(
                language_code=finished.result.language_code,
                mode=finished.result.mode,
                granularity=finished.result.granularity,
                group_size=finished.result.group_size,
                source_artifact_id=submitted.source_artifact_id,
                srt_artifact_id=finished.result.srt_artifact_id,
                srt_text=finished.result.srt_text,
                cues=finished.result.cues,
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
        request = subtitle_pb2.RenderSubtitlesRequest(
            audio_artifact_id=artifact.artifact_id,
            language_code=language_code,
            autodetect_language=resolved_autodetect,
            model_size=model_size,
            output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
            granularity=self._resolve_granularity(granularity),
            group_size=group_size,
            source_text_name=source_text_name,
            include_srt_text=include_srt_text,
        )
        if source_text is not None:
            request.source_text = source_text
        response = self._subtitle_stub.RenderSubtitles(
            request,
            metadata=self._metadata,
        )
        return SubtitleResult(
            language_code=response.language_code,
            mode=self._resolve_mode(response.mode),
            granularity=granularity,
            group_size=response.group_size,
            source_artifact_id=artifact.artifact_id,
            srt_artifact_id=response.srt_artifact_id,
            srt_text=response.srt_text,
            cues=tuple(
                {
                    "content": cue.content,
                    "start": cue.start_seconds,
                    "end": cue.end_seconds,
                    "itemCount": cue.item_count,
                }
                for cue in response.cues
            ),
        )

    def submit_render_file_job(
        self,
        audio_path: Path,
        *,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        granularity: str = "words",
        group_size: int = 1,
        source_text: str | None = None,
        source_text_file: Path | None = None,
        source_text_name: str = "",
        include_srt_text: bool = True,
        media_type: str | None = None,
    ) -> SubtitleJob:
        inline_source_text, resolved_source_name = self._resolve_source_text(
            source_text=source_text,
            source_text_file=source_text_file,
            source_text_name=source_text_name,
        )
        return self.submit_render_bytes_job(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            model_size=model_size,
            language_code=language_code,
            autodetect_language=autodetect_language,
            granularity=granularity,
            group_size=group_size,
            source_text=inline_source_text,
            source_text_name=resolved_source_name,
            include_srt_text=include_srt_text,
        )

    def submit_render_bytes_job(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        model_size: str = "base",
        language_code: str = "",
        autodetect_language: bool | None = None,
        granularity: str = "words",
        group_size: int = 1,
        source_text: str | None = None,
        source_text_name: str = "",
        include_srt_text: bool = True,
    ) -> SubtitleJob:
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
        request = subtitle_pb2.RenderSubtitlesRequest(
            audio_artifact_id=artifact.artifact_id,
            language_code=language_code,
            autodetect_language=resolved_autodetect,
            model_size=model_size,
            output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
            granularity=self._resolve_granularity(granularity),
            group_size=group_size,
            source_text_name=source_text_name,
            include_srt_text=include_srt_text,
        )
        if source_text is not None:
            request.source_text = source_text
        response = self._subtitle_stub.SubmitRenderSubtitlesJob(
            request,
            metadata=self._metadata,
        )
        return SubtitleJob(
            job_id=response.job_id,
            state=subtitle_pb2.SubtitleJobState.Name(response.state),
            source_artifact_id=artifact.artifact_id,
        )

    def get_subtitle_job(self, job_id: str) -> SubtitleJob:
        response = self._subtitle_stub.GetRenderSubtitlesJob(
            subtitle_pb2.GetRenderSubtitlesJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        result = None
        if response.state == subtitle_pb2.SUBTITLE_JOB_STATE_SUCCEEDED:
            result = SubtitleResult(
                language_code=response.language_code,
                mode=self._resolve_mode(response.mode),
                granularity=self._resolve_granularity_name(response.granularity),
                group_size=response.group_size,
                source_artifact_id="",
                srt_artifact_id=response.srt_artifact_id,
                srt_text=response.srt_text,
                cues=tuple(
                    {
                        "content": cue.content,
                        "start": cue.start_seconds,
                        "end": cue.end_seconds,
                        "itemCount": cue.item_count,
                    }
                    for cue in response.cues
                ),
            )
        return SubtitleJob(
            job_id=response.job_id,
            state=subtitle_pb2.SubtitleJobState.Name(response.state),
            error_code=response.error_code,
            error_message=response.error_message,
            result=result,
            created_at_unix_seconds=response.created_at_unix_seconds,
            started_at_unix_seconds=response.started_at_unix_seconds,
            finished_at_unix_seconds=response.finished_at_unix_seconds,
        )

    def wait_for_subtitle_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> SubtitleJob:
        return wait_for_job(
            lambda: self.get_subtitle_job(job_id),
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

    @staticmethod
    def _resolve_granularity(granularity: str) -> int:
        normalized = granularity.strip().lower()
        if normalized == "words":
            return subtitle_pb2.SUBTITLE_GRANULARITY_WORDS
        if normalized == "sentences":
            return subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES
        raise ValueError("granularity must be 'words' or 'sentences'")

    @staticmethod
    def _resolve_mode(mode: int) -> str:
        if mode == subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT:
            return "forced_alignment"
        return "transcription"

    @staticmethod
    def _resolve_granularity_name(granularity: int) -> str:
        if granularity == subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES:
            return "sentences"
        return "words"

    @staticmethod
    def _resolve_source_text(
        *,
        source_text: str | None,
        source_text_file: Path | None,
        source_text_name: str,
    ) -> tuple[str | None, str]:
        if source_text is not None and source_text_file is not None:
            raise ValueError("source_text and source_text_file cannot both be set")
        if source_text_file is not None:
            return source_text_file.read_text(encoding="utf-8"), source_text_name or source_text_file.name
        return source_text, source_text_name or "transcript.txt"
