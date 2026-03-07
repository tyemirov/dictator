"""Subtitle-focused gRPC client helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import grpc

from dictator.speech.v1 import artifacts_pb2_grpc, subtitle_pb2, subtitle_pb2_grpc

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
    ) -> SubtitleResult:
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
