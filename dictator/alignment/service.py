"""Alignment application service."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from .models import AlignTranscriptRequest, AlignTranscriptResult, AlignedWord
from .srt import build_srt
from .text import detect_default_language, normalize_language_value, normalize_transcript_for_alignment
from .whisperx_backend import WhisperXAlignmentBackend


class AlignmentBackend(Protocol):
    """Backend contract so transports can swap implementations."""

    def align(
        self,
        audio_path: Path,
        transcript_text: str,
        language: str,
        device: str = "auto",
        remove_punctuation: bool = False,
    ) -> Sequence[AlignedWord]:
        ...


class AlignmentService:
    """Service layer for transcript-to-audio forced alignment."""

    def __init__(self, backend: AlignmentBackend | None = None) -> None:
        self.backend = backend or WhisperXAlignmentBackend()

    def align(self, request: AlignTranscriptRequest) -> AlignTranscriptResult:
        normalized_transcript = normalize_transcript_for_alignment(
            request.transcript_text,
            request.transcript_source_name,
            request.remove_punctuation,
        )
        language = normalize_language_value(
            request.language,
            detect_default_language(normalized_transcript),
        )
        words = tuple(
            self.backend.align(
                audio_path=request.audio_path,
                transcript_text=normalized_transcript,
                language=language,
                device=request.device,
                remove_punctuation=request.remove_punctuation,
            )
        )
        srt_text = build_srt(words)
        if request.output_srt_path is not None:
            request.output_srt_path.write_text(srt_text, encoding="utf-8")
        return AlignTranscriptResult(
            audio_path=request.audio_path,
            language=language,
            words=words,
            srt_text=srt_text,
            output_srt_path=request.output_srt_path,
        )


def align_transcript(request: AlignTranscriptRequest) -> AlignTranscriptResult:
    """Convenience wrapper for one-shot alignment."""
    return AlignmentService().align(request)
