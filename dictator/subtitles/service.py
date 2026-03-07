"""Application service for grouped subtitle rendering."""

from __future__ import annotations

import re
from typing import Iterable, Protocol, Sequence

from dictator.alignment import AlignTranscriptRequest, AlignmentService
from dictator.alignment.srt import format_srt_timestamp, srt_timestamp_from_seconds
from dictator.runtime import ProcessingError, ValidationError
from dictator.transcription import TranscriptionService, TranscriptionResult, WordSegment

from .models import RenderSubtitlesRequest, RenderSubtitlesResult, SubtitleCue, TimedWord

_SENTENCE_END_PATTERN = re.compile(r"[.!?][\"')\]]*$")


class _AlignmentProtocol(Protocol):
    def align(self, request: AlignTranscriptRequest):
        ...


class _TranscriptionProtocol(Protocol):
    def transcribe(
        self,
        audio,
        language: str | None = None,
        model_size: str = "base",
        model: object | None = None,
        progress_cb=None,
    ) -> TranscriptionResult:
        ...


def _coerce_word_bounds(text: str, start_value: float | None, end_value: float | None) -> TimedWord:
    if not text.strip():
        raise ProcessingError(
            "dictator.subtitles.word_empty",
            "subtitle words must include text",
        )
    if start_value is None and end_value is None:
        raise ProcessingError(
            "dictator.subtitles.word_missing_timestamps",
            "subtitle words require timestamps",
        )
    start_seconds = float(start_value if start_value is not None else end_value)
    end_seconds = float(end_value if end_value is not None else start_seconds)
    if end_seconds < start_seconds:
        end_seconds = start_seconds
    return TimedWord(text=text.strip(), start_seconds=start_seconds, end_seconds=end_seconds)


def words_from_transcription(words: Iterable[WordSegment]) -> tuple[TimedWord, ...]:
    """Normalize transcription words into timed subtitle tokens."""
    normalized: list[TimedWord] = []
    for word in words:
        if not word.text.strip():
            continue
        normalized.append(
            _coerce_word_bounds(
                word.text,
                word.start_seconds,
                word.end_seconds,
            )
        )
    return tuple(normalized)


def words_from_alignment(words: Iterable[object]) -> tuple[TimedWord, ...]:
    """Normalize aligned words into timed subtitle tokens."""
    normalized: list[TimedWord] = []
    for word in words:
        text = getattr(word, "text", "")
        if not str(text).strip():
            continue
        normalized.append(
            _coerce_word_bounds(
                str(text),
                getattr(word, "start_seconds", None),
                getattr(word, "end_seconds", None),
            )
        )
    return tuple(normalized)


def sentence_units(words: Sequence[TimedWord]) -> tuple[TimedWord, ...]:
    """Group timed words into heuristic timed sentence units."""
    if not words:
        return ()
    sentences: list[TimedWord] = []
    current_words: list[TimedWord] = []
    for word in words:
        current_words.append(word)
        if _SENTENCE_END_PATTERN.search(word.text):
            sentences.append(_collapse_words(current_words))
            current_words = []
    if current_words:
        sentences.append(_collapse_words(current_words))
    return tuple(sentences)


def grouped_cues(units: Sequence[TimedWord], group_size: int) -> tuple[SubtitleCue, ...]:
    """Combine timed units into fixed-size subtitle cues."""
    if group_size <= 0:
        raise ValidationError(
            "dictator.subtitles.invalid_group_size",
            "group_size must be positive",
        )
    cues: list[SubtitleCue] = []
    for index, offset in enumerate(range(0, len(units), group_size), start=1):
        batch = units[offset : offset + group_size]
        if not batch:
            continue
        cues.append(
            SubtitleCue(
                index=index,
                text=" ".join(item.text for item in batch if item.text).strip(),
                start_seconds=batch[0].start_seconds,
                end_seconds=batch[-1].end_seconds,
                item_count=len(batch),
            )
        )
    return tuple(cues)


def render_srt(cues: Sequence[SubtitleCue]) -> str:
    """Render subtitle cues as SubRip text."""
    lines: list[str] = []
    for cue in cues:
        start_ms = srt_timestamp_from_seconds(cue.start_seconds, "floor")
        end_ms = srt_timestamp_from_seconds(cue.end_seconds, "ceil")
        lines.append(str(cue.index))
        lines.append(
            f"{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}"
        )
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if cues else "")


def _collapse_words(words: Sequence[TimedWord]) -> TimedWord:
    return TimedWord(
        text=" ".join(word.text for word in words if word.text).strip(),
        start_seconds=words[0].start_seconds,
        end_seconds=words[-1].end_seconds,
    )


class SubtitleService:
    """High-level service that chooses transcription or alignment before SRT rendering."""

    def __init__(
        self,
        transcription_service: _TranscriptionProtocol | None = None,
        alignment_service: _AlignmentProtocol | None = None,
    ) -> None:
        self.transcription_service = transcription_service or TranscriptionService()
        self.alignment_service = alignment_service or AlignmentService()

    def render(
        self,
        request: RenderSubtitlesRequest,
        *,
        model: object | None = None,
    ) -> RenderSubtitlesResult:
        if request.output_format != "srt":
            raise ValidationError(
                "dictator.subtitles.unsupported_format",
                "only SRT output is supported",
            )
        if request.granularity not in {"words", "sentences"}:
            raise ValidationError(
                "dictator.subtitles.invalid_granularity",
                "granularity must be 'words' or 'sentences'",
            )
        if request.group_size <= 0:
            raise ValidationError(
                "dictator.subtitles.invalid_group_size",
                "group_size must be positive",
            )

        if request.source_text is not None:
            language = request.language
            if language is None:
                detection = self.transcription_service.transcribe(
                    request.audio_path,
                    language=None,
                    model_size=request.model_size,
                    model=model,
                )
                language = detection.language or "en"
            alignment = self.alignment_service.align(
                AlignTranscriptRequest(
                    audio_path=request.audio_path,
                    transcript_text=request.source_text,
                    language=language,
                    transcript_source_name=request.source_text_name,
                    output_srt_path=None,
                )
            )
            words = words_from_alignment(alignment.words)
            mode = "forced_alignment"
            resolved_language = alignment.language
        else:
            transcription = self.transcription_service.transcribe(
                request.audio_path,
                language=request.language,
                model_size=request.model_size,
                model=model,
            )
            words = words_from_transcription(transcription.words)
            mode = "transcription"
            resolved_language = transcription.language or request.language or "en"

        units = words if request.granularity == "words" else sentence_units(words)
        cues = grouped_cues(units, request.group_size)
        srt_text = render_srt(cues)
        if request.output_srt_path is not None:
            request.output_srt_path.write_text(srt_text, encoding="utf-8")
        return RenderSubtitlesResult(
            language=resolved_language,
            mode=mode,
            output_format=request.output_format,
            granularity=request.granularity,
            group_size=request.group_size,
            cues=cues,
            srt_text=srt_text,
            output_srt_path=request.output_srt_path,
        )
