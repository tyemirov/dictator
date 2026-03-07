"""Typed models for grouped subtitle rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SubtitleFormat = Literal["srt"]
SubtitleGranularity = Literal["words", "sentences"]
SubtitleMode = Literal["transcription", "forced_alignment"]


@dataclass(frozen=True)
class TimedWord:
    """A timed token used to derive subtitle cues."""

    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class SubtitleCue:
    """A rendered subtitle cue."""

    index: int
    text: str
    start_seconds: float
    end_seconds: float
    item_count: int


@dataclass(frozen=True)
class RenderSubtitlesRequest:
    """Inputs for audio-to-SRT rendering."""

    audio_path: Path
    language: str | None = None
    model_size: str = "base"
    output_format: SubtitleFormat = "srt"
    granularity: SubtitleGranularity = "words"
    group_size: int = 1
    source_text: str | None = None
    source_text_name: str = "transcript.txt"
    output_srt_path: Path | None = None


@dataclass(frozen=True)
class RenderSubtitlesResult:
    """Rendered subtitles and mode metadata."""

    language: str
    mode: SubtitleMode
    output_format: SubtitleFormat
    granularity: SubtitleGranularity
    group_size: int
    cues: tuple[SubtitleCue, ...]
    srt_text: str
    output_srt_path: Path | None = None
