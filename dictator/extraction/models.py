"""Typed models for reference voice extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceExtractionRequest:
    """Inputs for extracting a clean reference clip."""

    input_path: Path
    output_path: Path | None = None
    model_size: str = "medium"
    duration_seconds: float = 20.0
    language: str | None = None
    max_speech_rate: float = 4.0
    min_centroid_hz: float = 500.0
    max_centroid_hz: float = 4000.0


@dataclass(frozen=True)
class ReferenceExtractionResult:
    """Selected reference clip metadata."""

    raw_words: tuple[dict[str, object], ...]
    dominant_speaker_words: tuple[dict[str, object], ...]
    window_start_seconds: float
    window_end_seconds: float
    trim_start_seconds: float
    trim_end_seconds: float
    output_path: Path | None
