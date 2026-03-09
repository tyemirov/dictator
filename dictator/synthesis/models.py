"""Typed synthesis results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class SynthesisEngine(str, Enum):
    """Supported speech synthesis engines."""

    XTTS = "xtts"
    QWEN3 = "qwen3"


@dataclass(frozen=True)
class SynthesisChunk:
    """A synthesiser-ready text chunk with its atomic sentence units."""

    text: str
    units: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "SynthesisChunk":
        return cls.from_units((text,))

    @classmethod
    def from_units(cls, units: Sequence[str]) -> "SynthesisChunk":
        normalized_units = tuple(unit.strip() for unit in units if unit.strip())
        if not normalized_units:
            raise ValueError("synthesis chunk units cannot be empty")
        return cls(
            text=" ".join(normalized_units),
            units=normalized_units,
        )


@dataclass(frozen=True)
class SynthesisRequest:
    """Engine-aware synthesis input."""

    engine: SynthesisEngine
    speaker_wav: Path
    text: str
    language_code: str
    cap_seconds: float | None
    speaker_artifact_id: str | None = None
    speaker_transcript_text: str | None = None


@dataclass(frozen=True)
class SynthesisedAudioChunk:
    """In-memory audio chunk produced by a synthesis session."""

    samples: Any
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True)
class SpeechSegment:
    """Synthesised speech segment with time bounds."""

    text: str
    start_seconds: float
    end_seconds: float

    def to_legacy_dict(self) -> dict[str, float | str]:
        return {
            "content": self.text,
            "start": self.start_seconds,
            "end": self.end_seconds,
        }


@dataclass(frozen=True)
class SynthesisResult:
    """Temporary chunk files and their output timeline."""

    temp_dir: Path
    wav_paths: tuple[Path, ...]
    segments: tuple[SpeechSegment, ...]
