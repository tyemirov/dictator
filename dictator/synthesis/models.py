"""Typed synthesis results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
