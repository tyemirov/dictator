"""Typed transcription models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordSegment:
    """A single transcribed word with time bounds."""

    text: str
    start_seconds: float | None
    end_seconds: float | None

    def to_legacy_dict(self) -> dict[str, float | str | None]:
        return {
            "content": self.text,
            "start": self.start_seconds,
            "end": self.end_seconds,
        }
