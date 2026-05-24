"""Typed synthesis results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class SynthesisEngine(str, Enum):
    """Supported speech synthesis engines."""

    QWEN3 = "qwen3"
    SILERO_RU = "silero_ru"


class SynthesisTextFormat(str, Enum):
    """Caller-declared synthesis text markup format."""

    AUTO = "auto"
    PLAIN_TEXT = "plain_text"
    SSML = "ssml"


@dataclass(frozen=True)
class SynthesisAudioFormat:
    """Resolved synthesis output-audio contract."""

    container: str
    codec: str
    sample_rate_hz: int
    channel_count: int
    bit_depth: int


DEFAULT_SYNTHESIS_AUDIO_FORMAT = SynthesisAudioFormat(
    container="wav",
    codec="pcm_s16le",
    sample_rate_hz=24_000,
    channel_count=1,
    bit_depth=16,
)

SILERO_RU_SYNTHESIS_AUDIO_FORMAT = SynthesisAudioFormat(
    container="wav",
    codec="pcm_s16le",
    sample_rate_hz=24_000,
    channel_count=1,
    bit_depth=16,
)
SILERO_RU_NATIVE_SAMPLE_RATES = (8_000, 24_000, 48_000)
SILERO_RU_SUPPORTED_SPEAKERS = ("baya", "xenia")


@dataclass(frozen=True)
class SynthesisChunk:
    """A synthesiser-ready text chunk with its atomic sentence units."""

    text: str
    units: tuple[str, ...]
    timeline_text: str | None = None

    @classmethod
    def from_text(cls, text: str, *, timeline_text: str | None = None) -> "SynthesisChunk":
        return cls.from_units((text,), timeline_text=timeline_text)

    @classmethod
    def from_units(cls, units: Sequence[str], *, timeline_text: str | None = None) -> "SynthesisChunk":
        normalized_units = tuple(unit.strip() for unit in units if unit.strip())
        if not normalized_units:
            raise ValueError("synthesis chunk units cannot be empty")
        return cls(
            text=" ".join(normalized_units),
            units=normalized_units,
            timeline_text=timeline_text.strip() if timeline_text and timeline_text.strip() else None,
        )


@dataclass(frozen=True)
class SynthesisRequest:
    """Engine-aware synthesis input."""

    engine: SynthesisEngine
    speaker_wav: Path | None
    text: str
    language_code: str
    cap_seconds: float | None
    speaker_artifact_id: str | None = None
    speaker_transcript_text: str | None = None
    preset_speaker: str | None = None
    audio_format: SynthesisAudioFormat | None = None
    text_format: SynthesisTextFormat = SynthesisTextFormat.AUTO


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

    def to_timeline_dict(self) -> dict[str, float | str]:
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
