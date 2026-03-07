"""Typed models for forced alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_ALIGNMENT_LANGUAGES = (
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("zh", "Chinese"),
    ("nl", "Dutch"),
    ("uk", "Ukrainian"),
    ("pt", "Portuguese"),
    ("ar", "Arabic"),
    ("cs", "Czech"),
    ("ru", "Russian"),
    ("pl", "Polish"),
    ("hu", "Hungarian"),
    ("fi", "Finnish"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("tr", "Turkish"),
    ("da", "Danish"),
    ("he", "Hebrew"),
    ("vi", "Vietnamese"),
    ("ko", "Korean"),
    ("ur", "Urdu"),
    ("te", "Telugu"),
    ("hi", "Hindi"),
    ("ca", "Catalan"),
    ("ml", "Malayalam"),
    ("no", "Norwegian Bokmal"),
    ("nn", "Norwegian Nynorsk"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("hr", "Croatian"),
    ("ro", "Romanian"),
    ("eu", "Basque"),
    ("gl", "Galician"),
    ("ka", "Georgian"),
)
SUPPORTED_LANGUAGE_CODES = {code for code, _ in SUPPORTED_ALIGNMENT_LANGUAGES}


@dataclass(frozen=True)
class AlignedWord:
    """Aligned token with time bounds."""

    text: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("aligned word text is empty")
        if self.start_seconds < 0:
            raise ValueError("aligned word start is negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("aligned word end is not after start")

    def to_legacy_dict(self) -> dict[str, float | str]:
        return {
            "content": self.text,
            "start": self.start_seconds,
            "end": self.end_seconds,
        }


@dataclass(frozen=True)
class AlignTranscriptRequest:
    """Inputs for transcript-to-audio forced alignment."""

    audio_path: Path
    transcript_text: str
    language: str = ""
    remove_punctuation: bool = False
    device: str = "auto"
    transcript_source_name: str = "transcript.txt"
    output_srt_path: Path | None = None


@dataclass(frozen=True)
class AlignTranscriptResult:
    """Aligned words plus SRT content."""

    audio_path: Path
    language: str
    words: tuple[AlignedWord, ...]
    srt_text: str
    output_srt_path: Path | None = None
