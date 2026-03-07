"""Client helpers for Dictator services."""

from .diarization import DiarizationClient, DiarizationResult
from .dictation import DictationClient, DictationResult
from .subtitles import SubtitleClient, SubtitleResult

__all__ = [
    "DiarizationClient",
    "DiarizationResult",
    "DictationClient",
    "DictationResult",
    "SubtitleClient",
    "SubtitleResult",
]
