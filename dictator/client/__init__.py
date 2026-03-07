"""Client helpers for Dictator services."""

from .diarization import DiarizationClient, DiarizationResult
from .dictation import DictationClient, DictationResult

__all__ = [
    "DiarizationClient",
    "DiarizationResult",
    "DictationClient",
    "DictationResult",
]
