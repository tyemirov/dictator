"""Whisper-backed transcription package."""

from .models import TranscriptionResult, WordSegment
from .service import TranscriptionService

__all__ = ["TranscriptionResult", "TranscriptionService", "WordSegment"]
