"""Client helpers for Dictator services."""

from ._jobs import RemoteJobFailedError
from .diarization import DiarizationClient, DiarizationResult
from .dictation import DictationClient, DictationJob, DictationResult
from .subtitles import SubtitleClient, SubtitleJob, SubtitleResult
from .voice import ReferenceSampleClient, ReferenceSampleJob, ReferenceSampleResult
from .diarization import DiarizationJob

__all__ = [
    "DiarizationClient",
    "DiarizationJob",
    "DiarizationResult",
    "DictationClient",
    "DictationJob",
    "DictationResult",
    "ReferenceSampleClient",
    "ReferenceSampleJob",
    "ReferenceSampleResult",
    "RemoteJobFailedError",
    "SubtitleClient",
    "SubtitleJob",
    "SubtitleResult",
]

from .alignment import AlignmentClient, AlignmentJob, AlignmentResult
