"""Client helpers for Dictator services."""

from ._jobs import RemoteJobCanceledError, RemoteJobFailedError
from .diarization import DiarizationClient, DiarizationResult
from .dictation import DictationClient, DictationJob, DictationResult
from .subtitles import SubtitleClient, SubtitleJob, SubtitleResult
from .voice import (
    ReferenceSampleClient,
    ReferenceSampleJob,
    ReferenceSampleResult,
    SynthesisClient,
    SynthesisJob,
    SynthesisResult,
    SynthesisVoice,
)
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
    "RemoteJobCanceledError",
    "RemoteJobFailedError",
    "SynthesisClient",
    "SynthesisJob",
    "SynthesisResult",
    "SynthesisVoice",
    "SubtitleClient",
    "SubtitleJob",
    "SubtitleResult",
]

from .alignment import AlignmentClient, AlignmentJob, AlignmentResult
