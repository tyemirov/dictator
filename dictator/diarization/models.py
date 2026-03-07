"""Typed models for speaker diarization results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerSegment:
    """A diarized speaker turn with a request-local speaker label."""

    speaker: str
    start_seconds: float
    end_seconds: float
    raw_label: str | None = None

    def to_json_dict(self) -> dict[str, float | str]:
        return {
            "speaker": self.speaker,
            "start": self.start_seconds,
            "end": self.end_seconds,
        }


@dataclass(frozen=True)
class DiarizedWord:
    """A transcribed word attributed to a speaker."""

    text: str
    start_seconds: float
    end_seconds: float
    speaker: str

    def to_json_dict(self) -> dict[str, float | str]:
        return {
            "word": self.text,
            "speaker": self.speaker,
            "start": self.start_seconds,
            "end": self.end_seconds,
        }

    def to_legacy_dict(self) -> dict[str, float | str]:
        return {
            "content": self.text,
            "speaker": self.speaker,
            "start": self.start_seconds,
            "end": self.end_seconds,
        }


@dataclass(frozen=True)
class DiarizedUtterance:
    """A contiguous run of words spoken by the same speaker."""

    speaker: str
    start_seconds: float
    end_seconds: float
    text: str
    words: tuple[DiarizedWord, ...]

    def to_json_dict(self, include_words: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "speaker": self.speaker,
            "start": self.start_seconds,
            "end": self.end_seconds,
            "text": self.text,
        }
        if include_words:
            payload["words"] = [word.to_json_dict() for word in self.words]
        return payload


@dataclass(frozen=True)
class SpeakerSummary:
    """Aggregated request-local speaker metadata."""

    speaker: str
    word_count: int
    utterance_count: int
    total_duration_seconds: float

    def to_json_dict(self) -> dict[str, float | int | str]:
        return {
            "speaker": self.speaker,
            "wordCount": self.word_count,
            "utteranceCount": self.utterance_count,
            "totalDurationSeconds": self.total_duration_seconds,
        }


@dataclass(frozen=True)
class DiarizeAudioRequest:
    """Inputs for full-speaker diarization with transcribed words."""

    input_path: Path
    language: str | None = None
    model_size: str = "base"
    include_words: bool = True
    include_utterances: bool = True
    include_speakers: bool = True
    include_speaker_segments: bool = False
    utterance_gap_seconds: float = 0.75


@dataclass(frozen=True)
class DiarizeAudioResult:
    """Speaker-attributed transcription output."""

    language: str | None
    text: str
    words: tuple[DiarizedWord, ...]
    utterances: tuple[DiarizedUtterance, ...]
    speakers: tuple[SpeakerSummary, ...]
    speaker_segments: tuple[SpeakerSegment, ...]

    def to_json_dict(
        self,
        *,
        include_words: bool = True,
        include_utterances: bool = True,
        include_speakers: bool = True,
        include_speaker_segments: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "text": self.text,
        }
        if self.language:
            payload["languageCode"] = self.language
        if include_words:
            payload["words"] = [word.to_json_dict() for word in self.words]
        if include_utterances:
            payload["utterances"] = [
                utterance.to_json_dict(include_words=include_words)
                for utterance in self.utterances
            ]
        if include_speakers:
            payload["speakers"] = [speaker.to_json_dict() for speaker in self.speakers]
        if include_speaker_segments:
            payload["speakerSegments"] = [
                segment.to_json_dict() for segment in self.speaker_segments
            ]
        return payload
