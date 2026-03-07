"""Speaker diarization service with parameterized JSON serialization."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

from dictator.audio.constants import PCM_SAMPLE_RATE
from dictator.runtime import DependencyError, ProcessingError, ValidationError

from .models import (
    DiarizeAudioRequest,
    DiarizeAudioResult,
    DiarizedUtterance,
    DiarizedWord,
    SpeakerSummary,
    SpeakerSegment,
)

_DEFAULT_UTTERANCE_GAP_SECONDS = 0.75

if TYPE_CHECKING:
    from dictator.transcription.service import TranscriptionService


def _coerce_word_bounds(payload: dict[str, object]) -> tuple[float, float]:
    start_value = payload.get("start")
    end_value = payload.get("end")
    if start_value is None and end_value is None:
        raise ProcessingError(
            "dictator.diarization.word_missing_timestamps",
            "word timestamps are required for diarization output",
        )
    start_seconds = float(start_value if start_value is not None else end_value)
    end_seconds = float(end_value if end_value is not None else start_seconds)
    if end_seconds < start_seconds:
        end_seconds = start_seconds
    return start_seconds, end_seconds


def _speaker_distance(midpoint: float, segment: SpeakerSegment) -> float:
    if segment.start_seconds <= midpoint <= segment.end_seconds:
        return 0.0
    return min(
        abs(midpoint - segment.start_seconds),
        abs(midpoint - segment.end_seconds),
    )


def _best_speaker_segment(
    start_seconds: float,
    end_seconds: float,
    speaker_segments: Sequence[SpeakerSegment],
) -> SpeakerSegment:
    if not speaker_segments:
        raise ProcessingError(
            "dictator.diarization.no_speaker_segments",
            "no speakers detected",
        )
    midpoint = (start_seconds + end_seconds) / 2.0
    best_segment: SpeakerSegment | None = None
    best_key: tuple[float, int, float, float, float] | None = None
    for index, segment in enumerate(speaker_segments):
        overlap = max(
            0.0,
            min(end_seconds, segment.end_seconds) - max(start_seconds, segment.start_seconds),
        )
        contains_midpoint = int(segment.start_seconds <= midpoint <= segment.end_seconds)
        distance = _speaker_distance(midpoint, segment)
        duration = segment.end_seconds - segment.start_seconds
        key = (
            overlap,
            contains_midpoint,
            -distance,
            duration,
            -float(index),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_segment = segment
    assert best_segment is not None
    return best_segment


def assign_words_to_speakers(
    words: Iterable[dict[str, object]],
    speaker_segments: Sequence[SpeakerSegment],
) -> tuple[DiarizedWord, ...]:
    """Attribute each word to the best matching speaker segment."""
    assigned_words: list[DiarizedWord] = []
    for payload in words:
        start_seconds, end_seconds = _coerce_word_bounds(payload)
        segment = _best_speaker_segment(start_seconds, end_seconds, speaker_segments)
        assigned_words.append(
            DiarizedWord(
                text=str(payload.get("content") or ""),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                speaker=segment.speaker,
            )
        )
    return tuple(assigned_words)


def build_utterances(
    words: Sequence[DiarizedWord],
    *,
    utterance_gap_seconds: float = _DEFAULT_UTTERANCE_GAP_SECONDS,
) -> tuple[DiarizedUtterance, ...]:
    """Group speaker-attributed words into utterances."""
    if utterance_gap_seconds < 0:
        raise ValidationError(
            "dictator.diarization.invalid_utterance_gap",
            "utterance_gap_seconds must be non-negative",
        )
    sorted_words = sorted(
        words,
        key=lambda word: (word.start_seconds, word.end_seconds, word.speaker, word.text),
    )
    if not sorted_words:
        return ()

    utterances: list[DiarizedUtterance] = []
    current_words: list[DiarizedWord] = [sorted_words[0]]
    current_speaker = sorted_words[0].speaker
    current_end_seconds = sorted_words[0].end_seconds

    for word in sorted_words[1:]:
        gap_seconds = max(0.0, word.start_seconds - current_end_seconds)
        if word.speaker != current_speaker or gap_seconds > utterance_gap_seconds:
            utterances.append(_build_utterance(current_words))
            current_words = [word]
            current_speaker = word.speaker
            current_end_seconds = word.end_seconds
            continue
        current_words.append(word)
        current_end_seconds = max(current_end_seconds, word.end_seconds)

    utterances.append(_build_utterance(current_words))
    return tuple(utterances)


def _build_utterance(words: Sequence[DiarizedWord]) -> DiarizedUtterance:
    text = " ".join(word.text for word in words if word.text)
    return DiarizedUtterance(
        speaker=words[0].speaker,
        start_seconds=words[0].start_seconds,
        end_seconds=max(word.end_seconds for word in words),
        text=text,
        words=tuple(words),
    )


def build_speaker_segments(
    raw_tracks: Iterable[tuple[float, float, str]],
) -> tuple[SpeakerSegment, ...]:
    """Convert raw diarization turns into stable request-local speaker labels."""
    ordered_tracks = sorted(
        raw_tracks,
        key=lambda track: (float(track[0]), float(track[1]), str(track[2])),
    )
    speaker_map: dict[str, str] = {}
    speaker_segments: list[SpeakerSegment] = []
    for start_seconds, end_seconds, raw_label in ordered_tracks:
        speaker_map.setdefault(str(raw_label), f"S{len(speaker_map) + 1}")
        speaker_segments.append(
            SpeakerSegment(
                speaker=speaker_map[str(raw_label)],
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds),
                raw_label=str(raw_label),
            )
        )
    return tuple(speaker_segments)


def build_speaker_summaries(
    words: Sequence[DiarizedWord],
    utterances: Sequence[DiarizedUtterance],
    speaker_segments: Sequence[SpeakerSegment],
) -> tuple[SpeakerSummary, ...]:
    """Aggregate per-speaker statistics for full diarization output."""
    word_counts: Counter[str] = Counter(word.speaker for word in words)
    utterance_counts: Counter[str] = Counter(utterance.speaker for utterance in utterances)
    durations: Counter[str] = Counter()
    for segment in speaker_segments:
        durations[segment.speaker] += segment.end_seconds - segment.start_seconds

    speaker_order: list[str] = []
    for source in (speaker_segments, utterances, words):
        for item in source:
            speaker = item.speaker
            if speaker not in speaker_order:
                speaker_order.append(speaker)

    return tuple(
        SpeakerSummary(
            speaker=speaker,
            word_count=word_counts[speaker],
            utterance_count=utterance_counts[speaker],
            total_duration_seconds=float(durations[speaker]),
        )
        for speaker in speaker_order
    )


def dominant_speaker_label(speaker_segments: Sequence[SpeakerSegment]) -> str:
    """Pick the speaker with the largest total diarized duration."""
    speaker_durations: Counter[str] = Counter()
    for segment in speaker_segments:
        speaker_durations[segment.speaker] += segment.end_seconds - segment.start_seconds
    if not speaker_durations:
        raise ProcessingError(
            "dictator.diarization.no_speaker_segments",
            "no speakers detected",
        )
    return speaker_durations.most_common(1)[0][0]


def run_diarization(
    diarization_pipeline: object,
    audio_file: Path,
) -> tuple[SpeakerSegment, ...]:
    """Execute the diarization pipeline and normalize speaker labels."""
    try:
        import numpy  # noqa: F401
        import torch
    except ImportError as exc:
        raise DependencyError(
            "dictator.diarization.dependencies_missing",
            "numpy and torch are required for diarization",
        ) from exc
    from dictator.audio.ffmpeg_ops import decode_pcm

    samples = decode_pcm(audio_file).astype("float32") / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)
    diarization_result = diarization_pipeline(
        {
            "uri": audio_file.stem,
            "waveform": waveform,
            "sample_rate": PCM_SAMPLE_RATE,
        }
    )
    raw_tracks = [
        (float(turn.start), float(turn.end), str(speaker_label))
        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True)
    ]
    speaker_segments = build_speaker_segments(raw_tracks)
    if not speaker_segments:
        raise ProcessingError(
            "dictator.diarization.no_speaker_segments",
            "no speakers detected",
        )
    return speaker_segments


class DiarizationService:
    """Application service for speaker-attributed transcription output."""

    def __init__(
        self,
        transcription_service: "TranscriptionService | None" = None,
        diarization_pipeline_loader: Callable[[], object] | None = None,
    ) -> None:
        if transcription_service is None:
            from dictator.transcription.service import TranscriptionService

            transcription_service = TranscriptionService()
        self._transcription_service = transcription_service
        self._diarization_pipeline_loader = diarization_pipeline_loader

    def diarize(
        self,
        request: DiarizeAudioRequest,
        *,
        model: object | None = None,
        diarization_pipeline: object | None = None,
    ) -> DiarizeAudioResult:
        if request.utterance_gap_seconds < 0:
            raise ValidationError(
                "dictator.diarization.invalid_utterance_gap",
                "utterance_gap_seconds must be non-negative",
            )
        resolved_pipeline = diarization_pipeline or self._load_pipeline()
        transcription = self._transcription_service.transcribe(
            request.input_path,
            language=request.language,
            model_size=request.model_size,
            model=model,
        )
        speaker_segments = run_diarization(resolved_pipeline, request.input_path)
        diarized_words = assign_words_to_speakers(
            [word.to_legacy_dict() for word in transcription.words],
            speaker_segments,
        )
        utterances = build_utterances(
            diarized_words,
            utterance_gap_seconds=request.utterance_gap_seconds,
        )
        speakers = build_speaker_summaries(diarized_words, utterances, speaker_segments)
        return DiarizeAudioResult(
            language=transcription.language,
            text=" ".join(word.text for word in diarized_words if word.text),
            words=diarized_words,
            utterances=utterances,
            speakers=speakers,
            speaker_segments=speaker_segments,
        )

    def _load_pipeline(self) -> object:
        if self._diarization_pipeline_loader is None:
            raise DependencyError(
                "dictator.diarization.pipeline_unavailable",
                "a diarization pipeline loader is required",
            )
        return self._diarization_pipeline_loader()
