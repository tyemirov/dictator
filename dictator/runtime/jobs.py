"""Background synthesis jobs and their persistent status store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Generic, TypeVar
import uuid

from dictator.alignment.models import AlignTranscriptRequest, AlignedWord
from dictator.diarization.models import DiarizeAudioRequest, DiarizeAudioResult
from dictator.runtime.errors import DictatorError, ServiceRequestError, ValidationError
from dictator.storage import LocalArtifactStore
from dictator.subtitles.models import RenderSubtitlesRequest, SubtitleCue
from dictator.synthesis.models import SynthesisAudioFormat
from dictator.synthesis.workflow import PreparedSynthesisRequest, execute_synthesis_request
from dictator.transcription.models import TranscriptionResult, WordSegment


PreparedT = TypeVar("PreparedT")
RecordT = TypeVar("RecordT")


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


_TERMINAL_JOB_STATES = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED}


SynthesisJobState = JobState


@dataclass(frozen=True)
class SynthesisJobRecord:
    job_id: str
    state: SynthesisJobState
    engine: str
    language_code: str
    include_timeline: bool
    speaker_artifact_id: str
    created_at_unix_seconds: float
    started_at_unix_seconds: float | None = None
    finished_at_unix_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    audio_artifact_id: str | None = None
    audio_duration_seconds: float | None = None
    audio_format: SynthesisAudioFormat | None = None
    timeline_artifact_id: str | None = None
    chunk_count: int | None = None
    estimated_total_chunks: int | None = None
    completed_chunks: int | None = None

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "SynthesisJobRecord":
        return cls(
            job_id=str(payload["job_id"]),
            state=SynthesisJobState(str(payload["state"])),
            engine=str(payload["engine"]),
            language_code=str(payload["language_code"]),
            include_timeline=bool(payload["include_timeline"]),
            speaker_artifact_id=str(payload["speaker_artifact_id"]),
            created_at_unix_seconds=float(payload["created_at_unix_seconds"]),
            started_at_unix_seconds=_optional_float(payload.get("started_at_unix_seconds")),
            finished_at_unix_seconds=_optional_float(payload.get("finished_at_unix_seconds")),
            error_code=_optional_str(payload.get("error_code")),
            error_message=_optional_str(payload.get("error_message")),
            audio_artifact_id=_optional_str(payload.get("audio_artifact_id")),
            audio_duration_seconds=_optional_float(payload.get("audio_duration_seconds")),
            audio_format=_optional_synthesis_audio_format(payload.get("audio_format")),
            timeline_artifact_id=_optional_str(payload.get("timeline_artifact_id")),
            chunk_count=_optional_int(payload.get("chunk_count")),
            estimated_total_chunks=_optional_int(payload.get("estimated_total_chunks")),
            completed_chunks=_optional_int(payload.get("completed_chunks")),
        )


AlignmentJobState = JobState


TranscriptionJobState = JobState
DiarizationJobState = JobState
SubtitleJobState = JobState
ExtractReferenceSampleJobState = JobState


@dataclass(frozen=True)
class PreparedAlignmentJob:
    audio_record: object
    transcript_text: str
    language_code: str
    remove_punctuation: bool
    include_srt_text: bool
    transcript_source_name: str = "transcript.txt"


@dataclass(frozen=True)
class AlignmentJobRecord:
    job_id: str
    state: AlignmentJobState
    audio_artifact_id: str
    include_srt_text: bool
    created_at_unix_seconds: float
    started_at_unix_seconds: float | None = None
    finished_at_unix_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    language_code: str | None = None
    words: tuple[AlignedWord, ...] = ()
    srt_text: str | None = None
    srt_artifact_id: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["words"] = _alignment_words_to_json(self.words)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "AlignmentJobRecord":
        return cls(
            job_id=str(payload["job_id"]),
            state=AlignmentJobState(str(payload["state"])),
            audio_artifact_id=str(payload["audio_artifact_id"]),
            include_srt_text=bool(payload["include_srt_text"]),
            created_at_unix_seconds=float(payload["created_at_unix_seconds"]),
            started_at_unix_seconds=_optional_float(payload.get("started_at_unix_seconds")),
            finished_at_unix_seconds=_optional_float(payload.get("finished_at_unix_seconds")),
            error_code=_optional_str(payload.get("error_code")),
            error_message=_optional_str(payload.get("error_message")),
            language_code=_optional_str(payload.get("language_code")),
            words=_alignment_words_from_json(payload.get("words")),
            srt_text=_optional_str(payload.get("srt_text")),
            srt_artifact_id=_optional_str(payload.get("srt_artifact_id")),
        )


@dataclass(frozen=True)
class PreparedTranscriptionJob:
    audio_record: object
    language_code: str | None
    model_size: str
    include_word_segments: bool


@dataclass(frozen=True)
class TranscriptionJobRecord:
    job_id: str
    state: TranscriptionJobState
    audio_artifact_id: str
    include_word_segments: bool
    created_at_unix_seconds: float
    started_at_unix_seconds: float | None = None
    finished_at_unix_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    text: str | None = None
    language_code: str | None = None
    words: tuple[WordSegment, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["words"] = _transcription_words_to_json(self.words)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "TranscriptionJobRecord":
        return cls(
            job_id=str(payload["job_id"]),
            state=TranscriptionJobState(str(payload["state"])),
            audio_artifact_id=str(payload["audio_artifact_id"]),
            include_word_segments=bool(payload["include_word_segments"]),
            created_at_unix_seconds=float(payload["created_at_unix_seconds"]),
            started_at_unix_seconds=_optional_float(payload.get("started_at_unix_seconds")),
            finished_at_unix_seconds=_optional_float(payload.get("finished_at_unix_seconds")),
            error_code=_optional_str(payload.get("error_code")),
            error_message=_optional_str(payload.get("error_message")),
            text=_optional_str(payload.get("text")),
            language_code=_optional_str(payload.get("language_code")),
            words=_transcription_words_from_json(payload.get("words")),
        )


@dataclass(frozen=True)
class PreparedDiarizationJob:
    audio_record: object
    language_code: str | None
    model_size: str
    include_words: bool
    include_utterances: bool
    include_speakers: bool
    include_speaker_segments: bool
    utterance_gap_seconds: float
    persist_json_artifact: bool


@dataclass(frozen=True)
class DiarizationJobRecord:
    job_id: str
    state: DiarizationJobState
    audio_artifact_id: str
    include_words: bool
    include_utterances: bool
    include_speakers: bool
    include_speaker_segments: bool
    persist_json_artifact: bool
    created_at_unix_seconds: float
    started_at_unix_seconds: float | None = None
    finished_at_unix_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    text: str | None = None
    language_code: str | None = None
    diarization: dict[str, object] | None = None
    diarization_artifact_id: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "DiarizationJobRecord":
        diarization = payload.get("diarization")
        if diarization is not None and not isinstance(diarization, dict):
            raise ValueError("diarization job payload must be a dict")
        return cls(
            job_id=str(payload["job_id"]),
            state=DiarizationJobState(str(payload["state"])),
            audio_artifact_id=str(payload["audio_artifact_id"]),
            include_words=bool(payload["include_words"]),
            include_utterances=bool(payload["include_utterances"]),
            include_speakers=bool(payload["include_speakers"]),
            include_speaker_segments=bool(payload["include_speaker_segments"]),
            persist_json_artifact=bool(payload["persist_json_artifact"]),
            created_at_unix_seconds=float(payload["created_at_unix_seconds"]),
            started_at_unix_seconds=_optional_float(payload.get("started_at_unix_seconds")),
            finished_at_unix_seconds=_optional_float(payload.get("finished_at_unix_seconds")),
            error_code=_optional_str(payload.get("error_code")),
            error_message=_optional_str(payload.get("error_message")),
            text=_optional_str(payload.get("text")),
            language_code=_optional_str(payload.get("language_code")),
            diarization=diarization,
            diarization_artifact_id=_optional_str(payload.get("diarization_artifact_id")),
        )


@dataclass(frozen=True)
class PreparedSubtitleJob:
    audio_record: object
    language_code: str | None
    model_size: str
    granularity: str
    group_size: int
    source_text: str | None
    source_text_name: str
    include_srt_text: bool


@dataclass(frozen=True)
class SubtitleJobRecord:
    job_id: str
    state: SubtitleJobState
    audio_artifact_id: str
    include_srt_text: bool
    created_at_unix_seconds: float
    started_at_unix_seconds: float | None = None
    finished_at_unix_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    language_code: str | None = None
    mode: str | None = None
    output_format: str | None = None
    granularity: str | None = None
    group_size: int = 0
    cues: tuple[SubtitleCue, ...] = ()
    srt_text: str | None = None
    srt_artifact_id: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["cues"] = _subtitle_cues_to_json(self.cues)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "SubtitleJobRecord":
        return cls(
            job_id=str(payload["job_id"]),
            state=SubtitleJobState(str(payload["state"])),
            audio_artifact_id=str(payload["audio_artifact_id"]),
            include_srt_text=bool(payload["include_srt_text"]),
            created_at_unix_seconds=float(payload["created_at_unix_seconds"]),
            started_at_unix_seconds=_optional_float(payload.get("started_at_unix_seconds")),
            finished_at_unix_seconds=_optional_float(payload.get("finished_at_unix_seconds")),
            error_code=_optional_str(payload.get("error_code")),
            error_message=_optional_str(payload.get("error_message")),
            language_code=_optional_str(payload.get("language_code")),
            mode=_optional_str(payload.get("mode")),
            output_format=_optional_str(payload.get("output_format")),
            granularity=_optional_str(payload.get("granularity")),
            group_size=int(payload.get("group_size") or 0),
            cues=_subtitle_cues_from_json(payload.get("cues")),
            srt_text=_optional_str(payload.get("srt_text")),
            srt_artifact_id=_optional_str(payload.get("srt_artifact_id")),
        )


@dataclass(frozen=True)
class PreparedExtractReferenceSampleJob:
    source_record: object
    model_size: str
    language_code: str | None
    duration_seconds: float
    max_speech_rate: float
    min_centroid_hz: float
    max_centroid_hz: float


@dataclass(frozen=True)
class ExtractReferenceSampleJobRecord:
    job_id: str
    state: ExtractReferenceSampleJobState
    source_artifact_id: str
    created_at_unix_seconds: float
    started_at_unix_seconds: float | None = None
    finished_at_unix_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    sample_artifact_id: str | None = None
    trim_start_seconds: float | None = None
    trim_end_seconds: float | None = None
    window_start_seconds: float | None = None
    window_end_seconds: float | None = None
    dominant_speaker_word_count: int | None = None

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "ExtractReferenceSampleJobRecord":
        return cls(
            job_id=str(payload["job_id"]),
            state=ExtractReferenceSampleJobState(str(payload["state"])),
            source_artifact_id=str(payload["source_artifact_id"]),
            created_at_unix_seconds=float(payload["created_at_unix_seconds"]),
            started_at_unix_seconds=_optional_float(payload.get("started_at_unix_seconds")),
            finished_at_unix_seconds=_optional_float(payload.get("finished_at_unix_seconds")),
            error_code=_optional_str(payload.get("error_code")),
            error_message=_optional_str(payload.get("error_message")),
            sample_artifact_id=_optional_str(payload.get("sample_artifact_id")),
            trim_start_seconds=_optional_float(payload.get("trim_start_seconds")),
            trim_end_seconds=_optional_float(payload.get("trim_end_seconds")),
            window_start_seconds=_optional_float(payload.get("window_start_seconds")),
            window_end_seconds=_optional_float(payload.get("window_end_seconds")),
            dominant_speaker_word_count=_optional_int(payload.get("dominant_speaker_word_count")),
        )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_synthesis_audio_format(value: object) -> SynthesisAudioFormat | None:
    if value is None:
        return None
    if isinstance(value, SynthesisAudioFormat):
        return value
    if not isinstance(value, dict):
        raise ValueError("synthesis audio_format must be a dict")
    return SynthesisAudioFormat(
        container=str(value["container"]),
        codec=str(value["codec"]),
        sample_rate_hz=int(value["sample_rate_hz"]),
        channel_count=int(value["channel_count"]),
        bit_depth=int(value["bit_depth"]),
    )


def _alignment_words_to_json(words: tuple[AlignedWord, ...]) -> list[dict[str, object]]:
    return [
        {
            "text": word.text,
            "start_seconds": word.start_seconds,
            "end_seconds": word.end_seconds,
        }
        for word in words
    ]


def _alignment_words_from_json(value: object) -> tuple[AlignedWord, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("alignment job words must be a list")
    return tuple(
        AlignedWord(
            text=str(item["text"]),
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
        )
        for item in value
    )


def _transcription_words_to_json(words: tuple[WordSegment, ...]) -> list[dict[str, object]]:
    return [
        {
            "text": word.text,
            "start_seconds": word.start_seconds,
            "end_seconds": word.end_seconds,
        }
        for word in words
    ]


def _transcription_words_from_json(value: object) -> tuple[WordSegment, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("transcription job words must be a list")
    return tuple(
        WordSegment(
            text=str(item["text"]),
            start_seconds=_optional_float(item.get("start_seconds")),
            end_seconds=_optional_float(item.get("end_seconds")),
        )
        for item in value
    )


def _subtitle_cues_to_json(cues: tuple[SubtitleCue, ...]) -> list[dict[str, object]]:
    return [
        {
            "index": cue.index,
            "text": cue.text,
            "start_seconds": cue.start_seconds,
            "end_seconds": cue.end_seconds,
            "item_count": cue.item_count,
        }
        for cue in cues
    ]


def _subtitle_cues_from_json(value: object) -> tuple[SubtitleCue, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("subtitle job cues must be a list")
    return tuple(
        SubtitleCue(
            index=int(item["index"]),
            text=str(item["text"]),
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
            item_count=int(item["item_count"]),
        )
        for item in value
    )


_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_synthesis_job_id(job_id: str) -> str:
    return validate_job_id(job_id)


def validate_alignment_job_id(job_id: str) -> str:
    return validate_job_id(job_id)


def validate_transcription_job_id(job_id: str) -> str:
    return validate_job_id(job_id)


def validate_diarization_job_id(job_id: str) -> str:
    return validate_job_id(job_id)


def validate_subtitle_job_id(job_id: str) -> str:
    return validate_job_id(job_id)


def validate_extract_reference_sample_job_id(job_id: str) -> str:
    return validate_job_id(job_id)


def validate_job_id(job_id: str) -> str:
    normalized = job_id.strip()
    if not normalized:
        raise ValidationError(
            "dictator.jobs.job_id_required",
            "job_id is required",
        )
    if not _JOB_ID_PATTERN.fullmatch(normalized):
        raise ValidationError(
            "dictator.jobs.invalid_job_id",
            "job_id contains unsupported characters",
        )
    return normalized
class _LocalJsonJobStore(Generic[RecordT]):
    def __init__(self, root_dir: Path, *, job_id_validator, record_from_json) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._job_id_validator = job_id_validator
        self._record_from_json = record_from_json

    def _job_path(self, job_id: str) -> Path:
        normalized = self._job_id_validator(job_id)
        return self.root_dir / f"{normalized}.json"

    def _write_record(self, record: RecordT) -> None:
        path = self._job_path(record.job_id)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(record.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _read_record_with_retry(self, path: Path) -> RecordT:
        # On some filesystems/OSs, read_text might race with replace(), leading to transient
        # FileNotFoundError or JSONDecodeError if we catch it mid-swap.
        for attempt in range(3):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return self._record_from_json(payload)
            except (FileNotFoundError, json.JSONDecodeError):
                if attempt == 2:
                    raise
                time.sleep(0.01)

    def _iter_records(self) -> tuple[RecordT, ...]:
        return tuple(self._read_record_with_retry(path) for path in self.root_dir.glob("*.json"))

    def get(self, job_id: str) -> RecordT:
        path = self._job_path(job_id)
        return self._read_record_with_retry(path)

    def update(self, job_id: str, **updates: object) -> RecordT:
        with self._lock:
            current = self.get(job_id)
            if current.state == JobState.CANCELED:
                return current
            payload = current.to_json_dict()
            payload.update(updates)
            record = self._record_from_json(payload)
            self._write_record(record)
            return record

    def cancel(self, job_id: str) -> RecordT:
        with self._lock:
            current = self.get(job_id)
            if current.state in _TERMINAL_JOB_STATES:
                return current
            record = replace(
                current,
                state=JobState.CANCELED,
                finished_at_unix_seconds=time.time(),
                error_code="dictator.jobs.canceled",
                error_message="job canceled",
            )
            self._write_record(record)
            return record


class LocalSynthesisJobStore(_LocalJsonJobStore[SynthesisJobRecord]):
    """Persist synthesis job status under a local root directory."""

    def __init__(self, root_dir: Path) -> None:
        super().__init__(
            root_dir,
            job_id_validator=validate_synthesis_job_id,
            record_from_json=SynthesisJobRecord.from_json_dict,
        )

    def create(self, prepared: PreparedSynthesisRequest) -> SynthesisJobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = SynthesisJobRecord(
                job_id=job_id,
                state=SynthesisJobState.QUEUED,
                engine=prepared.synthesis_request.engine.value,
                language_code=prepared.synthesis_request.language_code,
                include_timeline=prepared.include_timeline,
                speaker_artifact_id=prepared.speaker_record.artifact_id,
                created_at_unix_seconds=time.time(),
                audio_format=prepared.audio_format,
                estimated_total_chunks=0,
                completed_chunks=0,
            )
            self._write_record(record)
            return record

    def fail_incomplete_jobs(self, message: str) -> None:
        with self._lock:
            for record in self._iter_records():
                if record.state not in {SynthesisJobState.QUEUED, SynthesisJobState.RUNNING}:
                    continue
                failed = SynthesisJobRecord(
                    job_id=record.job_id,
                    state=SynthesisJobState.FAILED,
                    engine=record.engine,
                    language_code=record.language_code,
                    include_timeline=record.include_timeline,
                    speaker_artifact_id=record.speaker_artifact_id,
                    created_at_unix_seconds=record.created_at_unix_seconds,
                    started_at_unix_seconds=record.started_at_unix_seconds,
                    finished_at_unix_seconds=time.time(),
                    error_code="dictator.jobs.interrupted",
                    error_message=message,
                    audio_artifact_id=record.audio_artifact_id,
                    audio_duration_seconds=record.audio_duration_seconds,
                    audio_format=record.audio_format,
                    timeline_artifact_id=record.timeline_artifact_id,
                    chunk_count=record.chunk_count,
                    estimated_total_chunks=record.estimated_total_chunks,
                    completed_chunks=record.completed_chunks,
                )
                self._write_record(failed)


class LocalAlignmentJobStore(_LocalJsonJobStore[AlignmentJobRecord]):
    """Persist alignment job status under a local root directory."""

    def __init__(self, root_dir: Path) -> None:
        super().__init__(
            root_dir,
            job_id_validator=validate_alignment_job_id,
            record_from_json=AlignmentJobRecord.from_json_dict,
        )

    def create(self, prepared: PreparedAlignmentJob) -> AlignmentJobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = AlignmentJobRecord(
                job_id=job_id,
                state=AlignmentJobState.QUEUED,
                audio_artifact_id=prepared.audio_record.artifact_id,
                include_srt_text=prepared.include_srt_text,
                created_at_unix_seconds=time.time(),
            )
            self._write_record(record)
            return record

    def fail_incomplete_jobs(self, message: str) -> None:
        with self._lock:
            for record in self._iter_records():
                if record.state not in {AlignmentJobState.QUEUED, AlignmentJobState.RUNNING}:
                    continue
                failed = AlignmentJobRecord(
                    job_id=record.job_id,
                    state=AlignmentJobState.FAILED,
                    audio_artifact_id=record.audio_artifact_id,
                    include_srt_text=record.include_srt_text,
                    created_at_unix_seconds=record.created_at_unix_seconds,
                    started_at_unix_seconds=record.started_at_unix_seconds,
                    finished_at_unix_seconds=time.time(),
                    error_code="dictator.jobs.interrupted",
                    error_message=message,
                    language_code=record.language_code,
                    words=record.words,
                    srt_text=record.srt_text,
                    srt_artifact_id=record.srt_artifact_id,
                )
                self._write_record(failed)


class LocalTranscriptionJobStore(_LocalJsonJobStore[TranscriptionJobRecord]):
    def __init__(self, root_dir: Path) -> None:
        super().__init__(
            root_dir,
            job_id_validator=validate_transcription_job_id,
            record_from_json=TranscriptionJobRecord.from_json_dict,
        )

    def create(self, prepared: PreparedTranscriptionJob) -> TranscriptionJobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = TranscriptionJobRecord(
                job_id=job_id,
                state=TranscriptionJobState.QUEUED,
                audio_artifact_id=prepared.audio_record.artifact_id,
                include_word_segments=prepared.include_word_segments,
                created_at_unix_seconds=time.time(),
            )
            self._write_record(record)
            return record

    def fail_incomplete_jobs(self, message: str) -> None:
        with self._lock:
            for record in self._iter_records():
                if record.state not in {TranscriptionJobState.QUEUED, TranscriptionJobState.RUNNING}:
                    continue
                failed = TranscriptionJobRecord(
                    job_id=record.job_id,
                    state=TranscriptionJobState.FAILED,
                    audio_artifact_id=record.audio_artifact_id,
                    include_word_segments=record.include_word_segments,
                    created_at_unix_seconds=record.created_at_unix_seconds,
                    started_at_unix_seconds=record.started_at_unix_seconds,
                    finished_at_unix_seconds=time.time(),
                    error_code="dictator.jobs.interrupted",
                    error_message=message,
                    text=record.text,
                    language_code=record.language_code,
                    words=record.words,
                )
                self._write_record(failed)


class LocalDiarizationJobStore(_LocalJsonJobStore[DiarizationJobRecord]):
    def __init__(self, root_dir: Path) -> None:
        super().__init__(
            root_dir,
            job_id_validator=validate_diarization_job_id,
            record_from_json=DiarizationJobRecord.from_json_dict,
        )

    def create(self, prepared: PreparedDiarizationJob) -> DiarizationJobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = DiarizationJobRecord(
                job_id=job_id,
                state=DiarizationJobState.QUEUED,
                audio_artifact_id=prepared.audio_record.artifact_id,
                include_words=prepared.include_words,
                include_utterances=prepared.include_utterances,
                include_speakers=prepared.include_speakers,
                include_speaker_segments=prepared.include_speaker_segments,
                persist_json_artifact=prepared.persist_json_artifact,
                created_at_unix_seconds=time.time(),
            )
            self._write_record(record)
            return record

    def fail_incomplete_jobs(self, message: str) -> None:
        with self._lock:
            for record in self._iter_records():
                if record.state not in {DiarizationJobState.QUEUED, DiarizationJobState.RUNNING}:
                    continue
                failed = DiarizationJobRecord(
                    job_id=record.job_id,
                    state=DiarizationJobState.FAILED,
                    audio_artifact_id=record.audio_artifact_id,
                    include_words=record.include_words,
                    include_utterances=record.include_utterances,
                    include_speakers=record.include_speakers,
                    include_speaker_segments=record.include_speaker_segments,
                    persist_json_artifact=record.persist_json_artifact,
                    created_at_unix_seconds=record.created_at_unix_seconds,
                    started_at_unix_seconds=record.started_at_unix_seconds,
                    finished_at_unix_seconds=time.time(),
                    error_code="dictator.jobs.interrupted",
                    error_message=message,
                    text=record.text,
                    language_code=record.language_code,
                    diarization=record.diarization,
                    diarization_artifact_id=record.diarization_artifact_id,
                )
                self._write_record(failed)


class LocalSubtitleJobStore(_LocalJsonJobStore[SubtitleJobRecord]):
    def __init__(self, root_dir: Path) -> None:
        super().__init__(
            root_dir,
            job_id_validator=validate_subtitle_job_id,
            record_from_json=SubtitleJobRecord.from_json_dict,
        )

    def create(self, prepared: PreparedSubtitleJob) -> SubtitleJobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = SubtitleJobRecord(
                job_id=job_id,
                state=SubtitleJobState.QUEUED,
                audio_artifact_id=prepared.audio_record.artifact_id,
                include_srt_text=prepared.include_srt_text,
                created_at_unix_seconds=time.time(),
            )
            self._write_record(record)
            return record

    def fail_incomplete_jobs(self, message: str) -> None:
        with self._lock:
            for record in self._iter_records():
                if record.state not in {SubtitleJobState.QUEUED, SubtitleJobState.RUNNING}:
                    continue
                failed = SubtitleJobRecord(
                    job_id=record.job_id,
                    state=SubtitleJobState.FAILED,
                    audio_artifact_id=record.audio_artifact_id,
                    include_srt_text=record.include_srt_text,
                    created_at_unix_seconds=record.created_at_unix_seconds,
                    started_at_unix_seconds=record.started_at_unix_seconds,
                    finished_at_unix_seconds=time.time(),
                    error_code="dictator.jobs.interrupted",
                    error_message=message,
                    language_code=record.language_code,
                    mode=record.mode,
                    output_format=record.output_format,
                    granularity=record.granularity,
                    group_size=record.group_size,
                    cues=record.cues,
                    srt_text=record.srt_text,
                    srt_artifact_id=record.srt_artifact_id,
                )
                self._write_record(failed)


class LocalExtractReferenceSampleJobStore(_LocalJsonJobStore[ExtractReferenceSampleJobRecord]):
    def __init__(self, root_dir: Path) -> None:
        super().__init__(
            root_dir,
            job_id_validator=validate_extract_reference_sample_job_id,
            record_from_json=ExtractReferenceSampleJobRecord.from_json_dict,
        )

    def create(
        self,
        prepared: PreparedExtractReferenceSampleJob,
    ) -> ExtractReferenceSampleJobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = ExtractReferenceSampleJobRecord(
                job_id=job_id,
                state=ExtractReferenceSampleJobState.QUEUED,
                source_artifact_id=prepared.source_record.artifact_id,
                created_at_unix_seconds=time.time(),
            )
            self._write_record(record)
            return record

    def fail_incomplete_jobs(self, message: str) -> None:
        with self._lock:
            for record in self._iter_records():
                if record.state not in {
                    ExtractReferenceSampleJobState.QUEUED,
                    ExtractReferenceSampleJobState.RUNNING,
                }:
                    continue
                failed = ExtractReferenceSampleJobRecord(
                    job_id=record.job_id,
                    state=ExtractReferenceSampleJobState.FAILED,
                    source_artifact_id=record.source_artifact_id,
                    created_at_unix_seconds=record.created_at_unix_seconds,
                    started_at_unix_seconds=record.started_at_unix_seconds,
                    finished_at_unix_seconds=time.time(),
                    error_code="dictator.jobs.interrupted",
                    error_message=message,
                    sample_artifact_id=record.sample_artifact_id,
                    trim_start_seconds=record.trim_start_seconds,
                    trim_end_seconds=record.trim_end_seconds,
                    window_start_seconds=record.window_start_seconds,
                    window_end_seconds=record.window_end_seconds,
                    dominant_speaker_word_count=record.dominant_speaker_word_count,
                )
                self._write_record(failed)


class _QueuedJobManager(Generic[PreparedT, RecordT]):
    def __init__(self, *, job_store, max_workers: int, max_pending_jobs: int, thread_name_prefix: str) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if max_pending_jobs <= 0:
            raise ValueError("max_pending_jobs must be positive")
        self.job_store = job_store
        self.max_pending_jobs = max_pending_jobs
        self._lock = threading.Lock()
        self._pending_jobs = 0
        self._futures: dict[str, object] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self.job_store.fail_incomplete_jobs("service restarted before the job completed")

    def submit(self, prepared: PreparedT) -> RecordT:
        with self._lock:
            if self._pending_jobs >= self.max_pending_jobs:
                raise ServiceRequestError(
                    None,
                    "dictator.jobs.queue_full",
                    "too many queued jobs",
                )
            self._pending_jobs += 1
        try:
            record = self._create_record(prepared)
            future = self._executor.submit(self._run_job_wrapper, record.job_id, prepared)
            should_track = False
            with self._lock:
                if not getattr(future, "done", lambda: False)():
                    self._futures[record.job_id] = future
                    should_track = True
            add_done_callback = getattr(future, "add_done_callback", None)
            if should_track and add_done_callback is not None:
                add_done_callback(lambda _future, job_id=record.job_id: self._discard_future(job_id))
            return record
        except Exception:
            with self._lock:
                self._pending_jobs -= 1
            raise

    def get(self, job_id: str) -> RecordT:
        return self.job_store.get(job_id)

    def cancel(self, job_id: str) -> RecordT:
        record = self.job_store.cancel(job_id)
        if record.state == JobState.CANCELED:
            with self._lock:
                future = self._futures.get(record.job_id)
                cancel = getattr(future, "cancel", None)
                if cancel is not None and cancel():
                    self._futures.pop(record.job_id, None)
                    if self._pending_jobs > 0:
                        self._pending_jobs -= 1
        return record

    def _create_record(self, prepared: PreparedT) -> RecordT:
        raise NotImplementedError

    def _run_job(self, job_id: str, prepared: PreparedT) -> None:
        raise NotImplementedError

    def _run_job_wrapper(self, job_id: str, prepared: PreparedT) -> None:
        try:
            self._run_job(job_id, prepared)
        finally:
            self._discard_future(job_id)

    def _discard_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)


class SynthesisJobManager(_QueuedJobManager[PreparedSynthesisRequest, SynthesisJobRecord]):
    """Queue and run synthesis jobs in background worker threads."""

    def __init__(
        self,
        *,
        job_store: LocalSynthesisJobStore,
        artifact_store: LocalArtifactStore,
        execution_runtime,
        max_workers: int,
        max_pending_jobs: int,
    ) -> None:
        self.artifact_store = artifact_store
        self.execution_runtime = execution_runtime
        super().__init__(
            job_store=job_store,
            max_workers=max_workers,
            max_pending_jobs=max_pending_jobs,
            thread_name_prefix="dictator-synthesis-job",
        )

    def _create_record(self, prepared: PreparedSynthesisRequest) -> SynthesisJobRecord:
        return self.job_store.create(prepared)

    def _run_job(self, job_id: str, prepared: PreparedSynthesisRequest) -> None:
        def update_progress(completed_chunks: int, estimated_total_chunks: int) -> None:
            self.job_store.update(
                job_id,
                completed_chunks=completed_chunks,
                estimated_total_chunks=estimated_total_chunks,
            )

        try:
            self.job_store.update(
                job_id,
                state=SynthesisJobState.RUNNING.value,
                started_at_unix_seconds=time.time(),
            )
            outcome = execute_synthesis_request(
                artifact_store=self.artifact_store,
                execution_runtime=self.execution_runtime,
                prepared=prepared,
                progress_callback=update_progress,
            )
        except DictatorError as exc:
            self.job_store.update(
                job_id,
                state=SynthesisJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive logging around worker threads
            logging.exception("synthesis job %s failed", job_id)
            self.job_store.update(
                job_id,
                state=SynthesisJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code="dictator.jobs.failed",
                error_message=str(exc),
            )
        else:
            self.job_store.update(
                job_id,
                state=SynthesisJobState.SUCCEEDED.value,
                finished_at_unix_seconds=time.time(),
                audio_artifact_id=outcome.audio_record.artifact_id,
                audio_duration_seconds=outcome.audio_duration_seconds,
                audio_format=outcome.audio_format,
                timeline_artifact_id=outcome.timeline_artifact_id,
                chunk_count=outcome.chunk_count,
                estimated_total_chunks=outcome.chunk_count,
                completed_chunks=outcome.chunk_count,
            )
        finally:
            with self._lock:
                self._pending_jobs -= 1


class AlignmentJobManager(_QueuedJobManager[PreparedAlignmentJob, AlignmentJobRecord]):
    """Queue and run alignment jobs in background worker threads."""

    def __init__(
        self,
        *,
        job_store: LocalAlignmentJobStore,
        artifact_store: LocalArtifactStore,
        execution_runtime,
        max_workers: int,
        max_pending_jobs: int,
    ) -> None:
        self.artifact_store = artifact_store
        self.execution_runtime = execution_runtime
        super().__init__(
            job_store=job_store,
            max_workers=max_workers,
            max_pending_jobs=max_pending_jobs,
            thread_name_prefix="dictator-alignment-job",
        )

    def _create_record(self, prepared: PreparedAlignmentJob) -> AlignmentJobRecord:
        return self.job_store.create(prepared)

    def _run_job(self, job_id: str, prepared: PreparedAlignmentJob) -> None:
        try:
            self.job_store.update(
                job_id,
                state=AlignmentJobState.RUNNING.value,
                started_at_unix_seconds=time.time(),
            )
            audio_record = prepared.audio_record
            reservation = self.artifact_store.reserve_artifact(
                f"{Path(audio_record.filename).stem}.srt",
                media_type="application/x-subrip",
                fallback_suffix=".srt",
            )
            service = self.execution_runtime.get_alignment_service()
            result = service.align(
                AlignTranscriptRequest(
                    audio_path=audio_record.path,
                    transcript_text=prepared.transcript_text,
                    language=prepared.language_code,
                    remove_punctuation=prepared.remove_punctuation,
                    transcript_source_name=prepared.transcript_source_name,
                    output_srt_path=reservation.path,
                )
            )
            srt_record = self.artifact_store.finalize_artifact(reservation)
        except DictatorError as exc:
            self.job_store.update(
                job_id,
                state=AlignmentJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive logging around worker threads
            logging.exception("alignment job %s failed", job_id)
            self.job_store.update(
                job_id,
                state=AlignmentJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code="dictator.jobs.failed",
                error_message=str(exc),
            )
        else:
            self.job_store.update(
                job_id,
                state=AlignmentJobState.SUCCEEDED.value,
                finished_at_unix_seconds=time.time(),
                language_code=result.language,
                words=_alignment_words_to_json(result.words),
                srt_text=result.srt_text if prepared.include_srt_text else None,
                srt_artifact_id=srt_record.artifact_id,
            )
        finally:
            with self._lock:
                self._pending_jobs -= 1


class TranscriptionJobManager(_QueuedJobManager[PreparedTranscriptionJob, TranscriptionJobRecord]):
    def __init__(
        self,
        *,
        job_store: LocalTranscriptionJobStore,
        artifact_store: LocalArtifactStore,
        execution_runtime,
        max_workers: int,
        max_pending_jobs: int,
    ) -> None:
        self.artifact_store = artifact_store
        self.execution_runtime = execution_runtime
        super().__init__(
            job_store=job_store,
            max_workers=max_workers,
            max_pending_jobs=max_pending_jobs,
            thread_name_prefix="dictator-transcription-job",
        )

    def _create_record(self, prepared: PreparedTranscriptionJob) -> TranscriptionJobRecord:
        return self.job_store.create(prepared)

    def _run_job(self, job_id: str, prepared: PreparedTranscriptionJob) -> None:
        try:
            self.job_store.update(
                job_id,
                state=TranscriptionJobState.RUNNING.value,
                started_at_unix_seconds=time.time(),
            )
            transcription_service = self.execution_runtime.get_transcription_service()
            result: TranscriptionResult = transcription_service.transcribe(
                prepared.audio_record.path,
                language=prepared.language_code,
                model_size=prepared.model_size,
            )
        except DictatorError as exc:
            self.job_store.update(
                job_id,
                state=TranscriptionJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover
            logging.exception("transcription job %s failed", job_id)
            self.job_store.update(
                job_id,
                state=TranscriptionJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code="dictator.jobs.failed",
                error_message=str(exc),
            )
        else:
            self.job_store.update(
                job_id,
                state=TranscriptionJobState.SUCCEEDED.value,
                finished_at_unix_seconds=time.time(),
                text=result.text,
                language_code=result.language,
                words=_transcription_words_to_json(result.words if prepared.include_word_segments else ()),
            )
        finally:
            with self._lock:
                self._pending_jobs -= 1


class DiarizationJobManager(_QueuedJobManager[PreparedDiarizationJob, DiarizationJobRecord]):
    def __init__(
        self,
        *,
        job_store: LocalDiarizationJobStore,
        artifact_store: LocalArtifactStore,
        execution_runtime,
        max_workers: int,
        max_pending_jobs: int,
    ) -> None:
        self.artifact_store = artifact_store
        self.execution_runtime = execution_runtime
        super().__init__(
            job_store=job_store,
            max_workers=max_workers,
            max_pending_jobs=max_pending_jobs,
            thread_name_prefix="dictator-diarization-job",
        )

    def _create_record(self, prepared: PreparedDiarizationJob) -> DiarizationJobRecord:
        return self.job_store.create(prepared)

    def _run_job(self, job_id: str, prepared: PreparedDiarizationJob) -> None:
        try:
            self.job_store.update(
                job_id,
                state=DiarizationJobState.RUNNING.value,
                started_at_unix_seconds=time.time(),
            )
            diarization_service = self.execution_runtime.get_diarization_service()
            result: DiarizeAudioResult = diarization_service.diarize(
                DiarizeAudioRequest(
                    input_path=prepared.audio_record.path,
                    language=prepared.language_code,
                    model_size=prepared.model_size,
                    include_words=prepared.include_words,
                    include_utterances=prepared.include_utterances,
                    include_speakers=prepared.include_speakers,
                    include_speaker_segments=prepared.include_speaker_segments,
                    utterance_gap_seconds=prepared.utterance_gap_seconds,
                ),
                model=self.execution_runtime.get_whisper_model(prepared.model_size),
                diarization_pipeline=self.execution_runtime.get_diarization_pipeline(),
            )
            payload = result.to_json_dict(
                include_words=prepared.include_words,
                include_utterances=prepared.include_utterances,
                include_speakers=prepared.include_speakers,
                include_speaker_segments=prepared.include_speaker_segments,
            )
            diarization_artifact_id = None
            if prepared.persist_json_artifact:
                json_record = self.artifact_store.write_artifact(
                    [json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")],
                    filename=f"{Path(prepared.audio_record.filename).stem}.diarization.json",
                    media_type="application/json",
                    fallback_suffix=".json",
                )
                diarization_artifact_id = json_record.artifact_id
        except DictatorError as exc:
            self.job_store.update(
                job_id,
                state=DiarizationJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover
            logging.exception("diarization job %s failed", job_id)
            self.job_store.update(
                job_id,
                state=DiarizationJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code="dictator.jobs.failed",
                error_message=str(exc),
            )
        else:
            self.job_store.update(
                job_id,
                state=DiarizationJobState.SUCCEEDED.value,
                finished_at_unix_seconds=time.time(),
                text=result.text,
                language_code=result.language,
                diarization=payload,
                diarization_artifact_id=diarization_artifact_id,
            )
        finally:
            with self._lock:
                self._pending_jobs -= 1


class SubtitleJobManager(_QueuedJobManager[PreparedSubtitleJob, SubtitleJobRecord]):
    def __init__(
        self,
        *,
        job_store: LocalSubtitleJobStore,
        artifact_store: LocalArtifactStore,
        execution_runtime,
        max_workers: int,
        max_pending_jobs: int,
    ) -> None:
        self.artifact_store = artifact_store
        self.execution_runtime = execution_runtime
        super().__init__(
            job_store=job_store,
            max_workers=max_workers,
            max_pending_jobs=max_pending_jobs,
            thread_name_prefix="dictator-subtitle-job",
        )

    def _create_record(self, prepared: PreparedSubtitleJob) -> SubtitleJobRecord:
        return self.job_store.create(prepared)

    def _run_job(self, job_id: str, prepared: PreparedSubtitleJob) -> None:
        try:
            self.job_store.update(
                job_id,
                state=SubtitleJobState.RUNNING.value,
                started_at_unix_seconds=time.time(),
            )
            srt_reservation = self.artifact_store.reserve_artifact(
                f"{Path(prepared.audio_record.filename).stem}.grouped.srt",
                media_type="application/x-subrip",
                fallback_suffix=".srt",
            )
            subtitle_service = self.execution_runtime.get_subtitle_service()
            needs_whisper_model = prepared.source_text is None or prepared.language_code is None
            whisper_model = None
            if needs_whisper_model:
                whisper_model = self.execution_runtime.get_whisper_model(prepared.model_size)
            result = subtitle_service.render(
                RenderSubtitlesRequest(
                    audio_path=prepared.audio_record.path,
                    language=prepared.language_code,
                    model_size=prepared.model_size,
                    output_format="srt",
                    granularity=prepared.granularity,
                    group_size=prepared.group_size,
                    source_text=prepared.source_text,
                    source_text_name=prepared.source_text_name,
                    output_srt_path=srt_reservation.path,
                ),
                model=whisper_model,
            )
            srt_record = self.artifact_store.finalize_artifact(srt_reservation)
        except DictatorError as exc:
            self.job_store.update(
                job_id,
                state=SubtitleJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover
            logging.exception("subtitle job %s failed", job_id)
            self.job_store.update(
                job_id,
                state=SubtitleJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code="dictator.jobs.failed",
                error_message=str(exc),
            )
        else:
            self.job_store.update(
                job_id,
                state=SubtitleJobState.SUCCEEDED.value,
                finished_at_unix_seconds=time.time(),
                language_code=result.language,
                mode=result.mode,
                output_format=result.output_format,
                granularity=result.granularity,
                group_size=result.group_size,
                cues=_subtitle_cues_to_json(result.cues),
                srt_text=result.srt_text if prepared.include_srt_text else None,
                srt_artifact_id=srt_record.artifact_id,
            )
        finally:
            with self._lock:
                self._pending_jobs -= 1


class ExtractReferenceSampleJobManager(
    _QueuedJobManager[PreparedExtractReferenceSampleJob, ExtractReferenceSampleJobRecord]
):
    def __init__(
        self,
        *,
        job_store: LocalExtractReferenceSampleJobStore,
        artifact_store: LocalArtifactStore,
        execution_runtime,
        max_workers: int,
        max_pending_jobs: int,
    ) -> None:
        self.artifact_store = artifact_store
        self.execution_runtime = execution_runtime
        super().__init__(
            job_store=job_store,
            max_workers=max_workers,
            max_pending_jobs=max_pending_jobs,
            thread_name_prefix="dictator-reference-extraction-job",
        )

    def _create_record(
        self,
        prepared: PreparedExtractReferenceSampleJob,
    ) -> ExtractReferenceSampleJobRecord:
        return self.job_store.create(prepared)

    def _run_job(self, job_id: str, prepared: PreparedExtractReferenceSampleJob) -> None:
        try:
            self.job_store.update(
                job_id,
                state=ExtractReferenceSampleJobState.RUNNING.value,
                started_at_unix_seconds=time.time(),
            )
            reservation = self.artifact_store.reserve_artifact(
                f"{Path(prepared.source_record.filename).stem}_reference.wav",
                media_type="audio/wav",
                fallback_suffix=".wav",
            )
            from dictator.extraction.models import ReferenceExtractionRequest

            extraction_service = self.execution_runtime.get_reference_extraction_service()
            result = extraction_service.extract(
                ReferenceExtractionRequest(
                    input_path=prepared.source_record.path,
                    output_path=reservation.path,
                    model_size=prepared.model_size,
                    language=prepared.language_code,
                    duration_seconds=prepared.duration_seconds,
                    max_speech_rate=prepared.max_speech_rate,
                    min_centroid_hz=prepared.min_centroid_hz,
                    max_centroid_hz=prepared.max_centroid_hz,
                ),
                model=self.execution_runtime.get_whisper_model(prepared.model_size),
                diarization_pipeline=self.execution_runtime.get_diarization_pipeline(),
            )
            sample_record = self.artifact_store.finalize_artifact(reservation)
        except DictatorError as exc:
            self.job_store.update(
                job_id,
                state=ExtractReferenceSampleJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover
            logging.exception("reference extraction job %s failed", job_id)
            self.job_store.update(
                job_id,
                state=ExtractReferenceSampleJobState.FAILED.value,
                finished_at_unix_seconds=time.time(),
                error_code="dictator.jobs.failed",
                error_message=str(exc),
            )
        else:
            self.job_store.update(
                job_id,
                state=ExtractReferenceSampleJobState.SUCCEEDED.value,
                finished_at_unix_seconds=time.time(),
                sample_artifact_id=sample_record.artifact_id,
                trim_start_seconds=result.trim_start_seconds,
                trim_end_seconds=result.trim_end_seconds,
                window_start_seconds=result.window_start_seconds,
                window_end_seconds=result.window_end_seconds,
                dominant_speaker_word_count=len(result.dominant_speaker_words),
            )
        finally:
            with self._lock:
                self._pending_jobs -= 1
