"""Background synthesis jobs and their persistent status store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import threading
import time
import uuid

from dictator.runtime.errors import DictatorError, ServiceRequestError
from dictator.storage import LocalArtifactStore
from dictator.synthesis.workflow import PreparedSynthesisRequest, execute_synthesis_request


class SynthesisJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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
    timeline_artifact_id: str | None = None
    chunk_count: int | None = None

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
            timeline_artifact_id=_optional_str(payload.get("timeline_artifact_id")),
            chunk_count=_optional_int(payload.get("chunk_count")),
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


class LocalSynthesisJobStore:
    """Persist synthesis job status under a local root directory."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _job_path(self, job_id: str) -> Path:
        return self.root_dir / f"{job_id}.json"

    def _write_record(self, record: SynthesisJobRecord) -> None:
        path = self._job_path(record.job_id)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(record.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

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
            )
            self._write_record(record)
            return record

    def get(self, job_id: str) -> SynthesisJobRecord:
        path = self._job_path(job_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SynthesisJobRecord.from_json_dict(payload)

    def update(self, job_id: str, **updates: object) -> SynthesisJobRecord:
        with self._lock:
            current = self.get(job_id)
            payload = current.to_json_dict()
            payload.update(updates)
            record = SynthesisJobRecord.from_json_dict(payload)
            self._write_record(record)
            return record

    def fail_incomplete_jobs(self, message: str) -> None:
        with self._lock:
            for path in self.root_dir.glob("*.json"):
                record = SynthesisJobRecord.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
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
                    timeline_artifact_id=record.timeline_artifact_id,
                    chunk_count=record.chunk_count,
                )
                self._write_record(failed)


class SynthesisJobManager:
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
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if max_pending_jobs <= 0:
            raise ValueError("max_pending_jobs must be positive")
        self.job_store = job_store
        self.artifact_store = artifact_store
        self.execution_runtime = execution_runtime
        self.max_pending_jobs = max_pending_jobs
        self._lock = threading.Lock()
        self._pending_jobs = 0
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dictator-synthesis-job")
        self.job_store.fail_incomplete_jobs("service restarted before the job completed")

    def submit(self, prepared: PreparedSynthesisRequest) -> SynthesisJobRecord:
        with self._lock:
            if self._pending_jobs >= self.max_pending_jobs:
                raise ServiceRequestError(
                    None,
                    "dictator.jobs.queue_full",
                    "too many queued synthesis jobs",
                )
            self._pending_jobs += 1
        try:
            record = self.job_store.create(prepared)
            self._executor.submit(self._run_job, record.job_id, prepared)
            return record
        except Exception:
            with self._lock:
                self._pending_jobs -= 1
            raise

    def get(self, job_id: str) -> SynthesisJobRecord:
        return self.job_store.get(job_id)

    def _run_job(self, job_id: str, prepared: PreparedSynthesisRequest) -> None:
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
                timeline_artifact_id=outcome.timeline_artifact_id,
                chunk_count=outcome.chunk_count,
            )
        finally:
            with self._lock:
                self._pending_jobs -= 1
