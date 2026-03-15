"""Shared polling helpers for Dictator async client jobs."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

JobT = TypeVar("JobT")


class RemoteJobFailedError(RuntimeError):
    def __init__(self, *, job_id: str, state: str, error_code: str, error_message: str) -> None:
        self.job_id = job_id
        self.state = state
        self.error_code = error_code
        self.error_message = error_message
        detail = error_message or "remote job failed"
        if error_code:
            detail = f"{error_code}: {detail}"
        super().__init__(detail)


def wait_for_job(
    fetch_job: Callable[[], JobT],
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 1.0,
) -> JobT:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or None")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")

    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while True:
        job = fetch_job()
        state = str(getattr(job, "state", ""))
        if state.endswith("_SUCCEEDED"):
            return job
        if state.endswith("_FAILED"):
            raise RemoteJobFailedError(
                job_id=str(getattr(job, "job_id", "")),
                state=state,
                error_code=str(getattr(job, "error_code", "")),
                error_message=str(getattr(job, "error_message", "")),
            )
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"job {getattr(job, 'job_id', '')} did not complete within {timeout_seconds} seconds")
        time.sleep(poll_interval_seconds)
