"""Small timeout helpers for local CLI workflows."""

from __future__ import annotations

from queue import Queue
import threading
from typing import Callable, TypeVar

_T = TypeVar("_T")


def run_with_timeout(
    timeout_seconds: float,
    task_name: str,
    func: Callable[..., _T],
    *args,
    **kwargs,
) -> _T:
    """Run a callable with a wall-clock timeout.

    This is intended for CLI compatibility paths where process-global signal
    timeouts are unsafe. The worker runs on a daemon thread so a timeout does
    not pin the process on exit.
    """
    if timeout_seconds <= 0:
        return func(*args, **kwargs)

    result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("result", func(*args, **kwargs)))
        except BaseException as exc:  # pragma: no cover - propagated to caller
            result_queue.put(("error", exc))

    thread = threading.Thread(
        target=worker,
        name=f"dictator-timeout-{task_name}",
        daemon=True,
    )
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"{task_name} exceeded {timeout_seconds}s")
    if result_queue.empty():
        raise RuntimeError(f"{task_name} finished without producing a result")
    status, payload = result_queue.get_nowait()
    if status == "error":
        raise payload
    return payload
