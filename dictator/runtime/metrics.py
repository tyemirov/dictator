"""Thread-safe request metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Callable


@dataclass(frozen=True)
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""

    requests_total: int
    requests_succeeded: int
    requests_failed: int
    inflight: int
    bytes_received: int
    uptime_seconds: float
    average_latency_seconds: float
    max_latency_seconds: float


@dataclass
class MetricsRegistry:
    """Accumulates metrics across service requests."""

    clock: Callable[[], float] = time.monotonic
    started_at: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    requests_total: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    inflight: int = 0
    bytes_received: int = 0
    total_latency_seconds: float = 0.0
    max_latency_seconds: float = 0.0

    def record_start(self) -> None:
        with self.lock:
            self.requests_total += 1
            self.inflight += 1

    def record_bytes(self, received_bytes: int) -> None:
        with self.lock:
            self.bytes_received += max(0, received_bytes)

    def record_finish(self, success: bool, latency_seconds: float) -> None:
        with self.lock:
            self.inflight -= 1
            if success:
                self.requests_succeeded += 1
            else:
                self.requests_failed += 1
            self.total_latency_seconds += latency_seconds
            if latency_seconds > self.max_latency_seconds:
                self.max_latency_seconds = latency_seconds

    def snapshot(self) -> MetricsSnapshot:
        with self.lock:
            average_latency = (
                self.total_latency_seconds / self.requests_total
                if self.requests_total
                else 0.0
            )
            return MetricsSnapshot(
                requests_total=self.requests_total,
                requests_succeeded=self.requests_succeeded,
                requests_failed=self.requests_failed,
                inflight=self.inflight,
                bytes_received=self.bytes_received,
                uptime_seconds=max(0.0, self.clock() - self.started_at),
                average_latency_seconds=average_latency,
                max_latency_seconds=self.max_latency_seconds,
            )
