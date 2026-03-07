"""Inflight request limiting helpers."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator

from .errors import ServiceRequestError


class InflightLimiter:
    """Thread-safe inflight limiter for future service transports."""

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._lock = threading.Lock()
        self._inflight = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @contextmanager
    def acquire(self) -> Iterator[None]:
        with self._lock:
            if self._inflight >= self._limit:
                raise ServiceRequestError(
                    None,
                    "dictator.runtime.inflight_limit",
                    "too many inflight requests",
                )
            self._inflight += 1
        try:
            yield
        finally:
            with self._lock:
                self._inflight -= 1
