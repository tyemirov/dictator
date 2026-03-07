"""Shared service error types."""

from __future__ import annotations


class DictatorError(RuntimeError):
    """Base error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ValidationError(DictatorError):
    """Validation error for malformed input."""


class DependencyError(DictatorError):
    """Raised when a runtime dependency is missing or incompatible."""


class ProcessingError(DictatorError):
    """Raised when audio processing fails after validation succeeds."""


class ServiceRequestError(DictatorError):
    """Transport-facing request error with an optional status field."""

    def __init__(self, status: object | None, code: str, message: str) -> None:
        super().__init__(code, message)
        self.status = status
