"""Shared gRPC servicer helpers."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

import grpc

from dictator.runtime import (
    DependencyError,
    ProcessingError,
    ServiceRequestError,
    ValidationError,
)
from dictator.speech.v1 import common_pb2, subtitle_pb2
from dictator.storage import ArtifactRecord

from .context import ServiceContext

ERROR_CODE_METADATA = "x-dictator-error-code"
AUTH_HEADER = "authorization"
TOKEN_HEADER = "x-dictator-token"
DEFAULT_MODEL_SIZE = "base"


class BaseServicer:
    def __init__(self, service_context: ServiceContext) -> None:
        self.service_context = service_context

    def _ensure_request_active(self, context: grpc.ServicerContext) -> None:
        if not context.is_active():
            self._abort(
                context,
                grpc.StatusCode.CANCELLED,
                "dictator.grpc.request.cancelled",
                "request is no longer active",
            )
        time_remaining = context.time_remaining()
        if time_remaining is not None and time_remaining <= 0:
            self._abort(
                context,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                "dictator.grpc.request.deadline_exceeded",
                "request deadline exceeded",
            )

    def _require_auth(self, context: grpc.ServicerContext) -> None:
        expected = self.service_context.auth_token
        if not expected:
            return
        metadata = {key.lower(): value for key, value in context.invocation_metadata()}
        presented = metadata.get(TOKEN_HEADER)
        authorization = metadata.get(AUTH_HEADER, "")
        if not presented and authorization.lower().startswith("bearer "):
            presented = authorization[7:]
        if presented != expected:
            self._abort(
                context,
                grpc.StatusCode.UNAUTHENTICATED,
                "dictator.grpc.auth.required",
                "missing or invalid auth token",
            )

    def _abort(
        self,
        context: grpc.ServicerContext,
        status: grpc.StatusCode,
        code: str,
        message: str,
    ) -> None:
        context.set_trailing_metadata(((ERROR_CODE_METADATA, code),))
        context.abort(status, message)

    @contextmanager
    def _request_scope(
        self,
        context: grpc.ServicerContext,
        bytes_received: int = 0,
    ) -> Iterator[None]:
        started_at = time.monotonic()
        success = False
        self.service_context.metrics.record_start()
        if bytes_received:
            self.service_context.metrics.record_bytes(bytes_received)
        try:
            with self.service_context.limiter.acquire():
                try:
                    self._ensure_request_active(context)
                    self._require_auth(context)
                    yield
                    self._ensure_request_active(context)
                    success = True
                except ValidationError as exc:
                    self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, exc.code, str(exc))
                except DependencyError as exc:
                    self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, exc.code, str(exc))
                except ProcessingError as exc:
                    self._abort(context, grpc.StatusCode.INTERNAL, exc.code, str(exc))
                except FileNotFoundError as exc:
                    self._abort(
                        context,
                        grpc.StatusCode.NOT_FOUND,
                        "dictator.artifact.not_found",
                        str(exc),
                    )
                except ValueError as exc:
                    self._abort(
                        context,
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "dictator.input.invalid",
                        str(exc),
                    )
        except ServiceRequestError as exc:
            self._abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, exc.code, str(exc))
        finally:
            self.service_context.metrics.record_finish(success, time.monotonic() - started_at)

    def _artifact_ref(self, record: ArtifactRecord) -> common_pb2.ArtifactRef:
        return common_pb2.ArtifactRef(
            artifact_id=record.artifact_id,
            filename=record.filename,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
        )

    def _word_segment(self, payload: dict[str, object]) -> common_pb2.WordSegment:
        return common_pb2.WordSegment(
            content=str(payload.get("content", "")),
            start_seconds=float(payload.get("start") or 0.0),
            end_seconds=float(payload.get("end") or 0.0),
        )

    def _timeline_segment(self, payload: dict[str, object]) -> common_pb2.TimelineSegment:
        return common_pb2.TimelineSegment(
            content=str(payload.get("content", "")),
            start_seconds=float(payload.get("start") or 0.0),
            end_seconds=float(payload.get("end") or 0.0),
        )

    def _subtitle_cue(self, cue) -> subtitle_pb2.SubtitleCue:
        return subtitle_pb2.SubtitleCue(
            content=cue.text,
            start_seconds=cue.start_seconds,
            end_seconds=cue.end_seconds,
            item_count=cue.item_count,
        )

    def _resolve_language_request(
        self,
        *,
        language_code: str,
        autodetect_language: bool,
        error_scope: str,
    ) -> str | None:
        normalized = language_code.strip()
        if normalized and autodetect_language:
            raise ValidationError(
                f"{error_scope}.language_conflict",
                "language_code and autodetect_language cannot both be set",
            )
        if not normalized and not autodetect_language:
            raise ValidationError(
                f"{error_scope}.language_required",
                "set language_code or autodetect_language=true",
            )
        if autodetect_language:
            return None
        return normalized
