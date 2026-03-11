"""Runtime metrics gRPC servicer."""

from __future__ import annotations

from dictator.speech.v1 import runtime_pb2, runtime_pb2_grpc

from .base import BaseServicer


class RuntimeServiceServicer(BaseServicer, runtime_pb2_grpc.RuntimeServiceServicer):
    def GetMetrics(self, request, context):
        with self._request_scope(context):
            snapshot = self.service_context.metrics.snapshot()
            return runtime_pb2.GetMetricsResponse(
                requests_total=snapshot.requests_total,
                requests_succeeded=snapshot.requests_succeeded,
                requests_failed=snapshot.requests_failed,
                inflight=snapshot.inflight,
                bytes_received=snapshot.bytes_received,
                uptime_seconds=snapshot.uptime_seconds,
                average_latency_seconds=snapshot.average_latency_seconds,
                max_latency_seconds=snapshot.max_latency_seconds,
            )

    def GetReadiness(self, request, context):
        with self._request_scope(context):
            snapshot = self.service_context.execution_runtime.readiness_snapshot()
            return runtime_pb2.GetReadinessResponse(
                ready=snapshot.ready,
                warmup_started=snapshot.warmup_started,
                warmup_in_progress=snapshot.warmup_in_progress,
                components=[
                    runtime_pb2.ReadinessComponent(
                        name=component.name,
                        ready=component.ready,
                        detail=component.detail,
                    )
                    for component in snapshot.components
                ],
                last_error=snapshot.last_error,
            )
