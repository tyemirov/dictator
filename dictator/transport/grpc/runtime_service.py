"""Runtime metrics gRPC servicer."""

from __future__ import annotations

from dictator.speech.v1 import runtime_pb2, runtime_pb2_grpc

from .base import BaseServicer


class RuntimeServiceServicer(BaseServicer, runtime_pb2_grpc.RuntimeServiceServicer):
    def GetMetrics(self, request, context):
        with self._request_scope(context, is_inquiry=True):
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
        with self._request_scope(context, is_inquiry=True):
            components = []
            last_error = ""
            all_ready = True

            components.append(runtime_pb2.ReadinessComponent(
                name="grpc_transport",
                ready=True,
                detail="accepting requests",
            ))

            try:
                runtime = self.service_context.execution_runtime
                if runtime is not None:
                    components.append(runtime_pb2.ReadinessComponent(
                        name="execution_runtime",
                        ready=True,
                        detail="initialized",
                    ))
                else:
                    all_ready = False
                    components.append(runtime_pb2.ReadinessComponent(
                        name="execution_runtime",
                        ready=False,
                        detail="not initialized",
                    ))
            except Exception as exc:
                all_ready = False
                last_error = str(exc)
                components.append(runtime_pb2.ReadinessComponent(
                    name="execution_runtime",
                    ready=False,
                    detail=str(exc),
                ))

            return runtime_pb2.GetReadinessResponse(
                ready=all_ready,
                warmup_started=True,
                warmup_in_progress=False,
                components=components,
                last_error=last_error,
            )
