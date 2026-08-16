#!/usr/bin/env python
"""Verify the hosted Dictator gRPC response contract."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dictator.speech.v1 import runtime_pb2, runtime_pb2_grpc, transcription_pb2, transcription_pb2_grpc
from dictator.transport.grpc.transcription_service import DIARIZATION_JOB_REQUIRED_ERROR_CODE

AUTH_TOKEN_ENV = "DICTATOR_GRPC_AUTH_TOKEN"
HOSTED_GRPC_ADDRESS = "dictator.mprlab.com:443"
RPC_TIMEOUT_SECONDS = 10


def _required_auth_token() -> str:
    token = os.environ.get(AUTH_TOKEN_ENV, "")
    if not token:
        raise RuntimeError(f"{AUTH_TOKEN_ENV} is required")
    return token


def _verify_sync_diarization_status(channel: grpc.Channel, metadata: tuple[tuple[str, str], ...]) -> None:
    stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)
    try:
        stub.DiarizeAudio(
            transcription_pb2.DiarizeAudioRequest(),
            metadata=metadata,
            timeout=RPC_TIMEOUT_SECONDS,
        )
    except grpc.RpcError as error:
        if error.code() != grpc.StatusCode.FAILED_PRECONDITION:
            raise AssertionError(
                f"DiarizeAudio returned {error.code().name}, not FAILED_PRECONDITION"
            ) from error
        trailing_metadata = dict(error.trailing_metadata() or ())
        error_code = trailing_metadata.get("x-dictator-error-code", "")
        if error_code != DIARIZATION_JOB_REQUIRED_ERROR_CODE:
            raise AssertionError(
                "DiarizeAudio did not return the typed job-required error"
            ) from error
        return
    raise AssertionError("DiarizeAudio returned an unexpected synchronous result")


def main() -> int:
    token = _required_auth_token()
    metadata = (("authorization", f"Bearer {token}"),)
    channel = grpc.secure_channel(HOSTED_GRPC_ADDRESS, grpc.ssl_channel_credentials())
    try:
        health = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(service=""),
            timeout=RPC_TIMEOUT_SECONDS,
        )
        if health.status != health_pb2.HealthCheckResponse.SERVING:
            raise AssertionError(f"hosted gRPC health returned status {health.status}")
        runtime_pb2_grpc.RuntimeServiceStub(channel).GetMetrics(
            runtime_pb2.GetMetricsRequest(),
            metadata=metadata,
            timeout=RPC_TIMEOUT_SECONDS,
        )
        _verify_sync_diarization_status(channel, metadata)
    finally:
        channel.close()
    print("Hosted Dictator gRPC health, content type, authentication, and typed status passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
