"""Native input measurements through the authenticated gRPC service."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import wave

import grpc
import numpy as np

from dictator.client import DictationClient
from dictator.runtime import InflightLimiter, MetricsRegistry
from dictator.runtime.errors import ProcessingError
from dictator.runtime.jobs import LocalTranscriptionJobStore, TranscriptionJobManager
from dictator.speech.v1 import (
    artifacts_pb2, artifacts_pb2_grpc, transcription_pb2, transcription_pb2_grpc,
)
from dictator.storage import LocalArtifactStore
from dictator.transcription.service import TranscriptionService
from dictator.transport.grpc.config import ServerConfig
from dictator.transport.grpc.context import ServiceContext
from dictator.transport.grpc.server import build_server


class MeasuredModel:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def transcribe(self, audio, **kwargs):
        self.inputs.append(audio)
        return {
            "language": "en",
            "segments": [{"words": [{"word": "hello", "start": 0.0, "end": 0.25}]}],
        }


class FailingModel(MeasuredModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio, **kwargs):
        self.started.set()
        if not self.release.wait(5):
            raise AssertionError("test did not release the controlled model")
        raise ProcessingError("test.model.failed", "controlled processing failure")


def audio_payload() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(32000)
        wav.writeframes(np.zeros((64002, 2), dtype=np.int16).tobytes())
    return output.getvalue()


@contextmanager
def transcription_server(root: Path, model: MeasuredModel):
    store = LocalArtifactStore(root / "artifacts")
    service = TranscriptionService(model_loader=lambda _: model)
    runtime = SimpleNamespace(get_transcription_service=lambda: service)
    manager = TranscriptionJobManager(
        job_store=LocalTranscriptionJobStore(root / "jobs"),
        artifact_store=store,
        execution_runtime=runtime,
        max_workers=1,
        max_pending_jobs=2,
    )
    context = ServiceContext(
        artifact_store=store, execution_runtime=runtime,
        metrics=MetricsRegistry(), limiter=InflightLimiter(4),
        auth_token="usage-test", download_chunk_bytes=4096,
        transcription_job_manager=manager,
    )
    server = build_server(
        ServerConfig(artifact_root=root / "artifacts", auth_token="usage-test"),
        service_context=context,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        yield channel
    finally:
        channel.close()
        server.stop(None).wait()
        manager._executor.shutdown(wait=True)


class InputAudioUsageIntegrationTests(unittest.TestCase):
    metadata = (("x-dictator-token", "usage-test"),)

    def assert_usage(self, response) -> None:
        usage = getattr(response, "input_audio_usage", None)
        self.assertIsNotNone(usage, "native processed-input measurement is absent")
        self.assertEqual(usage.sample_count, 32001)
        self.assertEqual(usage.sample_rate_hz, 16000)

    def upload(self, channel) -> str:
        response = artifacts_pb2_grpc.ArtifactServiceStub(channel).UploadArtifact(
            iter([
                artifacts_pb2.UploadArtifactChunk(metadata=artifacts_pb2.UploadArtifactMetadata(
                    filename="stereo.wav", media_type="audio/wav",
                )),
                artifacts_pb2.UploadArtifactChunk(content=audio_payload()),
            ]), metadata=self.metadata,
        )
        return response.artifact.artifact_id

    def test_direct_transcription_measures_actual_resampled_input_without_word_output(self):
        model = MeasuredModel()
        with tempfile.TemporaryDirectory() as directory, transcription_server(Path(directory), model) as channel:
            artifact_id = self.upload(channel)
            response = transcription_pb2_grpc.TranscriptionServiceStub(channel).Transcribe(
                transcription_pb2.TranscribeRequest(
                    audio_artifact_id=artifact_id, language_code="en", include_word_segments=False,
                ), metadata=self.metadata,
            )
            self.assertEqual(response.text, "hello")
            self.assertEqual(len(response.words), 0)
            self.assert_usage(response)
            self.assertEqual(model.inputs[0].shape, (32001,))
            self.assertEqual(model.inputs[0].dtype, np.float32)

    def test_job_and_python_client_retain_input_usage_across_server_restart(self):
        model = MeasuredModel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with transcription_server(root, model) as channel:
                client = DictationClient(channel, metadata=self.metadata)
                submitted = client.submit_dictate_bytes_job(
                    audio_payload(), filename="stereo.wav", language_code="en", include_word_segments=True,
                )
                finished = client.wait_for_dictation_job(submitted.job_id, timeout_seconds=5, poll_interval_seconds=0.01)
                self.assert_usage(finished.result)
                self.assertEqual(finished.result.words[0]["end"], 0.25)
            with transcription_server(root, model) as channel:
                stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)
                response = stub.GetTranscribeJob(
                    transcription_pb2.GetTranscribeJobRequest(job_id=submitted.job_id), metadata=self.metadata,
                )
                self.assert_usage(response)
                self.assertEqual(response.source_artifact_id, submitted.source_artifact_id)
                self.assertEqual(response.state, transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED)
                self.assertEqual(len(model.inputs), 1)
                self.assert_usage(DictationClient(channel, metadata=self.metadata).get_dictation_job(submitted.job_id).result)

    def test_unknown_usage_stays_absent_for_running_queued_canceled_and_failed_jobs(self):
        model = FailingModel()
        with tempfile.TemporaryDirectory() as directory, transcription_server(Path(directory), model) as channel:
            stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)
            request = transcription_pb2.TranscribeRequest(audio_artifact_id=self.upload(channel), language_code="en")
            running = stub.SubmitTranscribeJob(request, metadata=self.metadata)
            try:
                self.assertTrue(model.started.wait(5))
                response = stub.GetTranscribeJob(
                    transcription_pb2.GetTranscribeJobRequest(job_id=running.job_id), metadata=self.metadata,
                )
                self.assertEqual(response.state, transcription_pb2.TRANSCRIPTION_JOB_STATE_RUNNING)
                self.assertFalse(response.HasField("input_audio_usage"))
                queued = stub.SubmitTranscribeJob(request, metadata=self.metadata)
                response = stub.GetTranscribeJob(
                    transcription_pb2.GetTranscribeJobRequest(job_id=queued.job_id), metadata=self.metadata,
                )
                self.assertEqual(response.state, transcription_pb2.TRANSCRIPTION_JOB_STATE_QUEUED)
                self.assertFalse(response.HasField("input_audio_usage"))
                stub.CancelTranscribeJob(
                    transcription_pb2.CancelTranscribeJobRequest(job_id=queued.job_id), metadata=self.metadata, timeout=2,
                )
                response = stub.GetTranscribeJob(
                    transcription_pb2.GetTranscribeJobRequest(job_id=queued.job_id), metadata=self.metadata,
                )
                self.assertEqual(response.state, transcription_pb2.TRANSCRIPTION_JOB_STATE_CANCELED)
                self.assertFalse(response.HasField("input_audio_usage"))
                # Repeated cancellation must not release queue capacity twice.
                stub.CancelTranscribeJob(
                    transcription_pb2.CancelTranscribeJobRequest(job_id=queued.job_id), metadata=self.metadata, timeout=2,
                )
                replacement = stub.SubmitTranscribeJob(request, metadata=self.metadata, timeout=2)
                with self.assertRaises(grpc.RpcError) as raised:
                    stub.SubmitTranscribeJob(request, metadata=self.metadata, timeout=2)
                self.assertEqual(raised.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
                stub.CancelTranscribeJob(
                    transcription_pb2.CancelTranscribeJobRequest(job_id=replacement.job_id), metadata=self.metadata, timeout=2,
                )
            finally:
                model.release.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                response = stub.GetTranscribeJob(
                    transcription_pb2.GetTranscribeJobRequest(job_id=running.job_id), metadata=self.metadata,
                )
                if response.state == transcription_pb2.TRANSCRIPTION_JOB_STATE_FAILED:
                    break
                time.sleep(0.01)
            self.assertEqual(response.state, transcription_pb2.TRANSCRIPTION_JOB_STATE_FAILED)
            self.assertEqual(response.error_code, "test.model.failed")
            self.assertFalse(response.HasField("input_audio_usage"))

    def test_corrupt_saved_measurements_fail_reads_until_evidence_is_restored(self):
        model = MeasuredModel()
        with tempfile.TemporaryDirectory() as directory, transcription_server(Path(directory), model) as channel:
            client = DictationClient(channel, metadata=self.metadata)
            job = client.submit_dictate_bytes_job(audio_payload(), filename="stereo.wav", language_code="en")
            client.wait_for_dictation_job(job.job_id, timeout_seconds=5, poll_interval_seconds=0.01)
            path = Path(directory) / "jobs" / f"{job.job_id}.json"
            saved = path.read_text()
            cases = {
                "missing": None,
                "null": None,
                "negative_count": dict(sample_count=-1, sample_rate_hz=16000),
                "fractional_count": dict(sample_count=1.5, sample_rate_hz=16000),
                "boolean_count": dict(sample_count=True, sample_rate_hz=16000),
                "excess_count": dict(sample_count=2**64, sample_rate_hz=16000),
                "zero_rate": dict(sample_count=32001, sample_rate_hz=0),
                "fractional_rate": dict(sample_count=32001, sample_rate_hz=1.5),
                "boolean_rate": dict(sample_count=32001, sample_rate_hz=True),
                "excess_rate": dict(sample_count=32001, sample_rate_hz=2**32),
            }
            for name, usage in cases.items():
                with self.subTest(name=name):
                    payload = json.loads(saved)
                    if name == "missing":
                        del payload["input_audio_usage"]
                    else:
                        payload["input_audio_usage"] = usage
                    path.write_text(json.dumps(payload))
                    try:
                        with self.assertRaises(grpc.RpcError):
                            client.get_dictation_job(job.job_id)
                    finally:
                        path.write_text(saved)
                    self.assert_usage(client.get_dictation_job(job.job_id).result)
            self.assertEqual(len(model.inputs), 1)

    def test_python_client_rejects_absent_or_invalid_measurements_on_success(self):
        class MeasurementServer(transcription_pb2_grpc.TranscriptionServiceServicer):
            def GetTranscribeJob(self, request, context):
                response = transcription_pb2.GetTranscribeJobResponse(
                    job_id=request.job_id, state=transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED,
                )
                if request.job_id == "invalid-rate":
                    response.input_audio_usage.sample_count = 32001
                return response

        with ThreadPoolExecutor(max_workers=1) as executor:
            server = grpc.server(executor)
            transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(MeasurementServer(), server)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    client = DictationClient(channel)
                    with self.assertRaisesRegex(ValueError, "requires input_audio_usage"):
                        client.get_dictation_job("absent")
                    with self.assertRaisesRegex(ValueError, "sample_rate_hz"):
                        client.get_dictation_job("invalid-rate")
            finally:
                server.stop(None).wait()
