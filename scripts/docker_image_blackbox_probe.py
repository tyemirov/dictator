#!/usr/bin/env python
"""Blackbox smoke probe that runs inside the built Docker image."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import wave

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dictator.runtime import InflightLimiter, MetricsRegistry
from dictator.speech.v1 import artifacts_pb2, artifacts_pb2_grpc, voice_pb2, voice_pb2_grpc
from dictator.storage import LocalArtifactStore
from dictator.synthesis.models import SpeechSegment, SynthesisResult
from dictator.transport.grpc.config import ServerConfig
from dictator.transport.grpc.server import build_server
from dictator.transport.grpc.services import ServiceContext


AUTH_TOKEN = "docker-image-blackbox-secret"
SAMPLE_RATE = 24_000


def assert_dependency_imports() -> None:
    import librosa  # noqa: F401
    import pyannote.audio  # noqa: F401
    import soundfile  # noqa: F401
    import whisper  # noqa: F401
    import whisperx  # noqa: F401
    from TTS.api import TTS  # noqa: F401


def assert_default_entrypoint_starts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_root = Path(tmpdir) / "artifacts"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        config_path = Path(tmpdir) / "config.yml"
        config_path.write_text(
            "\n".join(
                (
                    "grpc:",
                    "  host: 127.0.0.1",
                    f"  port: {port}",
                    f"  artifact_root: {artifact_root}",
                    f"  auth_token: {AUTH_TOKEN}",
                )
            )
            + "\n",
            encoding="utf-8",
        )

        process = subprocess.Popen(
            ["python", "serve.py", "--config", str(config_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.time() + 15.0
            last_error: Exception | None = None
            while time.time() < deadline:
                with contextlib.closing(grpc.insecure_channel(f"127.0.0.1:{port}")) as channel:
                    try:
                        grpc.channel_ready_future(channel).result(timeout=1)
                        response = health_pb2_grpc.HealthStub(channel).Check(
                            health_pb2.HealthCheckRequest(service="")
                        )
                        assert response.status == health_pb2.HealthCheckResponse.SERVING
                        return
                    except Exception as exc:  # pragma: no cover - exercised in container only
                        last_error = exc
                        time.sleep(0.25)
            raise AssertionError(f"default container entrypoint did not become healthy: {last_error!r}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()
                process.wait(timeout=10)


def assert_diarization_loader_call_shape() -> None:
    from dictator.extraction import service as extraction_service

    fake_pipeline = SimpleNamespace(to=MagicMock())
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda name: f"device:{name}",
    )
    with (
        patch.object(extraction_service, "configure_torch_runtime"),
        patch.object(extraction_service, "torch", fake_torch),
        patch("pyannote.audio.Pipeline.from_pretrained", return_value=fake_pipeline) as from_pretrained,
    ):
        loaded = extraction_service.load_diarization_pipeline()
    assert loaded is fake_pipeline
    from_pretrained.assert_called_once_with(
        extraction_service.DIARIZATION_MODEL_ID,
        revision=extraction_service.DIARIZATION_MODEL_REVISION,
    )
    fake_pipeline.to.assert_called_once_with("device:cpu")


def assert_whisper_loader_call_shape() -> None:
    from dictator.transcription import service as transcription_service

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with (
        patch.object(transcription_service, "torch", fake_torch),
        patch("whisper.load_model", return_value=object()) as load_model,
    ):
        transcription_service.load_whisper_model("base", cache_dir=Path("/tmp/whisper-cache"))
    load_model.assert_called_once_with(
        "base",
        device="cpu",
        download_root="/tmp/whisper-cache",
    )


def assert_xtts_loader_call_shape() -> None:
    from dictator.synthesis import service as synthesis_service

    class FakeTTS:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.device = None

        def to(self, device: str):
            self.device = device
            return self

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch("TTS.api.TTS", FakeTTS),
    ):
        backend = synthesis_service.XTTSBackend(model_id="xtts-model")
        loaded = backend.load()
    assert isinstance(loaded, FakeTTS)
    assert loaded.model_id == "xtts-model"
    assert loaded.device == "cpu"


def build_wav_bytes(duration_seconds: float = 0.25, frequency_hz: float = 440.0) -> bytes:
    frame_count = int(SAMPLE_RATE * duration_seconds)
    pcm = bytearray()
    for index in range(frame_count):
        value = int(0.2 * 32767 * math.sin(2.0 * math.pi * frequency_hz * (index / SAMPLE_RATE)))
        pcm.extend(struct.pack("<h", value))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(bytes(pcm))
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


class FakeReferenceExtractionService:
    def extract(self, request, model=None, diarization_pipeline=None):
        assert request.output_path is not None
        request.output_path.write_bytes(request.input_path.read_bytes())
        from dictator.extraction.models import ReferenceExtractionResult

        return ReferenceExtractionResult(
            raw_words=({"content": "hello", "start": 0.0, "end": 0.2},),
            dominant_speaker_words=({"content": "hello", "start": 0.0, "end": 0.2},),
            window_start_seconds=0.0,
            window_end_seconds=0.25,
            trim_start_seconds=0.0,
            trim_end_seconds=0.25,
            output_path=request.output_path,
        )


class FakeSynthesisService:
    def synthesise(self, speaker_wav, chunks, cap_seconds, language_code):
        temp_dir = Path(tempfile.mkdtemp(prefix="dictator_blackbox_tts_"))
        wav_paths = []
        segments = []
        start_seconds = 0.0
        for index, chunk in enumerate(chunks):
            wav_path = temp_dir / f"{index:04d}.wav"
            wav_path.write_bytes(build_wav_bytes(duration_seconds=0.15))
            end_seconds = start_seconds + 0.15
            wav_paths.append(wav_path)
            segments.append(SpeechSegment(text=chunk, start_seconds=start_seconds, end_seconds=end_seconds))
            start_seconds = end_seconds
        return SynthesisResult(
            temp_dir=temp_dir,
            wav_paths=tuple(wav_paths),
            segments=tuple(segments),
        )


class FakeRuntime:
    def get_reference_extraction_service(self):
        return FakeReferenceExtractionService()

    def get_synthesis_service(self):
        return FakeSynthesisService()

    def get_whisper_model(self, model_size: str):
        return object()

    def get_diarization_pipeline(self):
        return object()


def upload_artifact(stub, payload: bytes, *, metadata):
    def request_iter():
        yield artifacts_pb2.UploadArtifactChunk(
            metadata=artifacts_pb2.UploadArtifactMetadata(
                filename="sample.wav",
                media_type="audio/wav",
            )
        )
        for index in range(0, len(payload), 1024):
            yield artifacts_pb2.UploadArtifactChunk(content=payload[index : index + 1024])

    return stub.UploadArtifact(request_iter(), metadata=metadata).artifact


def download_artifact(stub, artifact_id: str, *, metadata) -> bytes:
    chunks = stub.DownloadArtifact(
        artifacts_pb2.DownloadArtifactRequest(
            artifact_id=artifact_id,
            chunk_size=1024,
        ),
        metadata=metadata,
    )
    return b"".join(chunk.content for chunk in chunks)


def assert_grpc_voice_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_root = Path(tmpdir) / "artifacts"
        service_context = ServiceContext(
            artifact_store=LocalArtifactStore(artifact_root),
            execution_runtime=FakeRuntime(),
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(4),
            auth_token=AUTH_TOKEN,
            download_chunk_bytes=1024,
        )
        server = build_server(
            ServerConfig(artifact_root=artifact_root, auth_token=AUTH_TOKEN),
            service_context=service_context,
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            metadata = (("authorization", f"Bearer {AUTH_TOKEN}"),)
            with contextlib.closing(grpc.insecure_channel(f"127.0.0.1:{port}")) as channel:
                grpc.channel_ready_future(channel).result(timeout=5)
                health = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(service=""))
                assert health.status == health_pb2.HealthCheckResponse.SERVING

                artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
                voice_stub = voice_pb2_grpc.VoiceServiceStub(channel)

                source_artifact = upload_artifact(artifact_stub, build_wav_bytes(), metadata=metadata)
                reference = voice_stub.ExtractReferenceSample(
                    voice_pb2.ExtractReferenceSampleRequest(
                        source_artifact_id=source_artifact.artifact_id,
                        language_code="en",
                        duration_seconds=0.25,
                    ),
                    metadata=metadata,
                )
                assert reference.sample_artifact.artifact_id

                synthesis = voice_stub.SynthesizeSpeech(
                    voice_pb2.SynthesizeSpeechRequest(
                        speaker_artifact_id=reference.sample_artifact.artifact_id,
                        text="Blackbox Docker probe.",
                        language_code="en",
                    ),
                    metadata=metadata,
                )
                assert synthesis.audio_artifact.artifact_id
                payload = download_artifact(
                    artifact_stub,
                    synthesis.audio_artifact.artifact_id,
                    metadata=metadata,
                )
                assert payload.startswith(b"RIFF"), "synthesized payload is not a WAV file"
                assert len(payload) > 44, "synthesized WAV payload is empty"
        finally:
            server.stop(None)


def main() -> int:
    assert_dependency_imports()
    assert_default_entrypoint_starts()
    assert_diarization_loader_call_shape()
    assert_whisper_loader_call_shape()
    assert_xtts_loader_call_shape()
    assert_grpc_voice_roundtrip()
    print("Docker image blackbox probe passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
