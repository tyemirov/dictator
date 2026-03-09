#!/usr/bin/env python
"""Blackbox smoke probe that runs inside the built Docker image."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dictator.speech.v1 import artifacts_pb2, artifacts_pb2_grpc, voice_pb2, voice_pb2_grpc


AUTH_TOKEN = "docker-image-blackbox-secret"
DIARIZATION_TOKEN_ENV = "HF_TOKEN"
PROBE_SAMPLE_TEXT = (
    "The quick brown fox jumped over the lazy dog. "
    "Eleven benevolent elephants balanced on bright blue bicycles. "
    "She sells sea shells by the seashore."
)


def assert_baked_model_paths() -> None:
    for env_name in (
        "DICTATOR_XTTS_MODEL_ID",
        "DICTATOR_QWEN3_TTS_MODEL_ID",
        "DICTATOR_COSYVOICE3_MODEL_DIR",
    ):
        path = Path(os.environ.get(env_name, "")).expanduser()
        assert path.is_dir(), f"{env_name} does not point to a baked model directory: {path}"
    wetext_cache_dir = (
        Path(os.environ.get("MODELSCOPE_CACHE", "~/.cache/modelscope")).expanduser()
        / "hub"
        / "pengzhendong"
        / "wetext"
    )
    assert wetext_cache_dir.is_dir(), f"wetext frontend cache is missing: {wetext_cache_dir}"


def assert_dependency_imports() -> None:
    import pkg_resources  # noqa: F401
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401
    import librosa  # noqa: F401
    import pyannote.audio
    import qwen_tts  # noqa: F401
    import soundfile  # noqa: F401
    from wetext import Normalizer  # noqa: F401
    import whisper  # noqa: F401
    import whisperx  # noqa: F401
    from TTS.api import TTS  # noqa: F401

    assert pyannote.audio.__version__.startswith("3.4."), pyannote.audio.__version__


def assert_default_entrypoint_starts() -> None:
    with running_default_entrypoint() as port:
        with contextlib.closing(grpc.insecure_channel(f"127.0.0.1:{port}")) as channel:
            response = health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(service="")
            )
            assert response.status == health_pb2.HealthCheckResponse.SERVING


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
        patch.object(extraction_service, "require_diarization_token", return_value="hf-token"),
        patch("pyannote.audio.Pipeline.from_pretrained", return_value=fake_pipeline) as from_pretrained,
    ):
        loaded = extraction_service.load_diarization_pipeline()
    assert loaded is fake_pipeline
    from_pretrained.assert_called_once_with(
        extraction_service.DIARIZATION_MODEL,
        use_auth_token="hf-token",
    )
    fake_pipeline.to.assert_called_once_with("device:cpu")


def assert_real_diarization_pipeline_loads() -> None:
    from dictator.extraction import service as extraction_service

    token = (os.environ.get(DIARIZATION_TOKEN_ENV, "") or "").strip()
    if not token:
        raise AssertionError(
            f"{DIARIZATION_TOKEN_ENV} must be set so the Docker image probe can load the real diarization pipeline."
        )

    pipeline = extraction_service.load_diarization_pipeline()
    assert pipeline is not None, "real diarization pipeline did not load"


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
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.device = None

        def to(self, device: str):
            self.device = device
            return self

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "xtts"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("TTS.api.TTS", FakeTTS),
        ):
            backend = synthesis_service.XTTSBackend(model_id=str(model_dir))
            loaded = backend.load()
        assert isinstance(loaded, FakeTTS)
        assert loaded.kwargs["model_dir"] == str(model_dir)
        assert loaded.kwargs["progress_bar"] is False
        assert loaded.device == "cpu"

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch("TTS.api.TTS", FakeTTS),
    ):
        backend = synthesis_service.XTTSBackend(model_id="xtts-model")
        loaded = backend.load()
    assert isinstance(loaded, FakeTTS)
    assert loaded.args == ("xtts-model",)
    assert loaded.device == "cpu"

@contextlib.contextmanager
def running_default_entrypoint() -> contextlib.AbstractContextManager[int]:
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
                        yield port
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


def synthesize_probe_sample_wav(temp_dir: Path) -> Path:
    sample_path = temp_dir / "probe-sample.wav"
    subprocess.run(
        [
            "espeak-ng",
            "-v",
            "en-us",
            "-s",
            "130",
            "-w",
            str(sample_path),
            PROBE_SAMPLE_TEXT,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not sample_path.exists():
        raise AssertionError(f"Probe sample WAV was not created: {sample_path}")
    return sample_path


def upload_artifact(stub, payload: bytes, *, filename: str, metadata):
    def request_iter():
        yield artifacts_pb2.UploadArtifactChunk(
            metadata=artifacts_pb2.UploadArtifactMetadata(
                filename=filename,
                media_type="audio/wav",
            )
        )
        for index in range(0, len(payload), 1024 * 1024):
            yield artifacts_pb2.UploadArtifactChunk(content=payload[index : index + 1024 * 1024])

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
    metadata = (("authorization", f"Bearer {AUTH_TOKEN}"),)

    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = synthesize_probe_sample_wav(Path(tmpdir))
        sample_payload = sample_path.read_bytes()
        assert sample_payload.startswith(b"RIFF"), "probe sample is not a WAV file"
        assert len(sample_payload) > 44, "probe sample WAV payload is empty"

        with running_default_entrypoint() as port:
            with contextlib.closing(grpc.insecure_channel(f"127.0.0.1:{port}")) as channel:
                grpc.channel_ready_future(channel).result(timeout=5)
                health = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(service=""))
                assert health.status == health_pb2.HealthCheckResponse.SERVING

                artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
                voice_stub = voice_pb2_grpc.VoiceServiceStub(channel)

                source_artifact = upload_artifact(
                    artifact_stub,
                    sample_payload,
                    filename=sample_path.name,
                    metadata=metadata,
                )
                assert source_artifact.artifact_id

                reference = voice_stub.ExtractReferenceSample(
                    voice_pb2.ExtractReferenceSampleRequest(
                        source_artifact_id=source_artifact.artifact_id,
                        language_code="en",
                        model_size="tiny",
                        duration_seconds=5.0,
                    ),
                    metadata=metadata,
                )
                assert reference.sample_artifact.artifact_id
                reference_payload = download_artifact(
                    artifact_stub,
                    reference.sample_artifact.artifact_id,
                    metadata=metadata,
                )
                assert reference_payload.startswith(b"RIFF"), "reference sample is not a WAV file"
                assert len(reference_payload) > 44, "reference sample WAV payload is empty"

                synthesis = voice_stub.SynthesizeSpeech(
                    voice_pb2.SynthesizeSpeechRequest(
                        speaker_artifact_id=reference.sample_artifact.artifact_id,
                        text="Blackbox Docker probe.",
                        language_code="en",
                        synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_XTTS,
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


def main() -> int:
    assert_baked_model_paths()
    assert_dependency_imports()
    assert_default_entrypoint_starts()
    assert_diarization_loader_call_shape()
    assert_real_diarization_pipeline_loads()
    assert_whisper_loader_call_shape()
    assert_xtts_loader_call_shape()
    assert_grpc_voice_roundtrip()
    print("Docker image blackbox probe passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
