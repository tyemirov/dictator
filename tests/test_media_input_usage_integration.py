"""Input quantities across native media processing and durable gRPC jobs."""

from contextlib import contextmanager, ExitStack
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import grpc
import numpy as np

from dictator.alignment.service import AlignmentService
from dictator.audio.ffmpeg_ops import decode_pcm
from dictator.client import AlignmentClient, DiarizationClient, ReferenceSampleClient, SubtitleClient
from dictator.diarization.service import DiarizationService
from dictator.extraction.service import ReferenceExtractionService, pitch_variation
from dictator.runtime import InflightLimiter, MetricsRegistry
from dictator.runtime import jobs
from dictator.speech.v1 import (
    alignment_pb2, alignment_pb2_grpc, artifacts_pb2, artifacts_pb2_grpc,
    subtitle_pb2, subtitle_pb2_grpc, voice_pb2, voice_pb2_grpc, transcription_pb2, transcription_pb2_grpc,
)
from dictator.storage import LocalArtifactStore
from dictator.subtitles.service import SubtitleService
from dictator.transcription.service import TranscriptionService
from dictator.transport.grpc.config import ServerConfig
from dictator.transport.grpc.context import ServiceContext
from dictator.transport.grpc.server import build_server

from test_input_audio_usage_integration import MeasuredModel, FailingModel


def processing_audio() -> bytes:
    mono = (10000 * np.sin(2 * np.pi * 1000 * np.arange(64002) / 32000)).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(32000)
        wav.writeframes(np.repeat(mono[:, None], 2, axis=1).tobytes())
    return output.getvalue()


class ProcessingModels:
    def __init__(self):
        self.whisper = MeasuredModel()
        self.alignment_inputs = []
        self.speaker_inputs = []

    def align(self, segments, model, metadata, audio, device, **kwargs):
        self.alignment_inputs.append(audio)
        return {"segments": [{"words": [{"word": "hello", "start": 0.0, "end": 0.25}]}]}

    def speakers(self, request):
        self.speaker_inputs.append(request["waveform"])
        return SimpleNamespace(itertracks=lambda **kwargs: [(SimpleNamespace(start=0.0, end=0.4), None, "speaker")])


class FailingProcessingModels(ProcessingModels):
    def __init__(self):
        super().__init__()
        self.whisper = FailingModel()

    def align(self, segments, model, metadata, audio, device, **kwargs):
        return self.whisper.transcribe(audio)


@contextmanager
def media_server(root, models):
    with ExitStack() as stack:
        stack.enter_context(patch("dictator.alignment.whisperx_backend.load_whisperx_alignment_modules", return_value=(
            SimpleNamespace(align=models.align),
            SimpleNamespace(load_audio=lambda path: decode_pcm(Path(path)).astype(np.float32) / 32768, SAMPLE_RATE=16000),
        )))
        stack.enter_context(patch("dictator.alignment.whisperx_backend.resolve_device", return_value="cpu"))
        stack.enter_context(patch("dictator.alignment.whisperx_backend.load_cached_alignment_model", return_value=(object(), {})))
        stack.enter_context(patch("torch.from_numpy", create=True, side_effect=lambda samples: SimpleNamespace(unsqueeze=lambda axis: np.expand_dims(samples, axis))))
        transcription = TranscriptionService(model_loader=lambda _: models.whisper)
        alignment = AlignmentService()
        diarization = DiarizationService(transcription_service=transcription)
        subtitles = SubtitleService(transcription_service=transcription, alignment_service=alignment)
        extraction = ReferenceExtractionService()
        runtime = SimpleNamespace(
            get_whisper_model=lambda _: models.whisper, get_diarization_pipeline=lambda: models.speakers,
            get_alignment_service=lambda: alignment, get_diarization_service=lambda: diarization,
            get_subtitle_service=lambda: subtitles, get_reference_extraction_service=lambda: extraction,
        )
        artifact_store = LocalArtifactStore(root / "artifacts")
        managers = {}
        for name, manager_type, store_type in (
            ("alignment_job_manager", jobs.AlignmentJobManager, jobs.LocalAlignmentJobStore),
            ("diarization_job_manager", jobs.DiarizationJobManager, jobs.LocalDiarizationJobStore),
            ("subtitle_job_manager", jobs.SubtitleJobManager, jobs.LocalSubtitleJobStore),
            ("reference_extraction_job_manager", jobs.ExtractReferenceSampleJobManager, jobs.LocalExtractReferenceSampleJobStore),
        ):
            manager = manager_type(job_store=store_type(root / name), artifact_store=artifact_store,
                                   execution_runtime=runtime, max_workers=1, max_pending_jobs=4)
            stack.callback(manager._executor.shutdown, wait=True)
            managers[name] = manager
        context = ServiceContext(artifact_store=artifact_store, execution_runtime=runtime,
                                 metrics=MetricsRegistry(), limiter=InflightLimiter(4),
                                 auth_token="media-usage-test", download_chunk_bytes=4096, **managers)
        server = build_server(ServerConfig(artifact_root=root / "artifacts"), service_context=context)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        stack.callback(lambda: server.stop(None).wait())
        channel = stack.enter_context(grpc.insecure_channel(f"127.0.0.1:{port}"))
        grpc.channel_ready_future(channel).result(timeout=5)
        yield channel


class MediaInputUsageIntegrationTests(unittest.TestCase):
    metadata = (("x-dictator-token", "media-usage-test"),)

    @classmethod
    def setUpClass(cls):
        # Prepare the local signal-processing dependency before RPC deadlines start.
        pitch_variation(np.zeros(16000, dtype=np.int16))

    def assert_usage(self, result):
        usage = getattr(result, "input_audio_usage", None)
        self.assertIsNotNone(usage, "native processed-input measurement is absent")
        self.assertEqual((usage.sample_count, usage.sample_rate_hz), (32001, 16000))

    def assert_model_inputs(self, models):
        for audio in models.whisper.inputs + models.alignment_inputs:
            self.assertEqual(audio.shape, (32001,))
        for audio in models.speaker_inputs:
            self.assertEqual(audio.shape, (1, 32001))

    def upload(self, channel):
        return artifacts_pb2_grpc.ArtifactServiceStub(channel).UploadArtifact(iter([
            artifacts_pb2.UploadArtifactChunk(metadata=artifacts_pb2.UploadArtifactMetadata(filename="tone.wav", media_type="audio/wav")),
            artifacts_pb2.UploadArtifactChunk(content=processing_audio()),
        ]), metadata=self.metadata).artifact.artifact_id

    def test_direct_routes_report_full_input_independent_of_output_duration(self):
        for route in ("alignment", "subtitle-transcription", "subtitle-alignment", "subtitle-detection", "extraction"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as directory:
                models = ProcessingModels()
                with media_server(Path(directory), models) as channel:
                    artifact_id = self.upload(channel)
                    if route == "alignment":
                        result = alignment_pb2_grpc.AlignmentServiceStub(channel).AlignTranscript(
                            alignment_pb2.AlignTranscriptRequest(audio_artifact_id=artifact_id, transcript_text="hello", language_code="en"), metadata=self.metadata)
                    elif route == "extraction":
                        result = voice_pb2_grpc.VoiceServiceStub(channel).ExtractReferenceSample(
                            voice_pb2.ExtractReferenceSampleRequest(source_artifact_id=artifact_id, language_code="en", duration_seconds=1), metadata=self.metadata)
                        self.assertLess(result.trim_end_seconds - result.trim_start_seconds, 1)
                    else:
                        request = subtitle_pb2.RenderSubtitlesRequest(audio_artifact_id=artifact_id,
                            language_code="" if route == "subtitle-detection" else "en", autodetect_language=route == "subtitle-detection",
                            output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT, granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS, group_size=1)
                        if route != "subtitle-transcription":
                            request.source_text = "hello"
                        result = subtitle_pb2_grpc.SubtitleServiceStub(channel).RenderSubtitles(request, metadata=self.metadata)
                    self.assert_usage(result)
                    self.assert_model_inputs(models)

    def test_clients_and_jobs_retain_exact_measurements_after_restart(self):
        for route in ("alignment", "diarization", "subtitle-transcription", "subtitle-alignment", "subtitle-detection", "extraction"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                models = ProcessingModels()
                with media_server(root, models) as channel:
                    client, submit, wait, get = self.client_methods(channel, route)
                    submitted = submit(processing_audio(), filename="tone.wav", **self.arguments(route))
                    result = wait(submitted.job_id, timeout_seconds=5, poll_interval_seconds=0.01).result
                    self.assert_usage(result)
                    self.assert_model_inputs(models)
                    count = len(models.whisper.inputs) + len(models.alignment_inputs) + len(models.speaker_inputs)
                    self.check_retained_measurements(root, route, submitted.job_id, get)
                with media_server(root, models) as channel:
                    client, submit, wait, get = self.client_methods(channel, route)
                    self.assert_usage(get(submitted.job_id).result)
                    self.assertEqual(len(models.whisper.inputs) + len(models.alignment_inputs) + len(models.speaker_inputs), count)

    def check_retained_measurements(self, root, route, job_id, get):
        directory = {"alignment": "alignment_job_manager", "diarization": "diarization_job_manager",
                     "extraction": "reference_extraction_job_manager"}.get(route, "subtitle_job_manager")
        path = root / directory / f"{job_id}.json"
        saved = path.read_text()
        for corruption in ("missing", "null", "zero-rate"):
            with self.subTest(corruption=corruption):
                payload = json.loads(saved)
                if corruption == "missing":
                    del payload["input_audio_usage"]
                else:
                    payload["input_audio_usage"] = None if corruption == "null" else dict(sample_count=32001, sample_rate_hz=0)
                path.write_text(json.dumps(payload))
                try:
                    with self.assertRaises(grpc.RpcError):
                        get(job_id)
                finally:
                    path.write_text(saved)
                self.assert_usage(get(job_id).result)

    def test_unsuccessful_jobs_do_not_report_zero_or_completed_usage(self):
        for route in ("alignment", "diarization", "subtitle-transcription", "subtitle-alignment", "extraction"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as directory:
                models = FailingProcessingModels()
                with media_server(Path(directory), models) as channel:
                    client, submit, wait, get = self.client_methods(channel, route)
                    submitted = submit(processing_audio(), filename="tone.wav", **self.arguments(route))
                    probe = self.native_job_probe(channel, route)
                    try:
                        self.assertTrue(models.whisper.started.wait(5))
                        running = get(submitted.job_id)
                        self.assertTrue(running.state.endswith("_RUNNING"))
                        self.assertFalse(probe(submitted.job_id).HasField("input_audio_usage"))
                        queued = submit(processing_audio(), filename="tone.wav", **self.arguments(route))
                        self.assertTrue(get(queued.job_id).state.endswith("_QUEUED"))
                        self.assertFalse(probe(queued.job_id).HasField("input_audio_usage"))
                        self.cancel_job(client, route, queued.job_id)
                        self.assertTrue(get(queued.job_id).state.endswith("_CANCELED"))
                        self.assertFalse(probe(queued.job_id).HasField("input_audio_usage"))
                    finally:
                        models.whisper.release.set()
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        failed = get(submitted.job_id)
                        if failed.state.endswith("_FAILED"):
                            break
                        time.sleep(0.01)
                    self.assertTrue(failed.state.endswith("_FAILED"))
                    self.assertIsNone(failed.result)
                    self.assertFalse(probe(submitted.job_id).HasField("input_audio_usage"))

    def native_job_probe(self, channel, route):
        if route == "alignment":
            stub = alignment_pb2_grpc.AlignmentServiceStub(channel)
            return lambda job_id: stub.GetAlignTranscriptJob(alignment_pb2.GetAlignTranscriptJobRequest(job_id=job_id), metadata=self.metadata)
        if route == "diarization":
            stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)
            return lambda job_id: stub.GetDiarizeAudioJob(transcription_pb2.GetDiarizeAudioJobRequest(job_id=job_id), metadata=self.metadata)
        if route == "extraction":
            stub = voice_pb2_grpc.VoiceServiceStub(channel)
            return lambda job_id: stub.GetExtractReferenceSampleJob(voice_pb2.GetExtractReferenceSampleJobRequest(job_id=job_id), metadata=self.metadata)
        stub = subtitle_pb2_grpc.SubtitleServiceStub(channel)
        return lambda job_id: stub.GetRenderSubtitlesJob(subtitle_pb2.GetRenderSubtitlesJobRequest(job_id=job_id), metadata=self.metadata)

    def cancel_job(self, client, route, job_id):
        if route == "alignment":
            return client.cancel_alignment_job(job_id)
        if route == "diarization":
            return client.cancel_diarization_job(job_id)
        if route == "extraction":
            return client.cancel_reference_sample_job(job_id)
        return client.cancel_subtitle_job(job_id)

    def arguments(self, route):
        if route == "alignment":
            return dict(language_code="en", transcript_text="hello")
        if route == "extraction":
            return dict(language_code="en", duration_seconds=1)
        if route == "subtitle-detection":
            return dict(autodetect_language=True, source_text="hello")
        if route == "subtitle-alignment":
            return dict(language_code="en", source_text="hello")
        return dict(language_code="en")

    def client_methods(self, channel, route):
        if route == "alignment":
            client = AlignmentClient(channel, metadata=self.metadata)
            return client, client.submit_align_bytes_job, client.wait_for_alignment_job, client.get_alignment_job
        if route == "diarization":
            client = DiarizationClient(channel, metadata=self.metadata)
            return client, client.submit_diarize_bytes_job, client.wait_for_diarization_job, client.get_diarization_job
        if route == "extraction":
            client = ReferenceSampleClient(channel, metadata=self.metadata)
            return client, client.submit_extract_bytes_job, client.wait_for_reference_sample_job, client.get_reference_sample_job
        client = SubtitleClient(channel, metadata=self.metadata)
        return client, client.submit_render_bytes_job, client.wait_for_subtitle_job, client.get_subtitle_job
