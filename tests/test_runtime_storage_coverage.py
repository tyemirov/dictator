import threading
import tempfile
import time
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from dictator.runtime.errors import ServiceRequestError, ValidationError
from dictator.runtime.inflight import InflightLimiter
from dictator.runtime.jobs import LocalSynthesisJobStore, SynthesisJobManager, SynthesisJobState
from dictator.runtime.metrics import MetricsRegistry
from dictator.runtime.service_runtime import SpeechExecutionRuntime
from dictator.runtime.timeouts import run_with_timeout
from dictator.synthesis.config import SynthesisConfig
from dictator.synthesis.models import SynthesisEngine, SynthesisRequest
from dictator.synthesis.workflow import PreparedSynthesisRequest, prepare_synthesis_request
from dictator.synthesis import text as synthesis_text
from dictator.storage.artifact_store import ArtifactReservation, LocalArtifactStore
from duration import parse_duration


class _FakeThread:
    def __init__(self, *args, **kwargs):
        self.alive = False

    def start(self):
        return None

    def join(self, timeout):
        return None

    def is_alive(self):
        return self.alive


class _ImmediateExecutor:
    def __init__(self, *args, **kwargs):
        return None

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


class _ExplodingExecutor(_ImmediateExecutor):
    def submit(self, fn, *args, **kwargs):
        raise RuntimeError("submit failed")


class RuntimeStorageCoverageTests(unittest.TestCase):
    def test_duration_and_synthesis_text_helpers(self):
        self.assertEqual(parse_duration("90"), 90.0)
        self.assertEqual(parse_duration("1m"), 60.0)
        with self.assertRaisesRegex(Exception, "invalid duration"):
            parse_duration("abc")

        self.assertEqual(synthesis_text.clean("A\x00  B\nC"), "A BC")
        self.assertEqual(synthesis_text.split_into_sentences("Hi. There?"), ["Hi.", "There?"])
        self.assertEqual(synthesis_text.parse_length(None), None)
        self.assertEqual(synthesis_text.parse_length("2m"), 120.0)
        with self.assertRaisesRegex(ValueError, "--length"):
            synthesis_text.parse_length("tomorrow")

    def test_inflight_limiter_tracks_limit_and_raises_when_exhausted(self):
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            InflightLimiter(0)
        limiter = InflightLimiter(1)
        self.assertEqual(limiter.limit, 1)
        with limiter.acquire():
            self.assertEqual(limiter.inflight, 1)
            with self.assertRaises(ServiceRequestError) as exc:
                with limiter.acquire():
                    pass
        self.assertEqual(exc.exception.code, "dictator.runtime.inflight_limit")
        self.assertEqual(limiter.inflight, 0)

    def test_run_with_timeout_handles_immediate_call_timeout_and_worker_failures(self):
        self.assertEqual(run_with_timeout(0, "task", lambda x: x + 1, 2), 3)
        with self.assertRaisesRegex(TimeoutError, "slow exceeded"):
            run_with_timeout(0.01, "slow", lambda: __import__("time").sleep(0.05))
        with self.assertRaisesRegex(RuntimeError, "without producing a result"):
            with patch("dictator.runtime.timeouts.threading.Thread", _FakeThread):
                run_with_timeout(1.0, "empty", lambda: None)
        with self.assertRaisesRegex(ValueError, "boom"):
            run_with_timeout(1.0, "error", lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_metrics_registry_snapshot(self):
        registry = MetricsRegistry(clock=lambda: 15.0, started_at=10.0)
        registry.record_start()
        registry.record_bytes(-5)
        registry.record_bytes(7)
        registry.record_finish(success=True, latency_seconds=0.5)
        registry.record_start()
        registry.record_finish(success=False, latency_seconds=1.5)

        snapshot = registry.snapshot()
        self.assertEqual(snapshot.requests_total, 2)
        self.assertEqual(snapshot.requests_succeeded, 1)
        self.assertEqual(snapshot.requests_failed, 1)
        self.assertEqual(snapshot.bytes_received, 7)
        self.assertEqual(snapshot.inflight, 0)
        self.assertAlmostEqual(snapshot.uptime_seconds, 5.0)
        self.assertAlmostEqual(snapshot.average_latency_seconds, 1.0)
        self.assertAlmostEqual(snapshot.max_latency_seconds, 1.5)

    def test_local_artifact_store_sanitizes_finalizes_and_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            record = store.write_artifact([b"hello"], filename="../../bad name", fallback_suffix=".txt")
            self.assertEqual(record.filename, "bad_name.txt")
            self.assertEqual(store.read_text(record.artifact_id), "hello")
            opened_record, handle = store.open_artifact(record.artifact_id)
            with handle:
                self.assertEqual(handle.read(), b"hello")
            self.assertEqual(opened_record.artifact_id, record.artifact_id)

            empty = store.write_artifact([], filename=None, fallback_suffix=".bin")
            chunks = list(store.iter_artifact_chunks(empty.artifact_id, chunk_size=4))
            self.assertEqual(chunks, [(store.get_artifact(empty.artifact_id), 0, b"", True)])

    def test_finalize_artifact_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            reservation = ArtifactReservation(
                artifact_id="abc",
                filename="sample.txt",
                media_type="text/plain",
                path=Path(tmpdir) / "abc" / "sample.txt",
                metadata_path=Path(tmpdir) / "abc" / "metadata.json",
            )
            with self.assertRaises(FileNotFoundError):
                store.finalize_artifact(reservation)

            reservation.path.parent.mkdir(parents=True, exist_ok=True)
            reservation.path.write_bytes(b"partial")
            store.discard_reservation(reservation)
            self.assertFalse(reservation.path.parent.exists())

    def test_speech_execution_runtime_caches_and_builds_services(self):
        runtime = SpeechExecutionRuntime()
        loader_calls = []

        class FakeTranscriptionService:
            def __init__(self, model_loader=None):
                self.model_loader = model_loader

        class FakeSynthesisService:
            def __init__(self, backends=None):
                self.backends = backends

        class FakeAlignmentService:
            def __init__(self, backend=None):
                self.backend = backend

        class FakeDiarizationService:
            def __init__(self, transcription_service=None, diarization_pipeline_loader=None):
                self.transcription_service = transcription_service
                self.diarization_pipeline_loader = diarization_pipeline_loader

        class FakeSubtitleService:
            def __init__(self, transcription_service=None, alignment_service=None):
                self.transcription_service = transcription_service
                self.alignment_service = alignment_service

        class FakeReferenceExtractionService:
            pass

        def fake_load_whisper_model(model_size):
            loader_calls.append(model_size)
            return {"model": model_size}

        torch_stub = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                cudnn=types.SimpleNamespace(allow_tf32=False),
            ),
            device=lambda *args, **kwargs: None,
            from_numpy=lambda array: array,
        )
        librosa_stub = types.SimpleNamespace(feature=types.SimpleNamespace(rms=lambda y: __import__("numpy").array([[0.0]])))
        pyannote_audio = types.SimpleNamespace(Pipeline=type("Pipeline", (), {}))

        with patch.dict(
            sys.modules,
            {
                "torch": torch_stub,
                "librosa": librosa_stub,
                "ffmpeg": types.SimpleNamespace(),
                "pyannote": types.ModuleType("pyannote"),
                "pyannote.audio": pyannote_audio,
            },
        ):
            import dictator.transcription.service  # noqa: F401
            import dictator.extraction.service  # noqa: F401
            import dictator.synthesis.service  # noqa: F401
            import dictator.alignment.service  # noqa: F401
            import dictator.alignment.whisperx_backend  # noqa: F401
            import dictator.diarization.service  # noqa: F401
            import dictator.subtitles.service  # noqa: F401

            with (
                patch("dictator.transcription.service.load_whisper_model", side_effect=fake_load_whisper_model),
                patch("dictator.transcription.service.TranscriptionService", FakeTranscriptionService),
                patch("dictator.extraction.service.load_diarization_pipeline", return_value="pipeline"),
                patch("dictator.synthesis.service.Qwen3TTSBackend", return_value="qwen-backend"),
                patch("dictator.synthesis.service.SpeechSynthesisService", FakeSynthesisService),
                patch("dictator.alignment.whisperx_backend.WhisperXAlignmentBackend", return_value="align-backend"),
                patch("dictator.alignment.service.AlignmentService", FakeAlignmentService),
                patch("dictator.diarization.service.DiarizationService", FakeDiarizationService),
                patch("dictator.subtitles.service.SubtitleService", FakeSubtitleService),
                patch("dictator.extraction.service.ReferenceExtractionService", FakeReferenceExtractionService),
            ):
                first_model = runtime.get_whisper_model("base")
                second_model = runtime.get_whisper_model("base")
                self.assertIs(first_model, second_model)
                self.assertEqual(loader_calls, ["base"])

                transcription_service = runtime.get_transcription_service()
                self.assertIs(transcription_service, runtime.get_transcription_service())
                self.assertIs(transcription_service.model_loader.__self__, runtime)
                self.assertEqual(transcription_service.model_loader.__func__.__name__, "get_whisper_model")

                self.assertEqual(runtime.get_diarization_pipeline(), "pipeline")
                self.assertEqual(runtime.get_diarization_pipeline(), "pipeline")

                synthesis_service = runtime.get_synthesis_service()
                self.assertEqual(synthesis_service.backends[SynthesisEngine.QWEN3], "qwen-backend")
                self.assertIs(synthesis_service, runtime.get_synthesis_service())
                sys.modules["dictator.synthesis.service"].Qwen3TTSBackend.assert_called_once_with(
                    model_id=runtime._synthesis_config.qwen3_model_id,
                    dtype=runtime._synthesis_config.qwen3_dtype,
                    text_token_budget=runtime._synthesis_config.qwen3_text_token_budget,
                )

                alignment_service = runtime.get_alignment_service()
                self.assertEqual(alignment_service.backend, "align-backend")
                self.assertEqual(runtime.get_alignment_service().backend, "align-backend")

                diarization_service = runtime.get_diarization_service()
                self.assertIs(diarization_service.transcription_service, transcription_service)
                self.assertIs(diarization_service.diarization_pipeline_loader.__self__, runtime)
                self.assertEqual(diarization_service.diarization_pipeline_loader.__func__.__name__, "get_diarization_pipeline")

                subtitle_service = runtime.get_subtitle_service()
                self.assertIs(subtitle_service.transcription_service, transcription_service)
                self.assertEqual(subtitle_service.alignment_service.backend, "align-backend")

                self.assertIsInstance(runtime.get_reference_extraction_service(), FakeReferenceExtractionService)
                self.assertFalse(runtime.readiness_snapshot().ready)
                runtime.mark_synthesis_ready()
                self.assertTrue(runtime.readiness_snapshot().components[-1].ready)

    def test_synthesis_config_reads_qwen_text_budget(self):
        config = SynthesisConfig.from_env(
            {
                "DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET": "256",
                "DICTATOR_QWEN3_TTS_DTYPE": "float16",
            }
        )
        self.assertEqual(config.qwen3_text_token_budget, 256)
        self.assertEqual(config.qwen3_dtype, "float16")

        with self.assertRaisesRegex(ValueError, "positive integer"):
            SynthesisConfig.from_env({"DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET": "0"})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SynthesisConfig.from_env({"DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET": "abc"})

    def test_runtime_readiness_tracks_warmup_state(self):
        runtime = SpeechExecutionRuntime()
        initial = runtime.readiness_snapshot()
        self.assertFalse(initial.ready)
        self.assertFalse(initial.warmup_started)

        with patch("dictator.runtime.service_runtime.threading.Thread", _FakeThread):
            runtime.start_background_warmup()
            runtime.start_background_warmup()
        warming = runtime.readiness_snapshot()
        self.assertTrue(warming.warmup_started)
        self.assertTrue(warming.warmup_in_progress)

        runtime._set_component_ready("transcription", True)
        runtime._set_component_ready("diarization", True)
        runtime.mark_synthesis_ready()
        runtime._warmup_in_progress = False
        ready = runtime.readiness_snapshot()
        self.assertTrue(ready.ready)

    def test_runtime_warmup_covers_success_and_failure_paths(self):
        runtime = SpeechExecutionRuntime()
        backend = types.SimpleNamespace(load=lambda: None)
        runtime.get_whisper_model = lambda model_size: "whisper"
        runtime.get_diarization_pipeline = lambda: "pipeline"
        runtime.get_synthesis_service = lambda: types.SimpleNamespace(backends={SynthesisEngine.QWEN3: backend})
        runtime._run_warmup()
        snapshot = runtime.readiness_snapshot()
        self.assertTrue(snapshot.components[-1].ready)
        self.assertFalse(snapshot.warmup_in_progress)
        self.assertEqual(snapshot.last_error, "")

        failing = SpeechExecutionRuntime()
        failing.get_whisper_model = lambda model_size: (_ for _ in ()).throw(RuntimeError("warmup failed"))
        failing._run_warmup()
        failed_snapshot = failing.readiness_snapshot()
        self.assertEqual(failed_snapshot.last_error, "warmup failed")
        self.assertFalse(failed_snapshot.warmup_in_progress)

    def test_local_synthesis_job_store_and_manager_cover_success_failure_and_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_store = LocalArtifactStore(root / "artifacts")
            speaker = artifact_store.write_artifact([b"wav"], filename="speaker.wav", media_type="audio/wav")
            prepared = PreparedSynthesisRequest(
                speaker_record=speaker,
                synthesis_request=SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=speaker.path,
                    text="hello world",
                    language_code="en",
                    cap_seconds=None,
                    speaker_artifact_id=speaker.artifact_id,
                    speaker_transcript_text="hello world",
                ),
                include_timeline=True,
            )
            store = LocalSynthesisJobStore(root / "jobs")
            with self.assertRaisesRegex(ValueError, "max_workers must be positive"):
                SynthesisJobManager(
                    job_store=store,
                    artifact_store=artifact_store,
                    execution_runtime=object(),
                    max_workers=0,
                    max_pending_jobs=1,
                )
            with self.assertRaisesRegex(ValueError, "max_pending_jobs must be positive"):
                SynthesisJobManager(
                    job_store=store,
                    artifact_store=artifact_store,
                    execution_runtime=object(),
                    max_workers=1,
                    max_pending_jobs=0,
                )
            queued = store.create(prepared)
            self.assertEqual(store.get(queued.job_id).state, SynthesisJobState.QUEUED)
            store.update(queued.job_id, state=SynthesisJobState.RUNNING.value, started_at_unix_seconds=2.0)
            running = store.get(queued.job_id)
            self.assertEqual(running.state, SynthesisJobState.RUNNING)
            store.fail_incomplete_jobs("restart")
            failed = store.get(queued.job_id)
            self.assertEqual(failed.state, SynthesisJobState.FAILED)
            self.assertEqual(failed.error_code, "dictator.jobs.interrupted")

            outcome = types.SimpleNamespace(
                audio_record=speaker,
                audio_duration_seconds=1.5,
                timeline_artifact_id="timeline-1",
                chunk_count=2,
            )
            with patch("dictator.runtime.jobs.ThreadPoolExecutor", _ImmediateExecutor):
                manager = SynthesisJobManager(
                    job_store=store,
                    artifact_store=artifact_store,
                    execution_runtime=object(),
                    max_workers=1,
                    max_pending_jobs=1,
                )
                with patch("dictator.runtime.jobs.execute_synthesis_request", return_value=outcome):
                    created = manager.submit(prepared)
                completed = manager.get(created.job_id)
                self.assertEqual(completed.state, SynthesisJobState.SUCCEEDED)
                self.assertEqual(completed.audio_artifact_id, speaker.artifact_id)
                self.assertEqual(manager._pending_jobs, 0)

                with patch(
                    "dictator.runtime.jobs.execute_synthesis_request",
                    side_effect=ValidationError("dictator.jobs.validation", "bad request"),
                ):
                    dictated_failure = manager.submit(prepared)
                dictated_record = manager.get(dictated_failure.job_id)
                self.assertEqual(dictated_record.state, SynthesisJobState.FAILED)
                self.assertEqual(dictated_record.error_code, "dictator.jobs.validation")

                with patch(
                    "dictator.runtime.jobs.execute_synthesis_request",
                    side_effect=RuntimeError("boom"),
                ):
                    failed_job = manager.submit(prepared)
                self.assertEqual(manager.get(failed_job.job_id).state, SynthesisJobState.FAILED)

                manager._pending_jobs = 1
                with self.assertRaises(ServiceRequestError):
                    manager.submit(prepared)
                manager._pending_jobs = 0

            with patch("dictator.runtime.jobs.ThreadPoolExecutor", _ExplodingExecutor):
                manager = SynthesisJobManager(
                    job_store=store,
                    artifact_store=artifact_store,
                    execution_runtime=object(),
                    max_workers=1,
                    max_pending_jobs=1,
                )
                with self.assertRaisesRegex(RuntimeError, "submit failed"):
                    manager.submit(prepared)
                self.assertEqual(manager._pending_jobs, 0)

    def test_prepare_synthesis_request_requires_inline_or_artifact_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            speaker = store.write_artifact([b"wav"], filename="speaker.wav", media_type="audio/wav")
            with self.assertRaises(ValidationError):
                prepare_synthesis_request(
                    store,
                    speaker_artifact_id=speaker.artifact_id,
                    text="   ",
                    text_artifact_id="",
                    language_code="en",
                    max_duration_seconds=0.0,
                    include_timeline=False,
                    engine=SynthesisEngine.QWEN3,
                    speaker_transcript_text=None,
                )

    def test_speech_execution_runtime_serializes_cold_model_and_pipeline_loads(self):
        runtime = SpeechExecutionRuntime()
        model_results = []
        pipeline_results = []

        model_started = threading.Event()
        model_release = threading.Event()
        model_calls = []

        def fake_load_whisper_model(model_size):
            model_calls.append(model_size)
            model_started.set()
            model_release.wait(1.0)
            return {"model": model_size}

        fake_transcription_module = types.ModuleType("dictator.transcription.service")
        fake_transcription_module.load_whisper_model = fake_load_whisper_model
        with patch.dict(sys.modules, {"dictator.transcription.service": fake_transcription_module}):
            first = threading.Thread(target=lambda: model_results.append(runtime.get_whisper_model("base")))
            second = threading.Thread(target=lambda: model_results.append(runtime.get_whisper_model("base")))
            first.start()
            self.assertTrue(model_started.wait(0.5))
            second.start()
            time.sleep(0.05)
            self.assertEqual(model_calls, ["base"])
            model_release.set()
            first.join()
            second.join()
        self.assertEqual(len(model_results), 2)
        self.assertIs(model_results[0], model_results[1])

        pipeline_started = threading.Event()
        pipeline_release = threading.Event()
        pipeline_calls = []

        def fake_load_diarization_pipeline():
            pipeline_calls.append("pipeline")
            pipeline_started.set()
            pipeline_release.wait(1.0)
            return {"pipeline": True}

        fake_extraction_module = types.ModuleType("dictator.extraction.service")
        fake_extraction_module.load_diarization_pipeline = fake_load_diarization_pipeline
        with patch.dict(sys.modules, {"dictator.extraction.service": fake_extraction_module}):
            first = threading.Thread(target=lambda: pipeline_results.append(runtime.get_diarization_pipeline()))
            second = threading.Thread(target=lambda: pipeline_results.append(runtime.get_diarization_pipeline()))
            first.start()
            self.assertTrue(pipeline_started.wait(0.5))
            second.start()
            time.sleep(0.05)
            self.assertEqual(pipeline_calls, ["pipeline"])
            pipeline_release.set()
            first.join()
            second.join()
        self.assertEqual(len(pipeline_results), 2)
        self.assertIs(pipeline_results[0], pipeline_results[1])


if __name__ == "__main__":
    unittest.main()
