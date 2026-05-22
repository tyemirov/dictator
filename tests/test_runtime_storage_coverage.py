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
from dictator.runtime.jobs import LocalSynthesisJobStore, SynthesisJobManager, SynthesisJobRecord, SynthesisJobState
from dictator.runtime.metrics import MetricsRegistry
from dictator.runtime.service_runtime import SpeechExecutionRuntime
from dictator.runtime.timeouts import run_with_timeout
from dictator.synthesis.config import SynthesisConfig
from dictator.synthesis.models import DEFAULT_SYNTHESIS_AUDIO_FORMAT, SILERO_RU_SYNTHESIS_AUDIO_FORMAT, SynthesisEngine, SynthesisRequest
from dictator.synthesis.workflow import PreparedSynthesisRequest, execute_synthesis_request, prepare_synthesis_request
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


class _ManualFuture:
    def __init__(self):
        self.callbacks = []
        self.cancel_called = False

    def done(self):
        return False

    def cancel(self):
        self.cancel_called = True
        return True

    def add_done_callback(self, callback):
        self.callbacks.append(callback)


class RuntimeStorageCoverageTests(unittest.TestCase):
    def test_duration_and_synthesis_text_helpers(self):
        self.assertEqual(parse_duration("90"), 90.0)
        self.assertEqual(parse_duration("1m"), 60.0)
        with self.assertRaisesRegex(Exception, "invalid duration"):
            parse_duration("abc")

        self.assertEqual(synthesis_text.clean("A\x00  B\nC"), "A B C")
        self.assertEqual(synthesis_text.clean("Though;\nHe"), "Though; He")
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
                patch("dictator.synthesis.service.SileroRuTTSBackend", return_value="silero-backend"),
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
                self.assertEqual(synthesis_service.backends[SynthesisEngine.SILERO_RU], "silero-backend")
                self.assertIs(synthesis_service, runtime.get_synthesis_service())
                sys.modules["dictator.synthesis.service"].Qwen3TTSBackend.assert_called_once_with(
                    model_id=runtime._synthesis_config.qwen3_model_id,
                    dtype=runtime._synthesis_config.qwen3_dtype,
                    text_token_budget=runtime._synthesis_config.qwen3_text_token_budget,
                )
                sys.modules["dictator.synthesis.service"].SileroRuTTSBackend.assert_called_once_with(
                    model_path=runtime._synthesis_config.silero_ru_model_path,
                    model_url=runtime._synthesis_config.silero_ru_model_url,
                    default_speaker=runtime._synthesis_config.silero_ru_default_speaker,
                    sample_rate=runtime._synthesis_config.silero_ru_sample_rate,
                    text_char_budget=runtime._synthesis_config.silero_ru_text_char_budget,
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
    def test_synthesis_config_reads_engine_settings(self):
        config = SynthesisConfig.from_env(
            {
                "DICTATOR_MODEL_ROOT": "/models",
                "DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET": "256",
                "DICTATOR_QWEN3_TTS_DTYPE": "float16",
                "DICTATOR_SILERO_RU_DEFAULT_SPEAKER": "xenia",
                "DICTATOR_SILERO_RU_SAMPLE_RATE": "24000",
                "DICTATOR_SILERO_RU_TEXT_CHAR_BUDGET": "512",
            }
        )
        self.assertEqual(config.qwen3_text_token_budget, 256)
        self.assertEqual(config.qwen3_dtype, "float16")
        self.assertEqual(config.silero_ru_model_path, "/models/silero/v5_5_ru.pt")
        self.assertEqual(config.silero_ru_default_speaker, "xenia")
        self.assertEqual(config.silero_ru_sample_rate, 24000)
        self.assertEqual(config.silero_ru_text_char_budget, 512)

        explicit_path_config = SynthesisConfig.from_env(
            {
                "DICTATOR_MODEL_ROOT": "/models",
                "DICTATOR_SILERO_RU_MODEL_PATH": "/custom/v5_5_ru.pt",
                "DICTATOR_SILERO_RU_MODEL_URL": "https://example.invalid/model.pt",
            }
        )
        self.assertEqual(explicit_path_config.silero_ru_model_path, "/custom/v5_5_ru.pt")
        self.assertEqual(explicit_path_config.silero_ru_model_url, "https://example.invalid/model.pt")

        with self.assertRaisesRegex(ValueError, "positive integer"):
            SynthesisConfig.from_env({"DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET": "0"})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SynthesisConfig.from_env({"DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET": "abc"})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SynthesisConfig.from_env({"DICTATOR_SILERO_RU_SAMPLE_RATE": "0"})

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
                audio_format=DEFAULT_SYNTHESIS_AUDIO_FORMAT,
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
                def execute_with_progress(**kwargs):
                    kwargs["progress_callback"](1, 3)
                    kwargs["progress_callback"](2, 3)
                    return outcome

                with patch("dictator.runtime.jobs.execute_synthesis_request", side_effect=execute_with_progress):
                    created = manager.submit(prepared)
                completed = manager.get(created.job_id)
                self.assertEqual(completed.state, SynthesisJobState.SUCCEEDED)
                self.assertEqual(completed.audio_artifact_id, speaker.artifact_id)
                self.assertEqual(completed.estimated_total_chunks, 2)
                self.assertEqual(completed.completed_chunks, 2)
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

                leaky = store.create(prepared)
                original_update = store.update
                first_update = True

                def flaky_update(job_id, **updates):
                    nonlocal first_update
                    if first_update:
                        first_update = False
                        raise RuntimeError("transient update failure")
                    return original_update(job_id, **updates)

                manager._pending_jobs = 1
                with patch.object(store, "update", side_effect=flaky_update):
                    manager._run_job(leaky.job_id, prepared)
                repaired = manager.get(leaky.job_id)
                self.assertEqual(repaired.state, SynthesisJobState.FAILED)
                self.assertEqual(repaired.error_code, "dictator.jobs.failed")
                self.assertEqual(manager._pending_jobs, 0)

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

    def test_synthesis_job_record_rejects_invalid_audio_format_payload(self):
        with self.assertRaisesRegex(ValueError, "audio_format"):
            SynthesisJobRecord.from_json_dict(
                {
                    "job_id": "job-1",
                    "state": "queued",
                    "engine": "qwen3",
                    "language_code": "en",
                    "include_timeline": False,
                    "speaker_artifact_id": "speaker-1",
                    "created_at_unix_seconds": 1.0,
                    "audio_format": "bad",
                }
            )

    def test_execute_synthesis_request_forwards_progress_callback(self):
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
                include_timeline=False,
            )
            temp_dir = root / "tmp"
            temp_dir.mkdir()
            wav_path = temp_dir / "0000.wav"
            wav_path.write_bytes(b"wav")
            progress_updates = []

            class _FakeSynthesisService:
                def synthesise_text(self, request, *, progress_callback=None):
                    if progress_callback is not None:
                        progress_callback(1, 2)
                    return types.SimpleNamespace(
                        temp_dir=temp_dir,
                        wav_paths=(wav_path,),
                        segments=(
                            types.SimpleNamespace(
                                end_seconds=0.4,
                                to_timeline_dict=lambda: {"content": "hello", "start": 0.0, "end": 0.4},
                            ),
                        ),
                    )

            runtime = types.SimpleNamespace(
                get_synthesis_service=lambda: _FakeSynthesisService(),
                mark_synthesis_ready=lambda: None,
            )

            def fake_concat_normalise(wav_paths, output_path, cap_seconds, *, target_sample_rate=0):
                self.assertEqual(wav_paths, (wav_path,))
                self.assertIsNone(cap_seconds)
                self.assertEqual(target_sample_rate, 24000)
                output_path.write_bytes(b"RIFF")

            with patch("dictator.audio.ffmpeg_ops.concat_normalise", side_effect=fake_concat_normalise):
                outcome = execute_synthesis_request(
                    artifact_store=artifact_store,
                    execution_runtime=runtime,
                    prepared=prepared,
                    progress_callback=lambda completed, total: progress_updates.append((completed, total)),
                )
        self.assertEqual(progress_updates, [(1, 2)])
        self.assertEqual(outcome.audio_record.media_type, "audio/wav")
        self.assertEqual(outcome.audio_record.audio_metadata.container, "wav")
        self.assertEqual(outcome.audio_record.audio_metadata.duration_seconds, 0.4)
        self.assertEqual(outcome.audio_format, DEFAULT_SYNTHESIS_AUDIO_FORMAT)
        self.assertEqual(outcome.audio_duration_seconds, 0.4)

    def test_execute_silero_synthesis_without_reference_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_store = LocalArtifactStore(root / "artifacts")
            prepared = PreparedSynthesisRequest(
                speaker_record=None,
                synthesis_request=SynthesisRequest(
                    engine=SynthesisEngine.SILERO_RU,
                    speaker_wav=None,
                    text="привет",
                    language_code="ru",
                    cap_seconds=None,
                    preset_speaker="xenia",
                    audio_format=SILERO_RU_SYNTHESIS_AUDIO_FORMAT,
                ),
                include_timeline=True,
                audio_format=SILERO_RU_SYNTHESIS_AUDIO_FORMAT,
            )
            temp_dir = root / "tmp"
            temp_dir.mkdir()
            wav_path = temp_dir / "0000.wav"
            wav_path.write_bytes(b"wav")

            class _FakeSynthesisService:
                def synthesise_text(self, request, *, progress_callback=None):
                    return types.SimpleNamespace(
                        temp_dir=temp_dir,
                        wav_paths=(wav_path,),
                        segments=(
                            types.SimpleNamespace(
                                end_seconds=0.25,
                                to_timeline_dict=lambda: {"content": "привет", "start": 0.0, "end": 0.25},
                            ),
                        ),
                    )

            runtime = types.SimpleNamespace(
                get_synthesis_service=lambda: _FakeSynthesisService(),
                mark_synthesis_ready=lambda: None,
            )

            def fake_concat_normalise(wav_paths, output_path, cap_seconds, *, target_sample_rate=0):
                self.assertEqual(target_sample_rate, 24000)
                output_path.write_bytes(b"RIFF")

            with patch("dictator.audio.ffmpeg_ops.concat_normalise", side_effect=fake_concat_normalise):
                outcome = execute_synthesis_request(
                    artifact_store=artifact_store,
                    execution_runtime=runtime,
                    prepared=prepared,
                )
            timeline = artifact_store.read_text(outcome.timeline_artifact_id)
        self.assertEqual(outcome.audio_record.filename, "xenia_synth.wav")
        self.assertEqual(outcome.audio_format, SILERO_RU_SYNTHESIS_AUDIO_FORMAT)
        self.assertIn('"speaker": "xenia"', timeline)
        self.assertIn('"engine": "silero_ru"', timeline)

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
            with self.assertRaisesRegex(ValidationError, "speaker_artifact_id"):
                prepare_synthesis_request(
                    store,
                    speaker_artifact_id="",
                    text="hello",
                    text_artifact_id="",
                    language_code="en",
                    max_duration_seconds=0.0,
                    include_timeline=False,
                    engine=SynthesisEngine.QWEN3,
                    speaker_transcript_text=None,
                )
            silero_prepared = prepare_synthesis_request(
                store,
                speaker_artifact_id="",
                text="привет",
                text_artifact_id="",
                language_code="",
                max_duration_seconds=0.0,
                include_timeline=False,
                engine=SynthesisEngine.SILERO_RU,
                speaker_transcript_text=None,
                preset_speaker="baya",
            )
            silero_record = LocalSynthesisJobStore(Path(tmpdir) / "silero-jobs").create(silero_prepared)
        self.assertIsNone(silero_prepared.speaker_record)
        self.assertEqual(silero_prepared.synthesis_request.language_code, "ru")
        self.assertEqual(silero_prepared.audio_format, SILERO_RU_SYNTHESIS_AUDIO_FORMAT)
        self.assertEqual(silero_record.speaker_artifact_id, "")

    def test_synthesis_job_store_rejects_path_traversal_job_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalSynthesisJobStore(Path(tmpdir))
            with self.assertRaises(ValidationError):
                store.get("   ")
            with self.assertRaises(ValidationError):
                store.get("../escape")
            with self.assertRaises(ValidationError):
                store.get("nested/job")

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


class GenericJobRuntimeCoverageTests(unittest.TestCase):
    @staticmethod
    def _swap_immediate_executor(manager):
        original = manager._executor
        original.shutdown(wait=False, cancel_futures=True)
        manager._executor = _ImmediateExecutor()
        return manager

    @staticmethod
    def _write_reserved_outputs(request, source_path):
        from pathlib import Path

        for value in vars(request).values():
            if isinstance(value, Path) and value != source_path:
                value.parent.mkdir(parents=True, exist_ok=True)
                if value.suffix == ".srt":
                    value.write_text("1\n00:00:00,000 --> 00:00:00,400\nhello\n", encoding="utf-8")
                else:
                    value.write_bytes(b"sample")

    def test_base_queued_job_manager_methods_raise(self):
        from dictator.runtime import jobs as jobs_module

        manager = jobs_module._QueuedJobManager(
            job_store=types.SimpleNamespace(fail_incomplete_jobs=lambda _message: 0),
            max_workers=1,
            max_pending_jobs=1,
            thread_name_prefix="test-jobs",
        )
        self.addCleanup(manager._executor.shutdown, wait=False, cancel_futures=True)

        with self.assertRaises(NotImplementedError):
            manager._create_record(None)
        with self.assertRaises(NotImplementedError):
            manager._run_job("job-1", None)

    def test_job_cancel_persists_terminal_state_and_cancels_queued_future(self):
        from dictator.runtime import jobs as jobs_module

        class _HoldingExecutor:
            def __init__(self):
                self.futures = []

            def submit(self, fn, *args, **kwargs):
                future = _ManualFuture()
                self.futures.append(future)
                return future

        class _TestManager(jobs_module._QueuedJobManager):
            def _create_record(self, prepared):
                return self.job_store.create(prepared)

            def _run_job(self, job_id, prepared):
                self.job_store.update(job_id, state=jobs_module.TranscriptionJobState.SUCCEEDED.value)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_record = types.SimpleNamespace(artifact_id="audio-1", path=root / "audio.wav", filename="audio.wav")
            store = jobs_module.LocalTranscriptionJobStore(root / "jobs")
            manager = _TestManager(
                job_store=store,
                max_workers=1,
                max_pending_jobs=2,
                thread_name_prefix="test-cancel",
            )
            manager._executor.shutdown(wait=False, cancel_futures=True)
            executor = _HoldingExecutor()
            manager._executor = executor
            prepared = jobs_module.PreparedTranscriptionJob(
                audio_record=audio_record,
                language_code="en",
                model_size="base",
                include_word_segments=True,
            )

            callback_record = manager.submit(prepared)
            self.assertIn(callback_record.job_id, manager._futures)
            executor.futures[0].callbacks[0](executor.futures[0])
            self.assertNotIn(callback_record.job_id, manager._futures)

            queued = manager.submit(prepared)
            canceled = manager.cancel(queued.job_id)
            self.assertEqual(canceled.state, jobs_module.TranscriptionJobState.CANCELED)
            self.assertEqual(canceled.error_code, "dictator.jobs.canceled")
            self.assertEqual(manager._pending_jobs, 1)
            self.assertTrue(executor.futures[1].cancel_called)

            self.assertEqual(store.update(queued.job_id, state=jobs_module.TranscriptionJobState.SUCCEEDED.value).state, jobs_module.TranscriptionJobState.CANCELED)
            self.assertEqual(manager.cancel(queued.job_id).state, jobs_module.TranscriptionJobState.CANCELED)

    def test_generic_job_stores_create_and_fail_incomplete_jobs(self):
        from dictator.runtime import jobs as jobs_module

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"audio")
            audio_record = types.SimpleNamespace(artifact_id="audio-1", path=audio_path, filename="audio.wav")
            source_record = types.SimpleNamespace(artifact_id="source-1", path=audio_path, filename="audio.wav")

            cases = (
                (
                    jobs_module.LocalAlignmentJobStore(root / "alignment"),
                    jobs_module.PreparedAlignmentJob(
                        audio_record=audio_record,
                        transcript_text="hello",
                        language_code="en",
                        remove_punctuation=False,
                        include_srt_text=True,
                    ),
                    jobs_module.AlignmentJobState,
                ),
                (
                    jobs_module.LocalTranscriptionJobStore(root / "transcription"),
                    jobs_module.PreparedTranscriptionJob(
                        audio_record=audio_record,
                        language_code="en",
                        model_size="base",
                        include_word_segments=True,
                    ),
                    jobs_module.TranscriptionJobState,
                ),
                (
                    jobs_module.LocalDiarizationJobStore(root / "diarization"),
                    jobs_module.PreparedDiarizationJob(
                        audio_record=audio_record,
                        language_code="en",
                        model_size="base",
                        include_words=True,
                        include_utterances=True,
                        include_speakers=True,
                        include_speaker_segments=True,
                        utterance_gap_seconds=0.8,
                        persist_json_artifact=True,
                    ),
                    jobs_module.DiarizationJobState,
                ),
                (
                    jobs_module.LocalSubtitleJobStore(root / "subtitle"),
                    jobs_module.PreparedSubtitleJob(
                        audio_record=audio_record,
                        language_code="en",
                        model_size="base",
                        granularity="words",
                        group_size=1,
                        source_text="hello",
                        source_text_name="source.txt",
                        include_srt_text=True,
                    ),
                    jobs_module.SubtitleJobState,
                ),
                (
                    jobs_module.LocalExtractReferenceSampleJobStore(root / "extract"),
                    jobs_module.PreparedExtractReferenceSampleJob(
                        source_record=source_record,
                        model_size="base",
                        language_code="en",
                        duration_seconds=10.0,
                        max_speech_rate=4.0,
                        min_centroid_hz=500.0,
                        max_centroid_hz=4000.0,
                    ),
                    jobs_module.ExtractReferenceSampleJobState,
                ),
            )

            for store, prepared, state_enum in cases:
                with self.subTest(store=type(store).__name__):
                    record = store.create(prepared)
                    self.assertEqual(record.state, state_enum.QUEUED)
                    store.fail_incomplete_jobs("service restarted before the job completed")
                    failed = store.get(record.job_id)
                    self.assertEqual(failed.state, state_enum.FAILED)
                    self.assertEqual(failed.error_code, "dictator.jobs.interrupted")

    def test_generic_job_managers_succeed(self):
        from dictator.alignment.models import AlignedWord
        from dictator.runtime import jobs as jobs_module

        class _AlignmentService:
            def __init__(self, source_path):
                self._source_path = source_path

            def align(self, request):
                GenericJobRuntimeCoverageTests._write_reserved_outputs(request, self._source_path)
                return types.SimpleNamespace(
                    language="en",
                    words=(AlignedWord(text="hello", start_seconds=0.0, end_seconds=0.4),),
                    srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello\n",
                )

        class _TranscriptionService:
            def transcribe(self, *_args, **_kwargs):
                return types.SimpleNamespace(
                    text="hello",
                    language="en",
                    words=(types.SimpleNamespace(text="hello", start_seconds=0.0, end_seconds=0.4),),
                )

        class _DiarizationResult:
            text = "hello"
            language = "en"

            def to_json_dict(self, **_kwargs):
                return {"text": "hello", "speakers": [{"speaker": "S1"}]}

        class _DiarizationService:
            def diarize(self, *_args, **_kwargs):
                return _DiarizationResult()

        class _SubtitleService:
            def __init__(self, source_path):
                self._source_path = source_path

            def render(self, request, **_kwargs):
                GenericJobRuntimeCoverageTests._write_reserved_outputs(request, self._source_path)
                return types.SimpleNamespace(
                    language="en",
                    mode="forced_alignment",
                    output_format="srt",
                    granularity="words",
                    group_size=1,
                    cues=(types.SimpleNamespace(index=1, text="hello", content="hello", start_seconds=0.0, end_seconds=0.4, item_count=1),),
                    srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello\n",
                )

        class _ReferenceExtractionRequest:
            def __init__(self, *args, **kwargs):
                self.__dict__.update(kwargs)

        class _ExtractionService:
            def __init__(self, source_path):
                self._source_path = source_path

            def extract(self, request, **_kwargs):
                GenericJobRuntimeCoverageTests._write_reserved_outputs(request, self._source_path)
                return types.SimpleNamespace(
                    trim_start_seconds=0.5,
                    trim_end_seconds=1.5,
                    window_start_seconds=0.0,
                    window_end_seconds=2.0,
                    dominant_speaker_words=("a", "b"),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_store = LocalArtifactStore(root / "artifacts")
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"audio")
            audio_record = types.SimpleNamespace(artifact_id="audio-1", path=audio_path, filename="audio.wav")
            source_record = types.SimpleNamespace(artifact_id="source-1", path=audio_path, filename="audio.wav")
            runtime = types.SimpleNamespace(
                get_alignment_service=lambda: _AlignmentService(audio_path),
                get_transcription_service=lambda: _TranscriptionService(),
                get_diarization_service=lambda: _DiarizationService(),
                get_subtitle_service=lambda: _SubtitleService(audio_path),
                get_reference_extraction_service=lambda: _ExtractionService(audio_path),
                get_whisper_model=lambda _model_size: object(),
                get_diarization_pipeline=lambda: object(),
            )

            with patch.dict(
                sys.modules,
                {
                    "dictator.extraction": types.SimpleNamespace(models=types.SimpleNamespace(ReferenceExtractionRequest=_ReferenceExtractionRequest)),
                    "dictator.extraction.models": types.SimpleNamespace(ReferenceExtractionRequest=_ReferenceExtractionRequest),
                },
            ):
                cases = (
                    (
                        self._swap_immediate_executor(
                            jobs_module.AlignmentJobManager(
                                job_store=jobs_module.LocalAlignmentJobStore(root / "alignment"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedAlignmentJob(
                            audio_record=audio_record,
                            transcript_text="hello",
                            language_code="en",
                            remove_punctuation=False,
                            include_srt_text=True,
                        ),
                        lambda record: (
                            self.assertEqual(record.state, jobs_module.AlignmentJobState.SUCCEEDED),
                            self.assertEqual(record.language_code, "en"),
                            self.assertEqual(record.words[0].text, "hello"),
                            self.assertTrue(record.srt_artifact_id),
                        ),
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.TranscriptionJobManager(
                                job_store=jobs_module.LocalTranscriptionJobStore(root / "transcription"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedTranscriptionJob(
                            audio_record=audio_record,
                            language_code="en",
                            model_size="base",
                            include_word_segments=True,
                        ),
                        lambda record: (
                            self.assertEqual(record.state, jobs_module.TranscriptionJobState.SUCCEEDED),
                            self.assertEqual(record.text, "hello"),
                            self.assertEqual(record.words[0].text, "hello"),
                        ),
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.DiarizationJobManager(
                                job_store=jobs_module.LocalDiarizationJobStore(root / "diarization"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedDiarizationJob(
                            audio_record=audio_record,
                            language_code="en",
                            model_size="base",
                            include_words=True,
                            include_utterances=True,
                            include_speakers=True,
                            include_speaker_segments=True,
                            utterance_gap_seconds=0.8,
                            persist_json_artifact=True,
                        ),
                        lambda record: (
                            self.assertEqual(record.state, jobs_module.DiarizationJobState.SUCCEEDED),
                            self.assertEqual(record.diarization["text"], "hello"),
                            self.assertTrue(record.diarization_artifact_id),
                        ),
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.SubtitleJobManager(
                                job_store=jobs_module.LocalSubtitleJobStore(root / "subtitle"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedSubtitleJob(
                            audio_record=audio_record,
                            language_code="en",
                            model_size="base",
                            granularity="words",
                            group_size=1,
                            source_text="hello",
                            source_text_name="source.txt",
                            include_srt_text=True,
                        ),
                        lambda record: (
                            self.assertEqual(record.state, jobs_module.SubtitleJobState.SUCCEEDED),
                            self.assertEqual(record.mode, "forced_alignment"),
                            self.assertTrue(record.srt_artifact_id),
                        ),
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.ExtractReferenceSampleJobManager(
                                job_store=jobs_module.LocalExtractReferenceSampleJobStore(root / "extract"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedExtractReferenceSampleJob(
                            source_record=source_record,
                            model_size="base",
                            language_code="en",
                            duration_seconds=10.0,
                            max_speech_rate=4.0,
                            min_centroid_hz=500.0,
                            max_centroid_hz=4000.0,
                        ),
                        lambda record: (
                            self.assertEqual(record.state, jobs_module.ExtractReferenceSampleJobState.SUCCEEDED),
                            self.assertTrue(record.sample_artifact_id),
                            self.assertEqual(record.dominant_speaker_word_count, 2),
                        ),
                    ),
                )

                for manager, prepared, assertion in cases:
                    with self.subTest(manager=type(manager).__name__):
                        submitted = manager.submit(prepared)
                        record = manager.get(submitted.job_id)
                        assertion(record)

    def test_generic_job_managers_capture_expected_failures(self):
        from dictator.runtime import jobs as jobs_module

        class _ReferenceExtractionRequest:
            def __init__(self, *args, **kwargs):
                self.__dict__.update(kwargs)

        def _runtime_for(mode):
            def _raise():
                if mode == "dictator":
                    raise jobs_module.DictatorError("dictator.test.failure", "broken")
                raise RuntimeError("broken")

            return types.SimpleNamespace(
                get_alignment_service=lambda: types.SimpleNamespace(align=lambda *_args, **_kwargs: _raise()),
                get_transcription_service=lambda: types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: _raise()),
                get_diarization_service=lambda: types.SimpleNamespace(diarize=lambda *_args, **_kwargs: _raise()),
                get_subtitle_service=lambda: types.SimpleNamespace(render=lambda *_args, **_kwargs: _raise()),
                get_reference_extraction_service=lambda: types.SimpleNamespace(extract=lambda *_args, **_kwargs: _raise()),
                get_whisper_model=lambda _model_size: object(),
                get_diarization_pipeline=lambda: object(),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_store = LocalArtifactStore(root / "artifacts")
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"audio")
            audio_record = types.SimpleNamespace(artifact_id="audio-1", path=audio_path, filename="audio.wav")
            source_record = types.SimpleNamespace(artifact_id="source-1", path=audio_path, filename="audio.wav")

            def _cases(runtime):
                return (
                    (
                        self._swap_immediate_executor(
                            jobs_module.AlignmentJobManager(
                                job_store=jobs_module.LocalAlignmentJobStore(root / f"alignment-{id(runtime)}"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedAlignmentJob(
                            audio_record=audio_record,
                            transcript_text="hello",
                            language_code="en",
                            remove_punctuation=False,
                            include_srt_text=True,
                        ),
                        jobs_module.AlignmentJobState,
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.TranscriptionJobManager(
                                job_store=jobs_module.LocalTranscriptionJobStore(root / f"transcription-{id(runtime)}"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedTranscriptionJob(
                            audio_record=audio_record,
                            language_code="en",
                            model_size="base",
                            include_word_segments=True,
                        ),
                        jobs_module.TranscriptionJobState,
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.DiarizationJobManager(
                                job_store=jobs_module.LocalDiarizationJobStore(root / f"diarization-{id(runtime)}"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedDiarizationJob(
                            audio_record=audio_record,
                            language_code="en",
                            model_size="base",
                            include_words=True,
                            include_utterances=True,
                            include_speakers=True,
                            include_speaker_segments=True,
                            utterance_gap_seconds=0.8,
                            persist_json_artifact=True,
                        ),
                        jobs_module.DiarizationJobState,
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.SubtitleJobManager(
                                job_store=jobs_module.LocalSubtitleJobStore(root / f"subtitle-{id(runtime)}"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedSubtitleJob(
                            audio_record=audio_record,
                            language_code="en",
                            model_size="base",
                            granularity="words",
                            group_size=1,
                            source_text="hello",
                            source_text_name="source.txt",
                            include_srt_text=True,
                        ),
                        jobs_module.SubtitleJobState,
                    ),
                    (
                        self._swap_immediate_executor(
                            jobs_module.ExtractReferenceSampleJobManager(
                                job_store=jobs_module.LocalExtractReferenceSampleJobStore(root / f"extract-{id(runtime)}"),
                                artifact_store=artifact_store,
                                execution_runtime=runtime,
                                max_workers=1,
                                max_pending_jobs=4,
                            )
                        ),
                        jobs_module.PreparedExtractReferenceSampleJob(
                            source_record=source_record,
                            model_size="base",
                            language_code="en",
                            duration_seconds=10.0,
                            max_speech_rate=4.0,
                            min_centroid_hz=500.0,
                            max_centroid_hz=4000.0,
                        ),
                        jobs_module.ExtractReferenceSampleJobState,
                    ),
                )

            for mode, expected_code in (("dictator", "dictator.test.failure"), ("unexpected", "dictator.jobs.failed")):
                runtime = _runtime_for(mode)
                with patch.dict(
                    sys.modules,
                    {
                        "dictator.extraction": types.SimpleNamespace(models=types.SimpleNamespace(ReferenceExtractionRequest=_ReferenceExtractionRequest)),
                        "dictator.extraction.models": types.SimpleNamespace(ReferenceExtractionRequest=_ReferenceExtractionRequest),
                    },
                ):
                    with patch("dictator.runtime.jobs.logging.exception") as logging_exception:
                        for manager, prepared, state_enum in _cases(runtime):
                            with self.subTest(mode=mode, manager=type(manager).__name__):
                                submitted = manager.submit(prepared)
                                record = manager.get(submitted.job_id)
                                self.assertEqual(record.state, state_enum.FAILED)
                                self.assertEqual(record.error_code, expected_code)
                                self.assertIn("broken", record.error_message)
                        if mode == "dictator":
                            logging_exception.assert_not_called()
                        else:
                            self.assertGreaterEqual(logging_exception.call_count, 5)


if __name__ == "__main__":
    unittest.main()
