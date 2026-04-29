from __future__ import annotations

import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc

from dictator.runtime import InflightLimiter, MetricsRegistry
from dictator.speech.v1 import alignment_pb2, subtitle_pb2, transcription_pb2, voice_pb2
from dictator.storage import LocalArtifactStore
from dictator.transport.grpc.context import ServiceContext
try:
    from test_grpc_services_unit import FakeContext, FakeJobManager, FakeRuntime, RpcAbort
except ModuleNotFoundError:  # pragma: no cover - package-style local test runs
    from tests.test_grpc_services_unit import FakeContext, FakeJobManager, FakeRuntime, RpcAbort


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status_code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._status_code = status_code
        self._details = details

    def code(self):
        return self._status_code

    def details(self):
        return self._details


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return types.SimpleNamespace(result=lambda timeout=None: None)


class _CancelJobManager(FakeJobManager):
    def cancel(self, job_id):
        self.lookups.append(job_id)
        return self.record


def _build_service_context(**overrides):
    artifact_store = overrides.pop(
        "artifact_store",
        types.SimpleNamespace(
            get_artifact=MagicMock(
                return_value=types.SimpleNamespace(
                    artifact_id="audio-1",
                    path=Path("/tmp/audio.wav"),
                    filename="audio.wav",
                )
            ),
            read_text=MagicMock(return_value="hello world"),
        ),
    )
    return ServiceContext(
        artifact_store=artifact_store,
        execution_runtime=overrides.pop("execution_runtime", FakeRuntime()),
        metrics=MetricsRegistry(),
        limiter=InflightLimiter(2),
        auth_token="",
        download_chunk_bytes=4,
        **overrides,
    )


class AsyncJobCoverageTests(unittest.TestCase):
    def test_client_static_fallback_helpers_and_default_granularity(self):
        from dictator.client.alignment import AlignmentClient
        from dictator.client.diarization import DiarizationClient
        from dictator.client.dictation import DictationClient
        from dictator.client.subtitles import SubtitleClient

        unimplemented = _FakeRpcError(grpc.StatusCode.UNIMPLEMENTED, "")
        self.assertTrue(AlignmentClient._should_fallback_to_sync(unimplemented))
        self.assertTrue(DiarizationClient._should_fallback_to_sync(unimplemented))
        self.assertTrue(DictationClient._should_fallback_to_sync(unimplemented))
        self.assertTrue(SubtitleClient._should_fallback_to_sync(unimplemented))
        self.assertEqual(
            SubtitleClient._resolve_granularity_name(subtitle_pb2.SUBTITLE_GRANULARITY_UNSPECIFIED),
            "words",
        )

    def test_reference_sample_extract_file_uses_extract_bytes(self):
        from dictator.client import ReferenceSampleClient

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "audio.wav"
            audio.write_bytes(b"audio")

            with (
                patch("dictator.client.voice.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
                patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=object()),
            ):
                client = ReferenceSampleClient(object())

            with patch.object(client, "extract_bytes", return_value="ok") as extract_mock:
                self.assertEqual(client.extract_file(audio, language_code="en"), "ok")
            self.assertEqual(extract_mock.call_args.args[0], b"audio")
            self.assertEqual(extract_mock.call_args.kwargs["filename"], "audio.wav")

    def test_runtime_job_json_helpers_cover_none_and_invalid_types(self):
        from dictator.runtime.jobs import (
            DiarizationJobRecord,
            DiarizationJobState,
            _alignment_words_from_json,
            _subtitle_cues_from_json,
            _transcription_words_from_json,
        )

        self.assertEqual(_alignment_words_from_json(None), ())
        with self.assertRaisesRegex(ValueError, "must be a list"):
            _alignment_words_from_json("bad")

        self.assertEqual(_transcription_words_from_json(None), ())
        with self.assertRaisesRegex(ValueError, "must be a list"):
            _transcription_words_from_json("bad")

        self.assertEqual(_subtitle_cues_from_json(None), ())
        with self.assertRaisesRegex(ValueError, "must be a list"):
            _subtitle_cues_from_json("bad")

        record = DiarizationJobRecord(
            job_id=uuid.uuid4().hex,
            state=DiarizationJobState.QUEUED,
            audio_artifact_id="audio-1",
            include_words=True,
            include_utterances=True,
            include_speakers=True,
            include_speaker_segments=True,
            persist_json_artifact=True,
            created_at_unix_seconds=1.0,
        )
        payload = record.to_json_dict()
        payload["diarization"] = "bad"
        with self.assertRaisesRegex(ValueError, "must be a dict"):
            DiarizationJobRecord.from_json_dict(payload)

    def test_runtime_job_stores_skip_completed_records(self):
        from dictator.runtime.jobs import (
            AlignmentJobRecord,
            AlignmentJobState,
            DiarizationJobRecord,
            DiarizationJobState,
            ExtractReferenceSampleJobRecord,
            ExtractReferenceSampleJobState,
            LocalAlignmentJobStore,
            LocalDiarizationJobStore,
            LocalExtractReferenceSampleJobStore,
            LocalSubtitleJobStore,
            LocalTranscriptionJobStore,
            SubtitleJobRecord,
            SubtitleJobState,
            TranscriptionJobRecord,
            TranscriptionJobState,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = (
                (
                    LocalAlignmentJobStore(root / "alignment"),
                    AlignmentJobRecord(
                        job_id=uuid.uuid4().hex,
                        state=AlignmentJobState.SUCCEEDED,
                        audio_artifact_id="audio-1",
                        include_srt_text=True,
                        created_at_unix_seconds=1.0,
                    ),
                ),
                (
                    LocalTranscriptionJobStore(root / "transcription"),
                    TranscriptionJobRecord(
                        job_id=uuid.uuid4().hex,
                        state=TranscriptionJobState.SUCCEEDED,
                        audio_artifact_id="audio-1",
                        include_word_segments=True,
                        created_at_unix_seconds=1.0,
                    ),
                ),
                (
                    LocalDiarizationJobStore(root / "diarization"),
                    DiarizationJobRecord(
                        job_id=uuid.uuid4().hex,
                        state=DiarizationJobState.SUCCEEDED,
                        audio_artifact_id="audio-1",
                        include_words=True,
                        include_utterances=True,
                        include_speakers=True,
                        include_speaker_segments=True,
                        persist_json_artifact=True,
                        created_at_unix_seconds=1.0,
                    ),
                ),
                (
                    LocalSubtitleJobStore(root / "subtitle"),
                    SubtitleJobRecord(
                        job_id=uuid.uuid4().hex,
                        state=SubtitleJobState.SUCCEEDED,
                        audio_artifact_id="audio-1",
                        include_srt_text=True,
                        created_at_unix_seconds=1.0,
                    ),
                ),
                (
                    LocalExtractReferenceSampleJobStore(root / "extract"),
                    ExtractReferenceSampleJobRecord(
                        job_id=uuid.uuid4().hex,
                        state=ExtractReferenceSampleJobState.SUCCEEDED,
                        source_artifact_id="audio-1",
                        created_at_unix_seconds=1.0,
                    ),
                ),
            )

            for store, record in cases:
                with self.subTest(store=type(store).__name__):
                    store._write_record(record)
                    store.fail_incomplete_jobs("ignored")
                    self.assertEqual(store.get(record.job_id).state, record.state)

    def test_subtitle_job_manager_loads_whisper_model_when_needed(self):
        from dictator.runtime.jobs import LocalSubtitleJobStore, PreparedSubtitleJob, SubtitleJobManager, SubtitleJobState

        class _SubtitleService:
            def __init__(self, source_path):
                self._source_path = source_path

            def render(self, request, **_kwargs):
                for value in vars(request).values():
                    if isinstance(value, Path) and value != self._source_path:
                        value.parent.mkdir(parents=True, exist_ok=True)
                        value.write_text("1\n00:00:00,000 --> 00:00:00,400\nhello\n", encoding="utf-8")
                return types.SimpleNamespace(
                    language="en",
                    mode="transcription",
                    output_format="srt",
                    granularity="words",
                    group_size=1,
                    cues=(
                        types.SimpleNamespace(
                            index=1,
                            text="hello",
                            start_seconds=0.0,
                            end_seconds=0.4,
                            item_count=1,
                        ),
                    ),
                    srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello\n",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_store = LocalArtifactStore(root / "artifacts")
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"audio")
            audio_record = types.SimpleNamespace(artifact_id="audio-1", path=audio_path, filename="audio.wav")
            runtime = types.SimpleNamespace(
                get_subtitle_service=lambda: _SubtitleService(audio_path),
                get_whisper_model=MagicMock(return_value=object()),
            )
            manager = SubtitleJobManager(
                job_store=LocalSubtitleJobStore(root / "jobs"),
                artifact_store=artifact_store,
                execution_runtime=runtime,
                max_workers=1,
                max_pending_jobs=2,
            )
            original = manager._executor
            original.shutdown(wait=False, cancel_futures=True)
            manager._executor = _ImmediateExecutor()

            submitted = manager.submit(
                PreparedSubtitleJob(
                    audio_record=audio_record,
                    language_code="en",
                    model_size="base",
                    granularity="words",
                    group_size=1,
                    source_text=None,
                    source_text_name="source.txt",
                    include_srt_text=True,
                )
            )
            record = manager.get(submitted.job_id)

        self.assertEqual(record.state, SubtitleJobState.SUCCEEDED)
        runtime.get_whisper_model.assert_called_once_with("base")

    def test_alignment_service_get_job_requires_manager(self):
        from dictator.transport.grpc.alignment_service import AlignmentServiceServicer

        servicer = AlignmentServiceServicer(_build_service_context(alignment_job_manager=None))
        with self.assertRaises(RpcAbort) as raised:
            servicer.GetAlignTranscriptJob(
                alignment_pb2.GetAlignTranscriptJobRequest(job_id="job-1"),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

    def test_subtitle_service_job_mapping_and_validation(self):
        from dictator.runtime.jobs import SubtitleJobRecord, SubtitleJobState
        from dictator.transport.grpc.subtitle_service import SubtitleServiceServicer

        record = SubtitleJobRecord(
            job_id="subtitle-job",
            state=SubtitleJobState.QUEUED,
            audio_artifact_id="audio-1",
            include_srt_text=True,
            created_at_unix_seconds=1.0,
            mode="forced_alignment",
            output_format="srt",
            granularity="sentences",
            group_size=2,
        )
        servicer = SubtitleServiceServicer(
            _build_service_context(subtitle_job_manager=FakeJobManager(record))
        )
        response = servicer._subtitle_job_response(record)
        self.assertEqual(response.mode, subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT)
        self.assertEqual(response.granularity, subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES)

        missing_manager = SubtitleServiceServicer(_build_service_context(subtitle_job_manager=None))
        with self.assertRaises(RpcAbort) as raised:
            missing_manager.GetRenderSubtitlesJob(
                subtitle_pb2.GetRenderSubtitlesJobRequest(job_id="job-1"),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        with self.assertRaises(RpcAbort) as raised:
            servicer.GetRenderSubtitlesJob(
                subtitle_pb2.GetRenderSubtitlesJobRequest(job_id=" "),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

    def test_transcription_service_job_validation_branches(self):
        from dictator.runtime.jobs import (
            DiarizationJobRecord,
            DiarizationJobState,
            TranscriptionJobRecord,
            TranscriptionJobState,
        )
        from dictator.transport.grpc.transcription_service import TranscriptionServiceServicer

        transcribe_record = TranscriptionJobRecord(
            job_id="transcribe-job",
            state=TranscriptionJobState.QUEUED,
            audio_artifact_id="audio-1",
            include_word_segments=True,
            created_at_unix_seconds=1.0,
        )
        diarization_record = DiarizationJobRecord(
            job_id="diarize-job",
            state=DiarizationJobState.QUEUED,
            audio_artifact_id="audio-1",
            include_words=True,
            include_utterances=True,
            include_speakers=True,
            include_speaker_segments=True,
            persist_json_artifact=True,
            created_at_unix_seconds=1.0,
        )
        servicer = TranscriptionServiceServicer(
            _build_service_context(
                transcription_job_manager=FakeJobManager(transcribe_record),
                diarization_job_manager=FakeJobManager(diarization_record),
            )
        )
        with self.assertRaises(RpcAbort) as raised:
            servicer.GetTranscribeJob(
                transcription_pb2.GetTranscribeJobRequest(job_id=" "),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        missing_diarization = TranscriptionServiceServicer(
            _build_service_context(
                transcription_job_manager=FakeJobManager(transcribe_record),
                diarization_job_manager=None,
            )
        )
        with self.assertRaises(RpcAbort) as raised:
            missing_diarization.GetDiarizeAudioJob(
                transcription_pb2.GetDiarizeAudioJobRequest(job_id="job-1"),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        with self.assertRaises(RpcAbort) as raised:
            servicer.GetDiarizeAudioJob(
                transcription_pb2.GetDiarizeAudioJobRequest(job_id=" "),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

    def test_voice_reference_job_validation_branches(self):
        from dictator.runtime.jobs import ExtractReferenceSampleJobRecord, ExtractReferenceSampleJobState
        from dictator.transport.grpc.voice_service import VoiceServiceServicer

        missing_manager = VoiceServiceServicer(
            _build_service_context(reference_extraction_job_manager=None)
        )
        with self.assertRaises(RpcAbort) as raised:
            missing_manager.SubmitExtractReferenceSampleJob(
                voice_pb2.ExtractReferenceSampleRequest(source_artifact_id="audio-1"),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        record = ExtractReferenceSampleJobRecord(
            job_id="reference-job",
            state=ExtractReferenceSampleJobState.QUEUED,
            source_artifact_id="audio-1",
            created_at_unix_seconds=1.0,
        )
        servicer = VoiceServiceServicer(
            _build_service_context(reference_extraction_job_manager=FakeJobManager(record))
        )
        with self.assertRaises(RpcAbort) as raised:
            servicer.GetExtractReferenceSampleJob(
                voice_pb2.GetExtractReferenceSampleJobRequest(job_id=" "),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

    def test_parse_positive_float_rejects_non_positive(self):
        from dictator.transport.grpc.config import _parse_positive_float

        with self.assertRaisesRegex(ValueError, "must be positive"):
            _parse_positive_float("0", "poll interval")

    def test_cancel_job_grpc_handlers_map_canceled_and_validate_requests(self):
        from dictator.runtime.jobs import (
            AlignmentJobRecord,
            AlignmentJobState,
            DiarizationJobRecord,
            DiarizationJobState,
            ExtractReferenceSampleJobRecord,
            ExtractReferenceSampleJobState,
            SubtitleJobRecord,
            SubtitleJobState,
            SynthesisJobRecord,
            SynthesisJobState,
            TranscriptionJobRecord,
            TranscriptionJobState,
        )
        from dictator.transport.grpc.alignment_service import AlignmentServiceServicer
        from dictator.transport.grpc.subtitle_service import SubtitleServiceServicer
        from dictator.transport.grpc.transcription_service import TranscriptionServiceServicer
        from dictator.transport.grpc.voice_service import VoiceServiceServicer

        cases = (
            (
                TranscriptionServiceServicer,
                "transcription_job_manager",
                TranscriptionJobRecord("tx-job", TranscriptionJobState.CANCELED, "audio-1", True, 1.0),
                "CancelTranscribeJob",
                transcription_pb2.CancelTranscribeJobRequest,
                transcription_pb2.TRANSCRIPTION_JOB_STATE_CANCELED,
            ),
            (
                TranscriptionServiceServicer,
                "diarization_job_manager",
                DiarizationJobRecord("dia-job", DiarizationJobState.CANCELED, "audio-1", True, True, True, False, True, 1.0),
                "CancelDiarizeAudioJob",
                transcription_pb2.CancelDiarizeAudioJobRequest,
                transcription_pb2.DIARIZATION_JOB_STATE_CANCELED,
            ),
            (
                AlignmentServiceServicer,
                "alignment_job_manager",
                AlignmentJobRecord("align-job", AlignmentJobState.CANCELED, "audio-1", True, 1.0),
                "CancelAlignTranscriptJob",
                alignment_pb2.CancelAlignTranscriptJobRequest,
                alignment_pb2.ALIGNMENT_JOB_STATE_CANCELED,
            ),
            (
                SubtitleServiceServicer,
                "subtitle_job_manager",
                SubtitleJobRecord("sub-job", SubtitleJobState.CANCELED, "audio-1", True, 1.0),
                "CancelRenderSubtitlesJob",
                subtitle_pb2.CancelRenderSubtitlesJobRequest,
                subtitle_pb2.SUBTITLE_JOB_STATE_CANCELED,
            ),
            (
                VoiceServiceServicer,
                "reference_extraction_job_manager",
                ExtractReferenceSampleJobRecord("ref-job", ExtractReferenceSampleJobState.CANCELED, "source-1", 1.0),
                "CancelExtractReferenceSampleJob",
                voice_pb2.CancelExtractReferenceSampleJobRequest,
                voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_CANCELED,
            ),
            (
                VoiceServiceServicer,
                "synthesis_job_manager",
                SynthesisJobRecord("syn-job", SynthesisJobState.CANCELED, "qwen3", "en", False, "speaker-1", 1.0),
                "CancelSynthesizeSpeechJob",
                voice_pb2.CancelSynthesizeSpeechJobRequest,
                voice_pb2.SYNTHESIS_JOB_STATE_CANCELED,
            ),
        )

        for servicer_class, manager_name, record, method_name, request_class, expected_state in cases:
            with self.subTest(method=method_name):
                manager = _CancelJobManager(record)
                servicer = servicer_class(_build_service_context(**{manager_name: manager}))
                response = getattr(servicer, method_name)(request_class(job_id=record.job_id), FakeContext())
                self.assertEqual(response.job_id, record.job_id)
                self.assertEqual(response.state, expected_state)
                self.assertEqual(manager.lookups, [record.job_id])

                with self.assertRaises(RpcAbort) as blank:
                    getattr(servicer, method_name)(request_class(job_id=" "), FakeContext())
                self.assertEqual(blank.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

                missing_manager = servicer_class(_build_service_context(**{manager_name: None}))
                with self.assertRaises(RpcAbort) as missing:
                    getattr(missing_manager, method_name)(request_class(job_id=record.job_id), FakeContext())
                self.assertEqual(missing.exception.status, grpc.StatusCode.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
