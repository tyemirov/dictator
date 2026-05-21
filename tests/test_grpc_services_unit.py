from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import grpc

from dictator.diarization.models import DiarizeAudioResult, DiarizedUtterance, DiarizedWord, SpeakerSummary, SpeakerSegment
from dictator.runtime.jobs import SynthesisJobRecord, SynthesisJobState
from dictator.runtime import DependencyError, InflightLimiter, MetricsRegistry, ProcessingError, ServiceRequestError, ValidationError
from dictator.speech.v1 import alignment_pb2, artifacts_pb2, common_pb2, subtitle_pb2, transcription_pb2, voice_pb2
from dictator.storage import LocalArtifactStore
from dictator.synthesis.models import DEFAULT_SYNTHESIS_AUDIO_FORMAT, SynthesisEngine
from dictator.transport.grpc.services import (
    AlignmentServiceServicer,
    ArtifactServiceServicer,
    BaseServicer,
    ServiceContext,
    SubtitleServiceServicer,
    TranscriptionServiceServicer,
    VoiceServiceServicer,
)
from dictator.transcription.models import TranscriptionResult, WordSegment


class RpcAbort(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class FakeContext:
    def __init__(self, *, active=True, remaining=1.0, metadata=()):
        self._active = active
        self._remaining = remaining
        self._metadata = tuple(metadata)
        self.trailing_metadata = ()

    def is_active(self):
        return self._active

    def time_remaining(self):
        return self._remaining

    def invocation_metadata(self):
        return self._metadata

    def set_trailing_metadata(self, metadata):
        self.trailing_metadata = metadata

    def abort(self, status, message):
        raise RpcAbort(status, message)


class RaisingLimiter:
    @contextmanager
    def acquire(self):
        raise ServiceRequestError(None, "dictator.runtime.inflight_limit", "too many inflight requests")
        yield


class FakeTranscriptionService:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, model_size="base", model=None, progress_cb=None):
        self.calls.append((audio, language, model_size, model))
        return TranscriptionResult(
            language=language or "en",
            words=(WordSegment("hello", 0.0, 0.4), WordSegment("world", 0.4, 0.8)),
        )


class FakeDiarizationService:
    def __init__(self):
        self.calls = []

    def diarize(self, request, model=None, diarization_pipeline=None):
        self.calls.append((request, model, diarization_pipeline))
        words = (DiarizedWord("hello", 0.0, 0.4, "S1"),)
        utterances = (DiarizedUtterance("S1", 0.0, 0.4, "hello", words),)
        speakers = (SpeakerSummary("S1", 1, 1, 0.4),)
        segments = (SpeakerSegment("S1", 0.0, 0.5, raw_label="speaker_a"),)
        return DiarizeAudioResult(
            language=request.language or "en",
            text="hello",
            words=words,
            utterances=utterances,
            speakers=speakers,
            speaker_segments=segments,
        )


class FakeAlignmentService:
    def __init__(self):
        self.calls = []

    def align(self, request):
        self.calls.append(request)
        request.output_srt_path.write_text("srt", encoding="utf-8")
        return types.SimpleNamespace(
            language=request.language or "en",
            words=(types.SimpleNamespace(text="hello", start_seconds=0.0, end_seconds=0.4),),
            srt_text="srt",
        )


class FakeSubtitleService:
    def __init__(self, mode="transcription"):
        self.mode = mode
        self.calls = []

    def render(self, request, model=None):
        self.calls.append((request, model))
        if request.output_srt_path is not None:
            request.output_srt_path.write_text("1\n00:00:00,000 --> 00:00:00,400\nhello\n", encoding="utf-8")
        return types.SimpleNamespace(
            language=request.language or "en",
            mode=self.mode,
            group_size=request.group_size,
            cues=(types.SimpleNamespace(text="hello", start_seconds=0.0, end_seconds=0.4, item_count=1),),
            srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello\n",
        )


class FakeExtractionService:
    def extract(self, request, model=None, diarization_pipeline=None):
        request.output_path.write_bytes(b"wav")
        return types.SimpleNamespace(
            trim_start_seconds=0.1,
            trim_end_seconds=0.5,
            window_start_seconds=0.0,
            window_end_seconds=0.4,
            dominant_speaker_words=(1, 2),
        )


class FakeSynthesisService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def synthesise(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def synthesise_text(self, request, *, progress_callback=None):
        self.calls.append(request)
        return self.result


class FakeRuntime:
    def __init__(self, *, subtitle_mode="transcription", synthesis_result=None):
        self.transcription_service = FakeTranscriptionService()
        self.diarization_service = FakeDiarizationService()
        self.alignment_service = FakeAlignmentService()
        self.subtitle_service = FakeSubtitleService(mode=subtitle_mode)
        self.extraction_service = FakeExtractionService()
        self.synthesis_service = FakeSynthesisService(synthesis_result)
        self.whisper_model = object()
        self.pipeline = object()
        self.mark_synthesis_ready_calls = 0

    def get_transcription_service(self):
        return self.transcription_service

    def get_diarization_service(self):
        return self.diarization_service

    def get_alignment_service(self):
        return self.alignment_service

    def get_subtitle_service(self):
        return self.subtitle_service

    def get_reference_extraction_service(self):
        return self.extraction_service

    def get_synthesis_service(self):
        return self.synthesis_service

    def get_whisper_model(self, model_size):
        return self.whisper_model

    def get_diarization_pipeline(self):
        return self.pipeline

    def mark_synthesis_ready(self):
        self.mark_synthesis_ready_calls += 1


class FakeJobManager:
    def __init__(self, record):
        self.record = record
        self.submitted = []
        self.lookups = []

    def submit(self, prepared):
        self.submitted.append(prepared)
        return self.record

    def get(self, job_id):
        self.lookups.append(job_id)
        return self.record


class GrpcServicesUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        artifact_store = LocalArtifactStore(Path(self.tmpdir.name))
        self.runtime = FakeRuntime(
            synthesis_result=types.SimpleNamespace(
                wav_paths=(Path(self.tmpdir.name) / "chunk.wav",),
                segments=(types.SimpleNamespace(end_seconds=1.0, to_timeline_dict=lambda: {"content": "hello", "start": 0.0, "end": 1.0}),),
            )
        )
        self.context = ServiceContext(
            artifact_store=artifact_store,
            execution_runtime=self.runtime,
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(2),
            auth_token="secret",
            download_chunk_bytes=4,
        )
        self.audio_record = artifact_store.write_artifact([b"audio"], filename="sample.wav", media_type="audio/wav")
        self.text_record = artifact_store.write_artifact([b"hello world"], filename="transcript.txt", media_type="text/plain")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_base_servicer_helpers_and_request_scope_mappings(self):
        servicer = BaseServicer(self.context)

        with self.assertRaises(RpcAbort) as exc:
            servicer._ensure_request_active(FakeContext(active=False, metadata=(("x-dictator-token", "secret"),)))
        self.assertEqual(exc.exception.status, grpc.StatusCode.CANCELLED)

        with self.assertRaises(RpcAbort) as exc:
            servicer._ensure_request_active(FakeContext(remaining=0.0, metadata=(("x-dictator-token", "secret"),)))
        self.assertEqual(exc.exception.status, grpc.StatusCode.DEADLINE_EXCEEDED)

        auth_free = BaseServicer(ServiceContext(self.context.artifact_store, self.runtime, MetricsRegistry(), InflightLimiter(1), None, 4))
        auth_free._require_auth(FakeContext())
        servicer._require_auth(FakeContext(metadata=(("authorization", "Bearer secret"),)))
        with self.assertRaises(RpcAbort) as exc:
            servicer._require_auth(FakeContext(metadata=()))
        self.assertEqual(exc.exception.status, grpc.StatusCode.UNAUTHENTICATED)

        self.assertEqual(servicer._resolve_language_request(language_code="en", autodetect_language=False, error_scope="x"), "en")
        self.assertEqual(servicer._resolve_language_request(language_code="", autodetect_language=True, error_scope="x"), None)
        with self.assertRaises(ValidationError):
            servicer._resolve_language_request(language_code="en", autodetect_language=True, error_scope="x")
        with self.assertRaises(ValidationError):
            servicer._resolve_language_request(language_code="", autodetect_language=False, error_scope="x")

        self.assertEqual(servicer._artifact_ref(self.audio_record).artifact_id, self.audio_record.artifact_id)
        self.assertEqual(servicer._word_segment({"content": "hi", "start": 1.0, "end": 2.0}).content, "hi")
        self.assertEqual(servicer._timeline_segment({"content": "hi", "start": 1.0, "end": 2.0}).start_seconds, 1.0)

        error_cases = [
            (ValidationError("code.validation", "bad"), grpc.StatusCode.INVALID_ARGUMENT),
            (DependencyError("code.dependency", "bad"), grpc.StatusCode.FAILED_PRECONDITION),
            (ProcessingError("code.processing", "bad"), grpc.StatusCode.INTERNAL),
            (FileNotFoundError("missing"), grpc.StatusCode.NOT_FOUND),
            (ValueError("bad"), grpc.StatusCode.INVALID_ARGUMENT),
        ]
        for error, expected_status in error_cases:
            ctx = FakeContext(metadata=(("x-dictator-token", "secret"),))
            with self.assertRaises(RpcAbort) as exc:
                with servicer._request_scope(ctx, bytes_received=5):
                    raise error
            self.assertEqual(exc.exception.status, expected_status)

        ctx = FakeContext(metadata=(("x-dictator-token", "secret"),))
        busy_servicer = BaseServicer(
            ServiceContext(
                self.context.artifact_store,
                self.runtime,
                MetricsRegistry(),
                RaisingLimiter(),
                "secret",
                4,
            )
        )
        with self.assertRaises(RpcAbort) as exc:
            with busy_servicer._request_scope(ctx):
                pass
        self.assertEqual(exc.exception.status, grpc.StatusCode.RESOURCE_EXHAUSTED)

    def test_artifact_service_validation_branches(self):
        servicer = ArtifactServiceServicer(self.context)
        context = FakeContext(metadata=(("x-dictator-token", "secret"),))
        with self.assertRaises(RpcAbort) as exc:
            servicer.UploadArtifact(iter(()), context)
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        context = FakeContext(metadata=(("x-dictator-token", "secret"),))
        bad_stream = iter([artifacts_pb2.UploadArtifactChunk(content=b"abc")])
        with self.assertRaises(RpcAbort):
            servicer.UploadArtifact(bad_stream, context)

        context = FakeContext(metadata=(("x-dictator-token", "secret"),))
        stream = iter([
            artifacts_pb2.UploadArtifactChunk(metadata=artifacts_pb2.UploadArtifactMetadata(filename="a.bin", media_type="application/octet-stream")),
            artifacts_pb2.UploadArtifactChunk(metadata=artifacts_pb2.UploadArtifactMetadata(filename="oops", media_type="application/octet-stream")),
        ])
        before = {path.name for path in Path(self.tmpdir.name).iterdir()}
        with self.assertRaises(RpcAbort):
            servicer.UploadArtifact(stream, context)
        after = {path.name for path in Path(self.tmpdir.name).iterdir()}
        self.assertEqual(after, before)

    def test_transcription_and_alignment_servicer_branches(self):
        context = FakeContext(metadata=(("x-dictator-token", "secret"),))
        servicer = TranscriptionServiceServicer(self.context)
        response = servicer.DiarizeAudio(
            transcription_pb2.DiarizeAudioRequest(
                audio_artifact_id=self.audio_record.artifact_id,
                autodetect_language=True,
                model_size="base",
            ),
            context,
        )
        diarize_request = self.runtime.diarization_service.calls[0][0]
        self.assertTrue(diarize_request.include_words)
        self.assertTrue(diarize_request.include_utterances)
        self.assertTrue(diarize_request.include_speakers)
        self.assertEqual(response.text, "hello")

        aligner = AlignmentServiceServicer(self.context)
        response = aligner.AlignTranscript(
            alignment_pb2.AlignTranscriptRequest(
                audio_artifact_id=self.audio_record.artifact_id,
                transcript_artifact_id=self.text_record.artifact_id,
                language_code="en",
                include_srt_text=True,
            ),
            FakeContext(metadata=(("x-dictator-token", "secret"),)),
        )
        self.assertEqual(self.runtime.alignment_service.calls[0].transcript_text, "hello world")
        self.assertEqual(response.srt_text, "srt")

        with self.assertRaises(RpcAbort):
            aligner.AlignTranscript(
                alignment_pb2.AlignTranscriptRequest(audio_artifact_id=self.audio_record.artifact_id),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )

    def test_subtitle_servicer_branches(self):
        servicer = SubtitleServiceServicer(self.context)
        with self.assertRaises(RpcAbort):
            servicer.RenderSubtitles(
                subtitle_pb2.RenderSubtitlesRequest(
                    audio_artifact_id=self.audio_record.artifact_id,
                    language_code="en",
                    output_format=subtitle_pb2.SUBTITLE_FORMAT_UNSPECIFIED + 10,
                    autodetect_language=False,
                    granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
                    group_size=1,
                ),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        with self.assertRaises(RpcAbort):
            servicer.RenderSubtitles(
                subtitle_pb2.RenderSubtitlesRequest(
                    audio_artifact_id=self.audio_record.artifact_id,
                    language_code="en",
                    output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                    autodetect_language=False,
                    granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
                    group_size=0,
                ),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        response = servicer.RenderSubtitles(
            subtitle_pb2.RenderSubtitlesRequest(
                audio_artifact_id=self.audio_record.artifact_id,
                source_text_artifact_id=self.text_record.artifact_id,
                language_code="en",
                autodetect_language=False,
                output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                granularity=subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES,
                group_size=1,
                include_srt_text=True,
            ),
            FakeContext(metadata=(("x-dictator-token", "secret"),)),
        )
        self.assertEqual(self.runtime.subtitle_service.calls[0][0].source_text, "hello world")
        self.assertEqual(self.runtime.subtitle_service.calls[0][0].source_text_name, "transcript.txt")
        self.assertEqual(response.cues[0].content, "hello")

    def test_voice_servicer_branches(self):
        servicer = VoiceServiceServicer(self.context)
        self.assertEqual(servicer._resolve_synthesis_engine(voice_pb2.SYNTHESIS_ENGINE_QWEN3), SynthesisEngine.QWEN3)
        with self.assertRaisesRegex(ValidationError, "synthesis_engine must be set"):
            servicer._resolve_synthesis_engine(voice_pb2.SYNTHESIS_ENGINE_UNSPECIFIED)
        self.assertEqual(
            servicer._resolve_synthesis_audio_format(
                voice_pb2.SynthesizeSpeechRequest(audio_format=common_pb2.AudioFormat(sample_rate_hz=24000))
            ),
            DEFAULT_SYNTHESIS_AUDIO_FORMAT,
        )
        self.assertEqual(
            servicer._resolve_speaker_transcript_text(
                types.SimpleNamespace(
                    speaker_transcript_text="sample transcript",
                )
            ),
            "sample transcript",
        )
        with patch.dict(
            sys.modules,
            {
                "librosa": types.SimpleNamespace(feature=types.SimpleNamespace(rms=lambda y: None)),
                "ffmpeg": types.ModuleType("ffmpeg"),
            },
        ):
            response = servicer.ExtractReferenceSample(
                voice_pb2.ExtractReferenceSampleRequest(source_artifact_id=self.audio_record.artifact_id),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        self.assertEqual(response.dominant_speaker_word_count, 2)
        self.assertTrue(response.sample_artifact.artifact_id)

        with self.assertRaises(RpcAbort):
            servicer.SynthesizeSpeech(
                voice_pb2.SynthesizeSpeechRequest(
                    speaker_artifact_id=self.audio_record.artifact_id,
                    synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_UNSPECIFIED,
                ),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )

        with self.assertRaises(RpcAbort):
            servicer.SynthesizeSpeech(
                voice_pb2.SynthesizeSpeechRequest(
                    speaker_artifact_id=self.audio_record.artifact_id,
                    text="hello",
                    synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_QWEN3,
                    audio_format=common_pb2.AudioFormat(sample_rate_hz=48000),
                ),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )

        fake_ffmpeg_module = types.ModuleType("dictator.audio.ffmpeg_ops")
        fake_ffmpeg_module.concat_normalise = lambda inputs, dst, cap, target_sample_rate=0: dst.write_bytes(b"wav")
        cleanup_calls = []
        fake_synthesis_service = types.ModuleType("dictator.synthesis.service")
        fake_synthesis_service.cleanup_synthesis_result = lambda result: cleanup_calls.append(result)
        with patch.dict(
            sys.modules,
            {
                "dictator.audio.ffmpeg_ops": fake_ffmpeg_module,
                "dictator.synthesis.service": fake_synthesis_service,
            },
        ):
            response = servicer.SynthesizeSpeech(
                voice_pb2.SynthesizeSpeechRequest(
                    speaker_artifact_id=self.audio_record.artifact_id,
                    text_artifact_id=self.text_record.artifact_id,
                    include_timeline=False,
                    synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_QWEN3,
                ),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        self.assertEqual(response.chunk_count, 1)
        self.assertFalse(response.timeline_artifact_id)
        self.assertEqual(response.resolved_audio_format.container, common_pb2.AUDIO_CONTAINER_WAV)
        self.assertEqual(response.resolved_audio_format.codec, common_pb2.AUDIO_CODEC_PCM_S16LE)
        self.assertEqual(response.resolved_audio_format.sample_rate_hz, 24000)
        self.assertEqual(response.resolved_audio_format.channel_count, 1)
        self.assertEqual(response.resolved_audio_format.bit_depth, 16)
        self.assertEqual(self.runtime.synthesis_service.calls[0].speaker_artifact_id, self.audio_record.artifact_id)
        self.assertEqual(len(cleanup_calls), 1)
        self.assertEqual(self.runtime.mark_synthesis_ready_calls, 1)

        timeline_result = types.SimpleNamespace(
            wav_paths=(Path(self.tmpdir.name) / "chunk.wav",),
            segments=(types.SimpleNamespace(end_seconds=1.0, to_timeline_dict=lambda: {"content": "hello", "start": 0.0, "end": 1.0}),),
        )
        self.runtime.synthesis_service = FakeSynthesisService(timeline_result)
        with patch.dict(
            sys.modules,
            {
                "dictator.audio.ffmpeg_ops": fake_ffmpeg_module,
                "dictator.synthesis.service": fake_synthesis_service,
            },
        ):
            response = servicer.SynthesizeSpeech(
                voice_pb2.SynthesizeSpeechRequest(
                    speaker_artifact_id=self.audio_record.artifact_id,
                    text="hello",
                    include_timeline=True,
                    synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_QWEN3,
                ),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        self.assertTrue(response.timeline_artifact_id)
        self.assertEqual(response.timeline[0].content, "hello")

    def test_voice_servicer_job_submission_and_lookup(self):
        audio_record = self.context.artifact_store.write_artifact([b"wav"], filename="result.wav", media_type="audio/wav")
        manager = FakeJobManager(
            SynthesisJobRecord(
                job_id="job-1",
                state=SynthesisJobState.SUCCEEDED,
                engine="qwen3",
                language_code="en",
                include_timeline=True,
                speaker_artifact_id=self.audio_record.artifact_id,
                created_at_unix_seconds=1.0,
                started_at_unix_seconds=2.0,
                finished_at_unix_seconds=3.0,
                audio_artifact_id=audio_record.artifact_id,
                audio_duration_seconds=4.0,
                audio_format=None,
                timeline_artifact_id="timeline-1",
                chunk_count=2,
            )
        )
        context = ServiceContext(
            artifact_store=self.context.artifact_store,
            execution_runtime=self.runtime,
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(2),
            auth_token="secret",
            download_chunk_bytes=4,
            synthesis_job_manager=manager,
        )
        servicer = VoiceServiceServicer(context)

        submit = servicer.SubmitSynthesizeSpeechJob(
            voice_pb2.SynthesizeSpeechRequest(
                speaker_artifact_id=self.audio_record.artifact_id,
                text="hello world",
                include_timeline=True,
                synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_QWEN3,
                speaker_transcript_text="sample transcript",
            ),
            FakeContext(metadata=(("x-dictator-token", "secret"),)),
        )
        self.assertEqual(submit.job_id, "job-1")
        self.assertEqual(submit.state, voice_pb2.SYNTHESIS_JOB_STATE_SUCCEEDED)
        self.assertEqual(manager.submitted[0].synthesis_request.text, "hello world")

        lookup = servicer.GetSynthesizeSpeechJob(
            voice_pb2.GetSynthesizeSpeechJobRequest(job_id="job-1"),
            FakeContext(metadata=(("x-dictator-token", "secret"),)),
        )
        self.assertEqual(lookup.audio_artifact.artifact_id, audio_record.artifact_id)
        self.assertEqual(lookup.resolved_audio_format.sample_rate_hz, 24000)
        self.assertEqual(lookup.chunk_count, 2)
        self.assertEqual(manager.lookups, ["job-1"])

    def test_voice_servicer_rejects_job_requests_without_manager_or_job_id(self):
        servicer = VoiceServiceServicer(self.context)
        with self.assertRaises(RpcAbort) as exc:
            servicer.SubmitSynthesizeSpeechJob(
                voice_pb2.SynthesizeSpeechRequest(
                    speaker_artifact_id=self.audio_record.artifact_id,
                    text="hello",
                    synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_QWEN3,
                ),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        with self.assertRaises(RpcAbort) as exc:
            servicer.GetSynthesizeSpeechJob(
                voice_pb2.GetSynthesizeSpeechJobRequest(job_id="job-1"),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        manager = FakeJobManager(
            SynthesisJobRecord(
                job_id="job-1",
                state=SynthesisJobState.QUEUED,
                engine="qwen3",
                language_code="en",
                include_timeline=False,
                speaker_artifact_id=self.audio_record.artifact_id,
                created_at_unix_seconds=1.0,
            )
        )
        context = ServiceContext(
            artifact_store=self.context.artifact_store,
            execution_runtime=self.runtime,
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(2),
            auth_token="secret",
            download_chunk_bytes=4,
            synthesis_job_manager=manager,
        )
        servicer = VoiceServiceServicer(context)
        with self.assertRaises(RpcAbort) as exc:
            servicer.GetSynthesizeSpeechJob(
                voice_pb2.GetSynthesizeSpeechJobRequest(job_id=""),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        with self.assertRaises(RpcAbort) as exc:
            servicer.GetSynthesizeSpeechJob(
                voice_pb2.GetSynthesizeSpeechJobRequest(job_id="../escape"),
                FakeContext(metadata=(("x-dictator-token", "secret"),)),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)


class AlignmentJobGrpcServiceTests(unittest.TestCase):
    def _make_service(self, *, record):
        import types
        from pathlib import Path
        from unittest.mock import MagicMock

        from dictator.transport.grpc.alignment_service import AlignmentServiceServicer

        artifact_store = types.SimpleNamespace(
            get_artifact=MagicMock(
                return_value=types.SimpleNamespace(
                    artifact_id="audio-1",
                    path=Path("/tmp/audio.wav"),
                    filename="audio.wav",
                )
            ),
            read_text=MagicMock(return_value="hello world"),
        )
        service_context = ServiceContext(
            artifact_store=artifact_store,
            execution_runtime=FakeRuntime(),
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(2),
            auth_token="",
            download_chunk_bytes=4,
            alignment_job_manager=FakeJobManager(record),
        )
        return AlignmentServiceServicer(service_context), service_context

    def test_submit_alignment_job_uses_job_manager(self):
        from dictator.runtime.jobs import AlignmentJobRecord, AlignmentJobState

        service, service_context = self._make_service(
            record=AlignmentJobRecord(
                job_id="align-1",
                state=AlignmentJobState.QUEUED,
                audio_artifact_id="audio-1",
                include_srt_text=True,
                created_at_unix_seconds=1.0,
            )
        )

        response = service.SubmitAlignTranscriptJob(
            alignment_pb2.AlignTranscriptRequest(
                audio_artifact_id="audio-1",
                transcript_text="hello world",
                language_code="en",
                remove_punctuation=True,
                include_srt_text=True,
            ),
            FakeContext(),
        )

        self.assertEqual(response.job_id, "align-1")
        self.assertEqual(response.state, alignment_pb2.ALIGNMENT_JOB_STATE_QUEUED)
        prepared = service_context.alignment_job_manager.submitted[0]
        self.assertEqual(prepared.audio_record.artifact_id, "audio-1")
        self.assertEqual(prepared.transcript_text, "hello world")
        self.assertTrue(prepared.remove_punctuation)
        self.assertTrue(prepared.include_srt_text)

    def test_get_alignment_job_returns_marshaled_record(self):
        from dictator.alignment.models import AlignedWord
        from dictator.runtime.jobs import AlignmentJobRecord, AlignmentJobState

        service, _ = self._make_service(
            record=AlignmentJobRecord(
                job_id="align-2",
                state=AlignmentJobState.SUCCEEDED,
                audio_artifact_id="audio-1",
                include_srt_text=True,
                created_at_unix_seconds=1.0,
                started_at_unix_seconds=2.0,
                finished_at_unix_seconds=3.0,
                language_code="en",
                words=(AlignedWord(text="hello", start_seconds=0.0, end_seconds=0.4),),
                srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello\n",
                srt_artifact_id="srt-1",
            )
        )

        response = service.GetAlignTranscriptJob(
            alignment_pb2.GetAlignTranscriptJobRequest(job_id="align-2"),
            FakeContext(),
        )

        self.assertEqual(response.job_id, "align-2")
        self.assertEqual(response.state, alignment_pb2.ALIGNMENT_JOB_STATE_SUCCEEDED)
        self.assertEqual(response.language_code, "en")
        self.assertEqual(response.words[0].content, "hello")
        self.assertEqual(response.srt_artifact_id, "srt-1")

    def test_alignment_job_response_maps_all_states_and_validates_job_id(self):
        from dictator.runtime.jobs import AlignmentJobRecord, AlignmentJobState

        service, service_context = self._make_service(
            record=AlignmentJobRecord(
                job_id="align-3",
                state=AlignmentJobState.QUEUED,
                audio_artifact_id="audio-1",
                include_srt_text=True,
                created_at_unix_seconds=1.0,
            )
        )

        for state, expected in (
            (AlignmentJobState.QUEUED, alignment_pb2.ALIGNMENT_JOB_STATE_QUEUED),
            (AlignmentJobState.RUNNING, alignment_pb2.ALIGNMENT_JOB_STATE_RUNNING),
            (AlignmentJobState.SUCCEEDED, alignment_pb2.ALIGNMENT_JOB_STATE_SUCCEEDED),
            (AlignmentJobState.FAILED, alignment_pb2.ALIGNMENT_JOB_STATE_FAILED),
        ):
            with self.subTest(state=state):
                record = AlignmentJobRecord(
                    job_id="align-state",
                    state=state,
                    audio_artifact_id="audio-1",
                    include_srt_text=True,
                    created_at_unix_seconds=1.0,
                    error_code="dictator.jobs.failed" if state is AlignmentJobState.FAILED else None,
                    error_message="broken" if state is AlignmentJobState.FAILED else None,
                )
                self.assertEqual(service._job_response(record).state, expected)

        with self.assertRaises(RpcAbort) as raised:
            service.GetAlignTranscriptJob(
                alignment_pb2.GetAlignTranscriptJobRequest(job_id=" "),
                FakeContext(),
            )
        self.assertEqual(raised.exception.status, grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(service_context.alignment_job_manager.lookups, [])


if __name__ == "__main__":
    unittest.main()
