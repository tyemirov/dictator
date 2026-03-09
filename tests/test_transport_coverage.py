from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import Mock, patch

import grpc

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("librosa", types.SimpleNamespace())

from dictator.alignment.models import AlignTranscriptResult, AlignedWord
from dictator.diarization.models import (
    DiarizeAudioResult,
    DiarizedUtterance,
    DiarizedWord,
    SpeakerSegment,
    SpeakerSummary,
)
from dictator.extraction.models import ReferenceExtractionResult
from dictator.runtime import (
    DependencyError,
    InflightLimiter,
    MetricsRegistry,
    ProcessingError,
    ServiceRequestError,
    ValidationError,
)
from dictator.speech.v1 import alignment_pb2, artifacts_pb2, common_pb2, subtitle_pb2, transcription_pb2, voice_pb2
from dictator.storage import LocalArtifactStore
from dictator.subtitles.models import RenderSubtitlesResult, SubtitleCue
from dictator.synthesis.models import SpeechSegment, SynthesisResult
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


class AbortCalled(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class FakeContext:
    def __init__(self, *, active=True, remaining=10.0, metadata=()):
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
        raise AbortCalled(status, message)


class FakeTranscriptionService:
    def transcribe(self, audio, language=None, model_size="base", model=None):
        return TranscriptionResult(
            language=language or "en",
            words=(WordSegment("hello", 0.0, 0.4),),
        )


class FakeDiarizationService:
    def __init__(self):
        self.calls = []

    def diarize(self, request, model=None, diarization_pipeline=None):
        self.calls.append((request, model, diarization_pipeline))
        words = (DiarizedWord("hello", 0.0, 0.4, "S1"),)
        utterances = (DiarizedUtterance("S1", 0.0, 0.4, "hello", words),)
        speakers = (SpeakerSummary("S1", 1, 1, 1.0),)
        segments = (SpeakerSegment("S1", 0.0, 1.0, raw_label="speaker_a"),)
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
        srt_text = "1\n00:00:00,000 --> 00:00:00,400\nhello\n"
        if request.output_srt_path is not None:
            request.output_srt_path.write_text(srt_text, encoding="utf-8")
        return AlignTranscriptResult(
            audio_path=request.audio_path,
            language=request.language or "en",
            words=(AlignedWord("hello", 0.0, 0.4),),
            srt_text=srt_text,
            output_srt_path=request.output_srt_path,
        )


class FakeSubtitleService:
    def __init__(self):
        self.calls = []

    def render(self, request, model=None):
        self.calls.append((request, model))
        srt_text = "1\n00:00:00,000 --> 00:00:00,400\nhello\n"
        if request.output_srt_path is not None:
            request.output_srt_path.write_text(srt_text, encoding="utf-8")
        return RenderSubtitlesResult(
            language=request.language or "en",
            mode="forced_alignment" if request.source_text else "transcription",
            output_format=request.output_format,
            granularity=request.granularity,
            group_size=request.group_size,
            cues=(SubtitleCue(1, "hello", 0.0, 0.4, 1),),
            srt_text=srt_text,
            output_srt_path=request.output_srt_path,
        )


class FakeExtractionService:
    def __init__(self):
        self.calls = []

    def extract(self, request, model=None, diarization_pipeline=None):
        self.calls.append((request, model, diarization_pipeline))
        request.output_path.write_bytes(b"wav")
        return ReferenceExtractionResult(
            raw_words=(),
            dominant_speaker_words=({"content": "hello"},),
            window_start_seconds=0.0,
            window_end_seconds=1.0,
            trim_start_seconds=0.1,
            trim_end_seconds=0.9,
            output_path=request.output_path,
        )


class FakeSynthesisService:
    def __init__(self):
        self.calls = []

    def synthesise(self, speaker_wav, chunks, cap_seconds, language_code):
        self.calls.append((speaker_wav, chunks, cap_seconds, language_code))
        tmpdir = Path(tempfile.mkdtemp())
        wav_path = tmpdir / "0000.wav"
        wav_path.write_bytes(b"wav")
        return SynthesisResult(
            temp_dir=tmpdir,
            wav_paths=(wav_path,),
            segments=(SpeechSegment("hello", 0.0, 0.4),),
        )

    def synthesise_text(self, request):
        self.calls.append(request)
        tmpdir = Path(tempfile.mkdtemp())
        wav_path = tmpdir / "0000.wav"
        wav_path.write_bytes(b"wav")
        return SynthesisResult(
            temp_dir=tmpdir,
            wav_paths=(wav_path,),
            segments=(SpeechSegment("hello", 0.0, 0.4),),
        )


class FakeRuntime:
    def __init__(self):
        self.transcription = FakeTranscriptionService()
        self.diarization = FakeDiarizationService()
        self.alignment = FakeAlignmentService()
        self.subtitle = FakeSubtitleService()
        self.extraction = FakeExtractionService()
        self.synthesis = FakeSynthesisService()
        self.whisper_model_calls = []

    def get_transcription_service(self):
        return self.transcription

    def get_diarization_service(self):
        return self.diarization

    def get_alignment_service(self):
        return self.alignment

    def get_subtitle_service(self):
        return self.subtitle

    def get_reference_extraction_service(self):
        return self.extraction

    def get_synthesis_service(self):
        return self.synthesis

    def get_whisper_model(self, model_size):
        self.whisper_model_calls.append(model_size)
        return f"model:{model_size}"

    def get_diarization_pipeline(self):
        return "pipeline"


@contextmanager
def raising_limiter(exc):
    raise exc
    yield


class TransportCoverageTests(unittest.TestCase):
    def _build_context(self, tmpdir: str, auth_token: str | None = None):
        return ServiceContext(
            artifact_store=LocalArtifactStore(Path(tmpdir)),
            execution_runtime=FakeRuntime(),
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(4),
            auth_token=auth_token,
            download_chunk_bytes=2,
        )

    def test_base_servicer_helpers_and_scope_error_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service_context = self._build_context(tmpdir, auth_token="secret")
            servicer = BaseServicer(service_context)
            record = service_context.artifact_store.write_artifact([b"abc"], "audio.wav")
            artifact_ref = servicer._artifact_ref(record)
            self.assertEqual(artifact_ref.filename, "audio.wav")
            self.assertEqual(servicer._word_segment({"content": "hello", "start": 1.0}).content, "hello")
            self.assertEqual(servicer._timeline_segment({"content": "hi", "end": 2.0}).end_seconds, 2.0)
            cue = servicer._subtitle_cue(SubtitleCue(1, "hello", 0.0, 1.0, 1))
            self.assertEqual(cue.item_count, 1)
            self.assertEqual(servicer._resolve_language_request(language_code="en", autodetect_language=False, error_scope="scope"), "en")
            self.assertIsNone(servicer._resolve_language_request(language_code="", autodetect_language=True, error_scope="scope"))
            with self.assertRaises(ValidationError):
                servicer._resolve_language_request(language_code="en", autodetect_language=True, error_scope="scope")
            with self.assertRaises(ValidationError):
                servicer._resolve_language_request(language_code="", autodetect_language=False, error_scope="scope")

            inactive = FakeContext(active=False, metadata=(("x-dictator-token", "secret"),))
            with self.assertRaises(AbortCalled) as exc:
                servicer._ensure_request_active(inactive)
            self.assertEqual(exc.exception.status, grpc.StatusCode.CANCELLED)

            expired = FakeContext(active=True, remaining=0.0, metadata=(("x-dictator-token", "secret"),))
            with self.assertRaises(AbortCalled) as exc:
                servicer._ensure_request_active(expired)
            self.assertEqual(exc.exception.status, grpc.StatusCode.DEADLINE_EXCEEDED)

            with self.assertRaises(AbortCalled):
                servicer._require_auth(FakeContext(metadata=()))
            servicer._require_auth(FakeContext(metadata=(("authorization", "Bearer secret"),)))

            no_auth_context = self._build_context(tmpdir, auth_token=None)
            BaseServicer(no_auth_context)._require_auth(FakeContext())

            cases = [
                (ValidationError("code.validation", "bad"), grpc.StatusCode.INVALID_ARGUMENT),
                (DependencyError("code.dependency", "bad"), grpc.StatusCode.FAILED_PRECONDITION),
                (ProcessingError("code.processing", "bad"), grpc.StatusCode.INTERNAL),
                (FileNotFoundError("missing"), grpc.StatusCode.NOT_FOUND),
                (ValueError("bad value"), grpc.StatusCode.INVALID_ARGUMENT),
            ]
            for error, status in cases:
                context = FakeContext(metadata=(("x-dictator-token", "secret"),))
                with self.assertRaises(AbortCalled) as exc:
                    with servicer._request_scope(context, bytes_received=3):
                        raise error
                self.assertEqual(exc.exception.status, status)
            self.assertEqual(service_context.metrics.snapshot().bytes_received, 15)

            limiter_context = ServiceContext(
                artifact_store=service_context.artifact_store,
                execution_runtime=service_context.execution_runtime,
                metrics=MetricsRegistry(),
                limiter=types.SimpleNamespace(acquire=lambda: raising_limiter(ServiceRequestError(None, "limit", "too many"))),
                auth_token="secret",
                download_chunk_bytes=2,
            )
            with self.assertRaises(AbortCalled) as exc:
                with BaseServicer(limiter_context)._request_scope(FakeContext(metadata=(("x-dictator-token", "secret"),))):
                    pass
            self.assertEqual(exc.exception.status, grpc.StatusCode.RESOURCE_EXHAUSTED)

    def test_artifact_service_validation_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            servicer = ArtifactServiceServicer(self._build_context(tmpdir))
            context = FakeContext()
            with self.assertRaises(AbortCalled):
                servicer.UploadArtifact(iter(()), context)

            with self.assertRaises(AbortCalled):
                servicer.UploadArtifact(iter([artifacts_pb2.UploadArtifactChunk(content=b"x")]), FakeContext())

            chunks = iter([
                artifacts_pb2.UploadArtifactChunk(metadata=artifacts_pb2.UploadArtifactMetadata(filename="x.wav")),
                artifacts_pb2.UploadArtifactChunk(metadata=artifacts_pb2.UploadArtifactMetadata(filename="y.wav")),
            ])
            with self.assertRaises(AbortCalled):
                servicer.UploadArtifact(chunks, FakeContext())

            artifact_id = servicer.service_context.artifact_store.write_artifact([b"abc"], "x.wav").artifact_id
            with self.assertRaises(AbortCalled):
                list(servicer.DownloadArtifact(artifacts_pb2.DownloadArtifactRequest(artifact_id=artifact_id, chunk_size=-1), FakeContext()))

    def test_transcription_alignment_subtitle_and_voice_servicers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service_context = self._build_context(tmpdir)
            audio_record = service_context.artifact_store.write_artifact([b"audio"], "sample.wav", media_type="audio/wav")
            transcript_record = service_context.artifact_store.write_artifact([b"hello"], "transcript.txt", media_type="text/plain")
            speaker_record = service_context.artifact_store.write_artifact([b"speaker"], "speaker.wav", media_type="audio/wav")

            transcription_servicer = TranscriptionServiceServicer(service_context)
            diarize_response = transcription_servicer.DiarizeAudio(
                transcription_pb2.DiarizeAudioRequest(
                    audio_artifact_id=audio_record.artifact_id,
                    autodetect_language=True,
                    persist_json_artifact=True,
                ),
                FakeContext(),
            )
            diarization_payload = dict(diarize_response.diarization)
            self.assertIn("words", diarization_payload)
            self.assertIn("utterances", diarization_payload)
            self.assertIn("speakers", diarization_payload)
            self.assertNotIn("speakerSegments", diarization_payload)
            self.assertTrue(diarize_response.diarization_artifact_id)

            alignment_servicer = AlignmentServiceServicer(service_context)
            with self.assertRaises(AbortCalled):
                alignment_servicer.AlignTranscript(
                    alignment_pb2.AlignTranscriptRequest(audio_artifact_id=audio_record.artifact_id),
                    FakeContext(),
                )
            alignment_response = alignment_servicer.AlignTranscript(
                alignment_pb2.AlignTranscriptRequest(
                    audio_artifact_id=audio_record.artifact_id,
                    transcript_artifact_id=transcript_record.artifact_id,
                    include_srt_text=True,
                ),
                FakeContext(),
            )
            self.assertEqual(alignment_response.words[0].content, "hello")
            self.assertTrue(alignment_response.srt_artifact_id)
            self.assertTrue(alignment_response.srt_text)

            subtitle_servicer = SubtitleServiceServicer(service_context)
            with self.assertRaises(AbortCalled):
                subtitle_servicer.RenderSubtitles(
                    subtitle_pb2.RenderSubtitlesRequest(
                        audio_artifact_id=audio_record.artifact_id,
                        autodetect_language=True,
                        output_format=99,
                        granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
                        group_size=1,
                    ),
                    FakeContext(),
                )
            with self.assertRaises(AbortCalled):
                subtitle_servicer.RenderSubtitles(
                    subtitle_pb2.RenderSubtitlesRequest(
                        audio_artifact_id=audio_record.artifact_id,
                        autodetect_language=True,
                        output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                        granularity=subtitle_pb2.SUBTITLE_GRANULARITY_UNSPECIFIED,
                        group_size=1,
                    ),
                    FakeContext(),
                )
            with self.assertRaises(AbortCalled):
                subtitle_servicer.RenderSubtitles(
                    subtitle_pb2.RenderSubtitlesRequest(
                        audio_artifact_id=audio_record.artifact_id,
                        autodetect_language=True,
                        output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                        granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
                        group_size=0,
                    ),
                    FakeContext(),
                )
            subtitle_response = subtitle_servicer.RenderSubtitles(
                subtitle_pb2.RenderSubtitlesRequest(
                    audio_artifact_id=audio_record.artifact_id,
                    language_code="en",
                    output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                    granularity=subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES,
                    group_size=1,
                    source_text_artifact_id=transcript_record.artifact_id,
                ),
                FakeContext(),
            )
            self.assertEqual(subtitle_response.mode, subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT)
            self.assertEqual(service_context.execution_runtime.whisper_model_calls, ["base"])

            voice_servicer = VoiceServiceServicer(service_context)
            extract_response = voice_servicer.ExtractReferenceSample(
                voice_pb2.ExtractReferenceSampleRequest(source_artifact_id=audio_record.artifact_id),
                FakeContext(),
            )
            self.assertEqual(extract_response.dominant_speaker_word_count, 1)
            self.assertEqual(extract_response.sample_artifact.filename, "sample_reference.wav")

            with self.assertRaises(AbortCalled):
                voice_servicer.SynthesizeSpeech(
                    voice_pb2.SynthesizeSpeechRequest(
                        speaker_artifact_id=speaker_record.artifact_id,
                        synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_XTTS,
                    ),
                    FakeContext(),
                )

            with (
                patch("dictator.audio.ffmpeg_ops.concat_normalise", side_effect=lambda inputs, dst, cap: dst.write_bytes(b"wav")),
                patch("dictator.synthesis.service.cleanup_synthesis_result") as cleanup,
            ):
                synth_response = voice_servicer.SynthesizeSpeech(
                    voice_pb2.SynthesizeSpeechRequest(
                        speaker_artifact_id=speaker_record.artifact_id,
                        text_artifact_id=transcript_record.artifact_id,
                        include_timeline=True,
                        synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_XTTS,
                    ),
                    FakeContext(),
                )
            self.assertEqual(synth_response.chunk_count, 1)
            self.assertEqual(synth_response.timeline[0].content, "hello")
            self.assertTrue(synth_response.timeline_artifact_id)
            cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
