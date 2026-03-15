"""Async job API coverage for slow audio endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import grpc

from dictator.diarization.models import (
    DiarizeAudioResult,
    DiarizedUtterance,
    DiarizedWord,
    SpeakerSegment,
    SpeakerSummary,
)
from dictator.runtime import InflightLimiter, MetricsRegistry, ValidationError
from dictator.runtime.jobs import (
    DiarizationJobRecord,
    DiarizationJobState,
    ExtractReferenceSampleJobRecord,
    ExtractReferenceSampleJobState,
    SubtitleJobRecord,
    SubtitleJobState,
    TranscriptionJobRecord,
    TranscriptionJobState,
)
from dictator.storage import LocalArtifactStore
from dictator.subtitles.models import RenderSubtitlesResult, SubtitleCue
from dictator.speech.v1 import subtitle_pb2, transcription_pb2, voice_pb2
from dictator.transcription.models import TranscriptionResult, WordSegment
from dictator.transport.grpc.context import ServiceContext
from dictator.transport.grpc.subtitle_service import SubtitleServiceServicer
from dictator.transport.grpc.transcription_service import TranscriptionServiceServicer
from dictator.transport.grpc.voice_service import VoiceServiceServicer


class RpcAbort(Exception):
    def __init__(self, status, message, trailing_metadata):
        super().__init__(message)
        self.status = status
        self.message = message
        self.trailing_metadata = trailing_metadata


class FakeContext:
    def __init__(self, metadata=(("x-dictator-token", "secret"),)):
        self._metadata = metadata
        self._trailing_metadata = ()

    def invocation_metadata(self):
        return self._metadata

    def set_trailing_metadata(self, metadata):
        self._trailing_metadata = metadata

    def abort(self, status, message):
        raise RpcAbort(status, message, self._trailing_metadata)

    def is_active(self):
        return True

    def time_remaining(self):
        return 30.0


@dataclass
class _FakeJobManager:
    record: object

    def __post_init__(self):
        self.submitted = []
        self.lookups = []

    def submit(self, prepared):
        self.submitted.append(prepared)
        return self.record

    def get(self, job_id):
        self.lookups.append(job_id)
        return self.record


class _FakeRuntime:
    def get_transcription_service(self):
        raise AssertionError("not used in async submit/get tests")

    def get_whisper_model(self, model_size):
        raise AssertionError("not used in async submit/get tests")

    def get_diarization_service(self):
        raise AssertionError("not used in async submit/get tests")

    def get_diarization_pipeline(self):
        raise AssertionError("not used in async submit/get tests")

    def get_subtitle_service(self):
        raise AssertionError("not used in async submit/get tests")

    def get_reference_extraction_service(self):
        raise AssertionError("not used in async submit/get tests")


class AsyncJobEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.artifact_store = LocalArtifactStore(Path(self.tmpdir.name) / "artifacts")
        self.audio_record = self.artifact_store.write_artifact(
            [b"wav"],
            filename="sample.wav",
            media_type="audio/wav",
        )
        self.text_record = self.artifact_store.write_artifact(
            [b"hello world"],
            filename="sample.txt",
            media_type="text/plain",
            fallback_suffix=".txt",
        )
        self.base_context = dict(
            artifact_store=self.artifact_store,
            execution_runtime=_FakeRuntime(),
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(4),
            auth_token="secret",
            download_chunk_bytes=1024,
            synthesis_job_manager=None,
            alignment_job_manager=None,
        )

    def test_transcription_job_submit_and_lookup(self):
        manager = _FakeJobManager(
            TranscriptionJobRecord(
                job_id="tx-1",
                state=TranscriptionJobState.SUCCEEDED,
                audio_artifact_id=self.audio_record.artifact_id,
                include_word_segments=True,
                created_at_unix_seconds=1.0,
                started_at_unix_seconds=2.0,
                finished_at_unix_seconds=3.0,
                text="hello world",
                language_code="en",
                words=(
                    WordSegment(text="hello", start_seconds=0.0, end_seconds=0.5),
                    WordSegment(text="world", start_seconds=0.5, end_seconds=1.0),
                ),
            )
        )
        context = ServiceContext(
            **self.base_context,
            transcription_job_manager=manager,
        )
        servicer = TranscriptionServiceServicer(context)

        submitted = servicer.SubmitTranscribeJob(
            transcription_pb2.TranscribeRequest(
                audio_artifact_id=self.audio_record.artifact_id,
                language_code="en",
                model_size="base",
                include_word_segments=True,
            ),
            FakeContext(),
        )
        self.assertEqual(submitted.job_id, "tx-1")
        self.assertEqual(submitted.state, transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED)
        self.assertTrue(manager.submitted[0].include_word_segments)

        looked_up = servicer.GetTranscribeJob(
            transcription_pb2.GetTranscribeJobRequest(job_id="tx-1"),
            FakeContext(),
        )
        self.assertEqual(looked_up.text, "hello world")
        self.assertEqual([word.content for word in looked_up.words], ["hello", "world"])
        self.assertEqual(manager.lookups, ["tx-1"])

    def test_diarization_job_submit_and_lookup(self):
        diarized_word = DiarizedWord(text="hello", start_seconds=0.0, end_seconds=0.5, speaker="S1")
        diarized_utterance = DiarizedUtterance(
            speaker="S1",
            start_seconds=0.0,
            end_seconds=0.5,
            text="hello",
            words=(diarized_word,),
        )
        payload = DiarizeAudioResult(
            language="en",
            text="hello",
            words=(diarized_word,),
            utterances=(diarized_utterance,),
            speakers=(SpeakerSummary(speaker="S1", word_count=1, utterance_count=1, total_duration_seconds=0.5),),
            speaker_segments=(SpeakerSegment(speaker="S1", start_seconds=0.0, end_seconds=0.5),),
        ).to_json_dict(
            include_words=True,
            include_utterances=True,
            include_speakers=True,
            include_speaker_segments=True,
        )
        manager = _FakeJobManager(
            DiarizationJobRecord(
                job_id="dia-1",
                state=DiarizationJobState.SUCCEEDED,
                audio_artifact_id=self.audio_record.artifact_id,
                include_words=True,
                include_utterances=True,
                include_speakers=True,
                include_speaker_segments=True,
                persist_json_artifact=True,
                created_at_unix_seconds=1.0,
                started_at_unix_seconds=2.0,
                finished_at_unix_seconds=3.0,
                text="hello",
                language_code="en",
                diarization=payload,
                diarization_artifact_id="artifact-1",
            )
        )
        context = ServiceContext(
            **self.base_context,
            transcription_job_manager=_FakeJobManager(
                TranscriptionJobRecord(
                    job_id="unused",
                    state=TranscriptionJobState.QUEUED,
                    audio_artifact_id=self.audio_record.artifact_id,
                    include_word_segments=False,
                    created_at_unix_seconds=1.0,
                )
            ),
            diarization_job_manager=manager,
        )
        servicer = TranscriptionServiceServicer(context)

        submitted = servicer.SubmitDiarizeAudioJob(
            transcription_pb2.DiarizeAudioRequest(
                audio_artifact_id=self.audio_record.artifact_id,
                autodetect_language=True,
                include_words=True,
                include_utterances=True,
                include_speakers=True,
                include_speaker_segments=True,
                persist_json_artifact=True,
            ),
            FakeContext(),
        )
        self.assertEqual(submitted.job_id, "dia-1")
        self.assertEqual(submitted.state, transcription_pb2.DIARIZATION_JOB_STATE_SUCCEEDED)

        looked_up = servicer.GetDiarizeAudioJob(
            transcription_pb2.GetDiarizeAudioJobRequest(job_id="dia-1"),
            FakeContext(),
        )
        self.assertEqual(looked_up.text, "hello")
        self.assertEqual(looked_up.language_code, "en")
        self.assertEqual(looked_up.diarization_artifact_id, "artifact-1")
        self.assertIn("words", dict(looked_up.diarization))
        self.assertEqual(manager.lookups, ["dia-1"])

    def test_subtitle_job_submit_and_lookup(self):
        manager = _FakeJobManager(
            SubtitleJobRecord(
                job_id="sub-1",
                state=SubtitleJobState.SUCCEEDED,
                audio_artifact_id=self.audio_record.artifact_id,
                include_srt_text=True,
                created_at_unix_seconds=1.0,
                started_at_unix_seconds=2.0,
                finished_at_unix_seconds=3.0,
                language_code="en",
                mode="transcription",
                output_format="srt",
                granularity="words",
                group_size=1,
                cues=(SubtitleCue(index=1, text="hello", start_seconds=0.0, end_seconds=1.0, item_count=1),),
                srt_text="1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                srt_artifact_id="srt-1",
            )
        )
        context = ServiceContext(
            **self.base_context,
            subtitle_job_manager=manager,
        )
        servicer = SubtitleServiceServicer(context)

        submitted = servicer.SubmitRenderSubtitlesJob(
            subtitle_pb2.RenderSubtitlesRequest(
                audio_artifact_id=self.audio_record.artifact_id,
                language_code="en",
                output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
                group_size=1,
                include_srt_text=True,
            ),
            FakeContext(),
        )
        self.assertEqual(submitted.job_id, "sub-1")
        self.assertEqual(submitted.state, subtitle_pb2.SUBTITLE_JOB_STATE_SUCCEEDED)

        looked_up = servicer.GetRenderSubtitlesJob(
            subtitle_pb2.GetRenderSubtitlesJobRequest(job_id="sub-1"),
            FakeContext(),
        )
        self.assertEqual(looked_up.language_code, "en")
        self.assertEqual(looked_up.mode, subtitle_pb2.SUBTITLE_MODE_TRANSCRIPTION)
        self.assertEqual(looked_up.cues[0].content, "hello")
        self.assertEqual(looked_up.srt_artifact_id, "srt-1")
        self.assertEqual(manager.lookups, ["sub-1"])

    def test_reference_extraction_job_submit_and_lookup(self):
        sample_record = self.artifact_store.write_artifact(
            [b"sample"],
            filename="sample_reference.wav",
            media_type="audio/wav",
        )
        manager = _FakeJobManager(
            ExtractReferenceSampleJobRecord(
                job_id="ref-1",
                state=ExtractReferenceSampleJobState.SUCCEEDED,
                source_artifact_id=self.audio_record.artifact_id,
                created_at_unix_seconds=1.0,
                started_at_unix_seconds=2.0,
                finished_at_unix_seconds=3.0,
                sample_artifact_id=sample_record.artifact_id,
                trim_start_seconds=1.0,
                trim_end_seconds=3.0,
                window_start_seconds=0.0,
                window_end_seconds=4.0,
                dominant_speaker_word_count=12,
            )
        )
        context = ServiceContext(
            **self.base_context,
            reference_extraction_job_manager=manager,
        )
        servicer = VoiceServiceServicer(context)

        submitted = servicer.SubmitExtractReferenceSampleJob(
            voice_pb2.ExtractReferenceSampleRequest(source_artifact_id=self.audio_record.artifact_id),
            FakeContext(),
        )
        self.assertEqual(submitted.job_id, "ref-1")
        self.assertEqual(submitted.state, voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_SUCCEEDED)

        looked_up = servicer.GetExtractReferenceSampleJob(
            voice_pb2.GetExtractReferenceSampleJobRequest(job_id="ref-1"),
            FakeContext(),
        )
        self.assertEqual(looked_up.sample_artifact.artifact_id, sample_record.artifact_id)
        self.assertEqual(looked_up.trim_start_seconds, 1.0)
        self.assertEqual(looked_up.dominant_speaker_word_count, 12)
        self.assertEqual(manager.lookups, ["ref-1"])

    def test_async_job_apis_require_managers_and_valid_job_ids(self):
        context = ServiceContext(**self.base_context)
        transcriber = TranscriptionServiceServicer(context)
        subtitler = SubtitleServiceServicer(context)
        voicer = VoiceServiceServicer(context)

        with self.assertRaises(RpcAbort) as exc:
            transcriber.SubmitTranscribeJob(
                transcription_pb2.TranscribeRequest(
                    audio_artifact_id=self.audio_record.artifact_id,
                    language_code="en",
                ),
                FakeContext(),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        with self.assertRaises(RpcAbort) as exc:
            transcriber.GetTranscribeJob(
                transcription_pb2.GetTranscribeJobRequest(job_id=""),
                FakeContext(),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        with self.assertRaises(RpcAbort) as exc:
            subtitler.SubmitRenderSubtitlesJob(
                subtitle_pb2.RenderSubtitlesRequest(
                    audio_artifact_id=self.audio_record.artifact_id,
                    language_code="en",
                    output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                    granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
                    group_size=1,
                ),
                FakeContext(),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)

        with self.assertRaises(RpcAbort) as exc:
            voicer.GetExtractReferenceSampleJob(
                voice_pb2.GetExtractReferenceSampleJobRequest(job_id="../escape"),
                FakeContext(),
            )
        self.assertEqual(exc.exception.status, grpc.StatusCode.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
