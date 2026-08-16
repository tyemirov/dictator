from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import grpc
from google.protobuf import struct_pb2

from dictator.client.alignment import AlignmentClient, AlignmentJob, AlignmentResult
from dictator.client.diarization import DiarizationClient, DiarizationJob, DiarizationResult
from dictator.client.dictation import DictationClient, DictationJob, DictationResult
from dictator.client.subtitles import SubtitleClient, SubtitleJob, SubtitleResult
from dictator.client.voice import ReferenceSampleClient, ReferenceSampleJob, ReferenceSampleResult
from dictator.runtime.jobs import (
    AlignmentJobRecord,
    AlignmentJobState,
    DiarizationJobRecord,
    DiarizationJobState,
    ExtractReferenceSampleJobRecord,
    ExtractReferenceSampleJobState,
    SubtitleJobRecord,
    SubtitleJobState,
    TranscriptionJobRecord,
    TranscriptionJobState,
)
from dictator.speech.v1 import alignment_pb2, subtitle_pb2, transcription_pb2, voice_pb2
from dictator.transport.grpc.alignment_service import AlignmentServiceServicer
from dictator.transport.grpc.subtitle_service import SubtitleServiceServicer
from dictator.transport.grpc.transcription_service import TranscriptionServiceServicer
from dictator.transport.grpc.voice_service import VoiceServiceServicer


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status_code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._status_code = status_code
        self._details = details

    def code(self):
        return self._status_code

    def details(self):
        return self._details


class AsyncClientRegressionTests(unittest.TestCase):
    def test_wait_for_job_has_no_deadline_by_default(self):
        from dictator.client._jobs import wait_for_job

        queued = types.SimpleNamespace(job_id="job-1", state="TRANSCRIPTION_JOB_STATE_QUEUED")
        succeeded = types.SimpleNamespace(job_id="job-1", state="TRANSCRIPTION_JOB_STATE_SUCCEEDED")
        responses = iter((queued, succeeded))

        with (
            patch("dictator.client._jobs.time.monotonic", side_effect=AssertionError("deadline should be disabled by default")),
            patch("dictator.client._jobs.time.sleep"),
        ):
            result = wait_for_job(lambda: next(responses))

        self.assertEqual(result.state, "TRANSCRIPTION_JOB_STATE_SUCCEEDED")

    def test_blocking_convenience_methods_pass_timeout_none_by_default(self):
        dictation_client = self._make_dictation_client(
            stub=types.SimpleNamespace(
                SubmitTranscribeJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="tx-1",
                        state=transcription_pb2.TRANSCRIPTION_JOB_STATE_QUEUED,
                    )
                )
            )
        )
        with (
            patch("dictator.client.dictation.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-1")),
            patch.object(
                dictation_client,
                "wait_for_dictation_job",
                return_value=DictationJob(
                    job_id="tx-1",
                    state="TRANSCRIPTION_JOB_STATE_SUCCEEDED",
                    source_artifact_id="audio-1",
                    result=DictationResult(text="hello", language_code="en", artifact_id="audio-1"),
                ),
            ) as wait_mock,
        ):
            dictation_client.dictate_bytes(b"abc")
        self.assertIsNone(wait_mock.call_args.kwargs["timeout_seconds"])

        diarization_client = self._make_diarization_client(
            stub=types.SimpleNamespace(
                SubmitDiarizeAudioJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="dia-1",
                        state=transcription_pb2.DIARIZATION_JOB_STATE_QUEUED,
                    )
                )
            )
        )
        with (
            patch("dictator.client.diarization.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-1")),
            patch.object(
                diarization_client,
                "wait_for_diarization_job",
                return_value=DiarizationJob(
                    job_id="dia-1",
                    state="DIARIZATION_JOB_STATE_SUCCEEDED",
                    source_artifact_id="audio-1",
                    result=DiarizationResult(
                        text="hello",
                        language_code="en",
                        source_artifact_id="audio-1",
                        diarization={"text": "hello"},
                    ),
                ),
            ) as wait_mock,
        ):
            diarization_client.diarize_bytes(b"abc")
        self.assertIsNone(wait_mock.call_args.kwargs["timeout_seconds"])

        subtitle_client = self._make_subtitle_client(
            stub=types.SimpleNamespace(
                SubmitRenderSubtitlesJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="sub-1",
                        state=subtitle_pb2.SUBTITLE_JOB_STATE_QUEUED,
                    )
                )
            )
        )
        with (
            patch("dictator.client.subtitles.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-1")),
            patch.object(
                subtitle_client,
                "wait_for_subtitle_job",
                return_value=SubtitleJob(
                    job_id="sub-1",
                    state="SUBTITLE_JOB_STATE_SUCCEEDED",
                    source_artifact_id="audio-1",
                    result=SubtitleResult(
                        language_code="en",
                        mode="forced_alignment",
                        granularity="words",
                        group_size=1,
                        source_artifact_id="audio-1",
                        srt_artifact_id="srt-1",
                        srt_text="",
                        cues=(),
                    ),
                ),
            ) as wait_mock,
        ):
            subtitle_client.render_bytes(b"abc", source_text="hello")
        self.assertIsNone(wait_mock.call_args.kwargs["timeout_seconds"])

        alignment_client = self._make_alignment_client(
            stub=types.SimpleNamespace(
                SubmitAlignTranscriptJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="align-1",
                        state=alignment_pb2.ALIGNMENT_JOB_STATE_QUEUED,
                    )
                )
            )
        )
        with (
            patch("dictator.client.alignment.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-1")),
            patch.object(
                alignment_client,
                "wait_for_alignment_job",
                return_value=AlignmentJob(
                    job_id="align-1",
                    state="ALIGNMENT_JOB_STATE_SUCCEEDED",
                    source_artifact_id="audio-1",
                    result=AlignmentResult(
                        language_code="en",
                        source_artifact_id="audio-1",
                        srt_artifact_id="srt-1",
                        srt_text="",
                        words=(),
                    ),
                ),
            ) as wait_mock,
        ):
            alignment_client.align_bytes(b"abc", transcript_text="hello")
        self.assertIsNone(wait_mock.call_args.kwargs["timeout_seconds"])

        reference_client = self._make_reference_client()
        with (
            patch.object(
                reference_client,
                "submit_extract_bytes_job",
                return_value=ReferenceSampleJob(job_id="ref-1", state="EXTRACT_REFERENCE_SAMPLE_JOB_STATE_QUEUED", source_artifact_id="audio-1"),
            ),
            patch.object(
                reference_client,
                "wait_for_reference_sample_job",
                return_value=ReferenceSampleJob(
                    job_id="ref-1",
                    state="EXTRACT_REFERENCE_SAMPLE_JOB_STATE_SUCCEEDED",
                    source_artifact_id="audio-1",
                    result=ReferenceSampleResult(
                        sample_artifact_id="sample-1",
                        trim_start_seconds=0.0,
                        trim_end_seconds=1.0,
                        window_start_seconds=0.0,
                        window_end_seconds=1.2,
                        dominant_speaker_word_count=3,
                    ),
                ),
            ) as wait_mock,
        ):
            reference_client.extract_bytes(b"abc")
        self.assertIsNone(wait_mock.call_args.kwargs["timeout_seconds"])

    def test_get_job_helpers_preserve_source_artifact_ids(self):
        dictation_client = self._make_dictation_client(
            stub=types.SimpleNamespace(
                GetTranscribeJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="tx-1",
                        state=transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED,
                        source_artifact_id="audio-1",
                        error_code="",
                        error_message="",
                        text="hello",
                        language_code="en",
                        words=(),
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    )
                )
            )
        )
        dictation_job = dictation_client.get_dictation_job("tx-1")
        self.assertEqual(dictation_job.source_artifact_id, "audio-1")
        self.assertEqual(dictation_job.result.artifact_id, "audio-1")

        diarization_struct = struct_pb2.Struct()
        diarization_struct.update({"text": "hello"})
        diarization_client = self._make_diarization_client(
            stub=types.SimpleNamespace(
                GetDiarizeAudioJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="dia-1",
                        state=transcription_pb2.DIARIZATION_JOB_STATE_SUCCEEDED,
                        source_artifact_id="audio-2",
                        error_code="",
                        error_message="",
                        text="hello",
                        language_code="en",
                        diarization=diarization_struct,
                        diarization_artifact_id="json-1",
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    )
                )
            )
        )
        diarization_job = diarization_client.get_diarization_job("dia-1")
        self.assertEqual(diarization_job.source_artifact_id, "audio-2")
        self.assertEqual(diarization_job.result.source_artifact_id, "audio-2")

        subtitle_client = self._make_subtitle_client(
            stub=types.SimpleNamespace(
                GetRenderSubtitlesJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="sub-1",
                        state=subtitle_pb2.SUBTITLE_JOB_STATE_SUCCEEDED,
                        source_artifact_id="audio-3",
                        error_code="",
                        error_message="",
                        language_code="en",
                        mode=subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT,
                        output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                        granularity=subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
                        group_size=1,
                        cues=(),
                        srt_text="",
                        srt_artifact_id="srt-1",
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    )
                )
            )
        )
        subtitle_job = subtitle_client.get_subtitle_job("sub-1")
        self.assertEqual(subtitle_job.source_artifact_id, "audio-3")
        self.assertEqual(subtitle_job.result.source_artifact_id, "audio-3")

        alignment_client = self._make_alignment_client(
            stub=types.SimpleNamespace(
                GetAlignTranscriptJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="align-1",
                        state=alignment_pb2.ALIGNMENT_JOB_STATE_SUCCEEDED,
                        source_artifact_id="audio-4",
                        error_code="",
                        error_message="",
                        language_code="en",
                        words=(),
                        srt_text="",
                        srt_artifact_id="srt-2",
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    )
                )
            )
        )
        alignment_job = alignment_client.get_alignment_job("align-1")
        self.assertEqual(alignment_job.source_artifact_id, "audio-4")
        self.assertEqual(alignment_job.result.source_artifact_id, "audio-4")

        reference_client = self._make_reference_client(
            stub=types.SimpleNamespace(
                GetExtractReferenceSampleJob=MagicMock(
                    return_value=types.SimpleNamespace(
                        job_id="ref-1",
                        state=voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_SUCCEEDED,
                        source_artifact_id="audio-5",
                        error_code="",
                        error_message="",
                        sample_artifact=types.SimpleNamespace(artifact_id="sample-1"),
                        trim_start_seconds=0.0,
                        trim_end_seconds=1.0,
                        window_start_seconds=0.0,
                        window_end_seconds=1.2,
                        dominant_speaker_word_count=3,
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    )
                )
            )
        )
        reference_job = reference_client.get_reference_sample_job("ref-1")
        self.assertEqual(reference_job.source_artifact_id, "audio-5")

    def test_sync_fallback_reuses_uploaded_audio(self):
        artifact = types.SimpleNamespace(artifact_id="audio-1")

        dictation_stub = types.SimpleNamespace(
            SubmitTranscribeJob=MagicMock(
                side_effect=_FakeRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "transcription job manager is not configured",
                )
            ),
            Transcribe=MagicMock(
                return_value=types.SimpleNamespace(text="hello", language_code="en", words=())
            ),
        )
        with patch("dictator.client.dictation.upload_audio_artifact", return_value=artifact) as upload_mock:
            client = self._make_dictation_client(stub=dictation_stub)
            result = client.dictate_bytes(b"abc", language_code="en")
        self.assertEqual(result.artifact_id, "audio-1")
        self.assertEqual(upload_mock.call_count, 1)
        self.assertEqual(dictation_stub.Transcribe.call_args.args[0].audio_artifact_id, "audio-1")

        diarization_stub = types.SimpleNamespace(
            SubmitDiarizeAudioJob=MagicMock(
                side_effect=_FakeRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "diarization job manager is not configured",
                )
            ),
            DiarizeAudio=MagicMock(),
        )
        with patch("dictator.client.diarization.upload_audio_artifact", return_value=artifact) as upload_mock:
            client = self._make_diarization_client(stub=diarization_stub)
            with self.assertRaises(grpc.RpcError):
                client.diarize_bytes(b"abc", language_code="en")
        self.assertEqual(upload_mock.call_count, 1)
        diarization_stub.DiarizeAudio.assert_not_called()

        subtitle_stub = types.SimpleNamespace(
            SubmitRenderSubtitlesJob=MagicMock(
                side_effect=_FakeRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "subtitle job manager is not configured",
                )
            ),
            RenderSubtitles=MagicMock(
                return_value=types.SimpleNamespace(
                    language_code="en",
                    mode=subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT,
                    group_size=1,
                    cues=(),
                    srt_text="",
                    srt_artifact_id="srt-1",
                )
            ),
        )
        with patch("dictator.client.subtitles.upload_audio_artifact", return_value=artifact) as upload_mock:
            client = self._make_subtitle_client(stub=subtitle_stub)
            result = client.render_bytes(b"abc", language_code="en", source_text="hello")
        self.assertEqual(result.source_artifact_id, "audio-1")
        self.assertEqual(upload_mock.call_count, 1)
        self.assertEqual(subtitle_stub.RenderSubtitles.call_args.args[0].audio_artifact_id, "audio-1")

        alignment_stub = types.SimpleNamespace(
            SubmitAlignTranscriptJob=MagicMock(
                side_effect=_FakeRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "alignment job manager is not configured",
                )
            ),
            AlignTranscript=MagicMock(
                return_value=types.SimpleNamespace(
                    language_code="en",
                    words=(),
                    srt_text="",
                    srt_artifact_id="srt-1",
                )
            ),
        )
        with patch("dictator.client.alignment.upload_audio_artifact", return_value=artifact) as upload_mock:
            client = self._make_alignment_client(stub=alignment_stub)
            result = client.align_bytes(b"abc", transcript_text="hello")
        self.assertEqual(result.source_artifact_id, "audio-1")
        self.assertEqual(upload_mock.call_count, 1)
        self.assertEqual(alignment_stub.AlignTranscript.call_args.args[0].audio_artifact_id, "audio-1")

    def test_job_response_builders_include_source_artifact_ids(self):
        alignment_servicer = AlignmentServiceServicer(types.SimpleNamespace())
        alignment_response = alignment_servicer._job_response(
            AlignmentJobRecord(
                job_id="align-1",
                state=AlignmentJobState.SUCCEEDED,
                audio_artifact_id="audio-1",
                include_srt_text=True,
                created_at_unix_seconds=1.0,
            )
        )
        self.assertEqual(alignment_response.source_artifact_id, "audio-1")

        transcription_servicer = TranscriptionServiceServicer(types.SimpleNamespace())
        transcription_response = transcription_servicer._transcription_job_response(
            TranscriptionJobRecord(
                job_id="tx-1",
                state=TranscriptionJobState.SUCCEEDED,
                audio_artifact_id="audio-2",
                include_word_segments=True,
                created_at_unix_seconds=1.0,
            )
        )
        self.assertEqual(transcription_response.source_artifact_id, "audio-2")

        diarization_response = transcription_servicer._diarization_job_response(
            DiarizationJobRecord(
                job_id="dia-1",
                state=DiarizationJobState.SUCCEEDED,
                audio_artifact_id="audio-3",
                include_words=True,
                include_utterances=True,
                include_speakers=True,
                include_speaker_segments=True,
                persist_json_artifact=False,
                created_at_unix_seconds=1.0,
            )
        )
        self.assertEqual(diarization_response.source_artifact_id, "audio-3")

        subtitle_servicer = SubtitleServiceServicer(types.SimpleNamespace())
        subtitle_response = subtitle_servicer._subtitle_job_response(
            SubtitleJobRecord(
                job_id="sub-1",
                state=SubtitleJobState.SUCCEEDED,
                audio_artifact_id="audio-4",
                include_srt_text=True,
                created_at_unix_seconds=1.0,
            )
        )
        self.assertEqual(subtitle_response.source_artifact_id, "audio-4")

        voice_servicer = VoiceServiceServicer(types.SimpleNamespace())
        reference_response = voice_servicer._reference_extraction_job_response(
            ExtractReferenceSampleJobRecord(
                job_id="ref-1",
                state=ExtractReferenceSampleJobState.SUCCEEDED,
                source_artifact_id="audio-5",
                created_at_unix_seconds=1.0,
            )
        )
        self.assertEqual(reference_response.source_artifact_id, "audio-5")

    @staticmethod
    def _make_dictation_client(stub=None) -> DictationClient:
        with (
            patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub", return_value=stub or object()),
        ):
            return DictationClient(object())

    @staticmethod
    def _make_diarization_client(stub=None) -> DiarizationClient:
        with (
            patch("dictator.client.diarization.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.diarization.transcription_pb2_grpc.TranscriptionServiceStub", return_value=stub or object()),
        ):
            return DiarizationClient(object())

    @staticmethod
    def _make_subtitle_client(stub=None) -> SubtitleClient:
        with (
            patch("dictator.client.subtitles.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.subtitles.subtitle_pb2_grpc.SubtitleServiceStub", return_value=stub or object()),
        ):
            return SubtitleClient(object())

    @staticmethod
    def _make_alignment_client(stub=None) -> AlignmentClient:
        with (
            patch("dictator.client.alignment.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.alignment.alignment_pb2_grpc.AlignmentServiceStub", return_value=stub or object()),
        ):
            return AlignmentClient(object())

    @staticmethod
    def _make_reference_client(stub=None) -> ReferenceSampleClient:
        with (
            patch("dictator.client.voice.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=stub or object()),
        ):
            return ReferenceSampleClient(object())


if __name__ == "__main__":
    unittest.main()
