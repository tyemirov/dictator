from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

from google.protobuf import struct_pb2

from dictator.client import (
    DiarizationClient,
    DictationClient,
    ReferenceSampleClient,
    ReferenceSampleResult,
    RemoteJobFailedError,
    SubtitleClient,
)
from dictator.speech.v1 import subtitle_pb2, transcription_pb2, voice_pb2


class ClientJobHelpersTests(unittest.TestCase):
    def test_dictation_job_helpers_submit_wait_and_failure(self):
        stub = types.SimpleNamespace(
            SubmitTranscribeJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="tx-1",
                    state=transcription_pb2.TRANSCRIPTION_JOB_STATE_QUEUED,
                )
            ),
            GetTranscribeJob=MagicMock(
                side_effect=[
                    types.SimpleNamespace(
                        job_id="tx-1",
                        state=transcription_pb2.TRANSCRIPTION_JOB_STATE_QUEUED,
                        error_code="",
                        error_message="",
                        text="",
                        language_code="",
                        words=[],
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=0.0,
                        finished_at_unix_seconds=0.0,
                    ),
                    types.SimpleNamespace(
                        job_id="tx-1",
                        state=transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED,
                        error_code="",
                        error_message="",
                        text="hello world",
                        language_code="en",
                        words=[
                            types.SimpleNamespace(content="hello", start_seconds=0.0, end_seconds=0.5),
                            types.SimpleNamespace(content="world", start_seconds=0.5, end_seconds=1.0),
                        ],
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    ),
                ]
            ),
        )
        with (
            patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub", return_value=stub),
            patch("dictator.client.dictation.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-1")),
            patch("dictator.client._jobs.time.sleep"),
        ):
            client = DictationClient(object())
            submitted = client.submit_dictate_bytes_job(b"abc", language_code="en")
            self.assertEqual(submitted.job_id, "tx-1")
            self.assertEqual(submitted.source_artifact_id, "audio-1")
            finished = client.wait_for_dictation_job("tx-1")
        self.assertEqual(finished.state, "TRANSCRIPTION_JOB_STATE_SUCCEEDED")
        self.assertEqual(finished.result.text, "hello world")
        self.assertEqual(finished.result.words[0]["content"], "hello")

        failing_stub = types.SimpleNamespace(
            SubmitTranscribeJob=MagicMock(),
            GetTranscribeJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="tx-2",
                    state=transcription_pb2.TRANSCRIPTION_JOB_STATE_FAILED,
                    error_code="dictator.jobs.failed",
                    error_message="broken",
                    text="",
                    language_code="",
                    words=[],
                    created_at_unix_seconds=1.0,
                    started_at_unix_seconds=2.0,
                    finished_at_unix_seconds=3.0,
                )
            ),
        )
        with (
            patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub", return_value=failing_stub),
        ):
            client = DictationClient(object())
            with self.assertRaisesRegex(RemoteJobFailedError, "dictator.jobs.failed"):
                client.wait_for_dictation_job("tx-2", poll_interval_seconds=0.01)

    def test_diarization_job_helpers_submit_and_wait(self):
        diarization_struct = struct_pb2.Struct()
        diarization_struct.update({"text": "hello", "speakers": [{"speaker": "S1"}]})
        stub = types.SimpleNamespace(
            SubmitDiarizeAudioJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="dia-1",
                    state=transcription_pb2.DIARIZATION_JOB_STATE_QUEUED,
                )
            ),
            GetDiarizeAudioJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="dia-1",
                    state=transcription_pb2.DIARIZATION_JOB_STATE_SUCCEEDED,
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
            ),
        )
        with (
            patch("dictator.client.diarization.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.diarization.transcription_pb2_grpc.TranscriptionServiceStub", return_value=stub),
            patch("dictator.client.diarization.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-2")),
        ):
            client = DiarizationClient(object())
            submitted = client.submit_diarize_bytes_job(b"abc", autodetect_language=True, persist_json_artifact=True)
            finished = client.wait_for_diarization_job("dia-1", poll_interval_seconds=0.01)
        self.assertEqual(submitted.source_artifact_id, "audio-2")
        self.assertEqual(finished.result.diarization_artifact_id, "json-1")
        self.assertEqual(finished.result.diarization["text"], "hello")

    def test_subtitle_job_helpers_submit_and_wait(self):
        stub = types.SimpleNamespace(
            SubmitRenderSubtitlesJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="sub-1",
                    state=subtitle_pb2.SUBTITLE_JOB_STATE_QUEUED,
                )
            ),
            GetRenderSubtitlesJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="sub-1",
                    state=subtitle_pb2.SUBTITLE_JOB_STATE_SUCCEEDED,
                    error_code="",
                    error_message="",
                    language_code="en",
                    mode=subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT,
                    output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                    granularity=subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES,
                    group_size=2,
                    cues=[types.SimpleNamespace(content="hello world", start_seconds=0.0, end_seconds=0.5, item_count=2)],
                    srt_text="1\n00:00:00,000 --> 00:00:00,500\nhello world\n",
                    srt_artifact_id="srt-1",
                    created_at_unix_seconds=1.0,
                    started_at_unix_seconds=2.0,
                    finished_at_unix_seconds=3.0,
                )
            ),
        )
        with (
            patch("dictator.client.subtitles.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.subtitles.subtitle_pb2_grpc.SubtitleServiceStub", return_value=stub),
            patch("dictator.client.subtitles.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-3")),
        ):
            client = SubtitleClient(object())
            submitted = client.submit_render_bytes_job(
                b"abc",
                language_code="en",
                granularity="sentences",
                group_size=2,
                source_text="hello world",
            )
            finished = client.wait_for_subtitle_job("sub-1", poll_interval_seconds=0.01)
        self.assertEqual(submitted.source_artifact_id, "audio-3")
        self.assertEqual(finished.result.mode, "forced_alignment")
        self.assertEqual(finished.result.granularity, "sentences")

    def test_reference_sample_job_helpers_submit_and_wait(self):
        stub = types.SimpleNamespace(
            SubmitExtractReferenceSampleJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="ref-1",
                    state=voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_QUEUED,
                )
            ),
            GetExtractReferenceSampleJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="ref-1",
                    state=voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_SUCCEEDED,
                    error_code="",
                    error_message="",
                    sample_artifact=types.SimpleNamespace(artifact_id="sample-1"),
                    trim_start_seconds=1.0,
                    trim_end_seconds=3.0,
                    window_start_seconds=0.5,
                    window_end_seconds=3.5,
                    dominant_speaker_word_count=12,
                    created_at_unix_seconds=1.0,
                    started_at_unix_seconds=2.0,
                    finished_at_unix_seconds=3.0,
                )
            ),
        )
        with (
            patch("dictator.client.voice.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=stub),
            patch("dictator.client.voice.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-4")),
        ):
            client = ReferenceSampleClient(object())
            submitted = client.submit_extract_bytes_job(b"abc", language_code="en")
            finished = client.wait_for_reference_sample_job("ref-1", poll_interval_seconds=0.01)
        self.assertEqual(submitted.source_artifact_id, "audio-4")
        self.assertEqual(finished.result.sample_artifact_id, "sample-1")
        self.assertEqual(finished.result.dominant_speaker_word_count, 12)

    def test_reference_sample_convenience_helper_waits_for_job(self):
        stub = types.SimpleNamespace(
            SubmitExtractReferenceSampleJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="ref-2",
                    state=voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_QUEUED,
                )
            ),
            GetExtractReferenceSampleJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="ref-2",
                    state=voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_SUCCEEDED,
                    error_code="",
                    error_message="",
                    sample_artifact=types.SimpleNamespace(artifact_id="sample-2"),
                    trim_start_seconds=2.0,
                    trim_end_seconds=4.0,
                    window_start_seconds=1.0,
                    window_end_seconds=4.5,
                    dominant_speaker_word_count=9,
                    created_at_unix_seconds=1.0,
                    started_at_unix_seconds=2.0,
                    finished_at_unix_seconds=3.0,
                )
            ),
        )
        with (
            patch("dictator.client.voice.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=stub),
            patch("dictator.client.voice.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-5")),
        ):
            client = ReferenceSampleClient(object())
            result = client.extract_bytes(b"abc", language_code="en", poll_interval_seconds=0.01)
        self.assertEqual(
            result,
            ReferenceSampleResult(
                sample_artifact_id="sample-2",
                trim_start_seconds=2.0,
                trim_end_seconds=4.0,
                window_start_seconds=1.0,
                window_end_seconds=4.5,
                dominant_speaker_word_count=9,
            ),
        )
