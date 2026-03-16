from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import grpc
from google.protobuf import struct_pb2

from dictator.client import (
    DiarizationClient,
    DictationClient,
    ReferenceSampleClient,
    ReferenceSampleResult,
    RemoteJobFailedError,
    SynthesisClient,
    SubtitleClient,
)
from dictator.speech.v1 import subtitle_pb2, transcription_pb2, voice_pb2


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status_code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._status_code = status_code
        self._details = details

    def code(self):
        return self._status_code

    def details(self):
        return self._details


class ClientJobHelpersTests(unittest.TestCase):
    def test_synthesis_job_helpers_submit_get_and_wait(self):
        stub = types.SimpleNamespace(
            SubmitSynthesizeSpeechJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="syn-1",
                    state=voice_pb2.SYNTHESIS_JOB_STATE_QUEUED,
                )
            ),
            GetSynthesizeSpeechJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="syn-1",
                    state=voice_pb2.SYNTHESIS_JOB_STATE_SUCCEEDED,
                    error_code="",
                    error_message="",
                    audio_artifact=types.SimpleNamespace(artifact_id="audio-1"),
                    audio_duration_seconds=4.2,
                    timeline_artifact_id="timeline-1",
                    chunk_count=3,
                    estimated_total_chunks=3,
                    completed_chunks=2,
                    created_at_unix_seconds=1.0,
                    started_at_unix_seconds=2.0,
                    finished_at_unix_seconds=3.0,
                )
            ),
        )
        waited_job = types.SimpleNamespace(job_id="syn-1", state="SYNTHESIS_JOB_STATE_SUCCEEDED")
        with (
            patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=stub),
            patch("dictator.client.voice.wait_for_job", return_value=waited_job) as wait_for_job,
        ):
            client = SynthesisClient(object(), metadata=[("authorization", "Bearer secret")])
            submitted = client.submit_synthesize_job(
                speaker_artifact_id="speaker-1",
                text_artifact_id="text-1",
                language_code="en",
                max_duration_seconds=5.0,
                include_timeline=True,
                speaker_transcript_text="sample transcript",
            )
            submitted_with_text = client.submit_synthesize_job(
                speaker_artifact_id="speaker-2",
                text="hello world",
            )
            fetched = client.get_synthesis_job("syn-1")
            waited = client.wait_for_synthesis_job("syn-1", timeout_seconds=9.0, poll_interval_seconds=0.25)

        request = stub.SubmitSynthesizeSpeechJob.call_args_list[0].args[0]
        self.assertEqual(request.speaker_artifact_id, "speaker-1")
        self.assertEqual(request.text_artifact_id, "text-1")
        self.assertEqual(request.language_code, "en")
        self.assertEqual(request.max_duration_seconds, 5.0)
        self.assertTrue(request.include_timeline)
        self.assertEqual(request.speaker_transcript_text, "sample transcript")
        self.assertEqual(stub.SubmitSynthesizeSpeechJob.call_args.kwargs["metadata"], (("authorization", "Bearer secret"),))
        self.assertEqual(submitted.job_id, "syn-1")
        self.assertEqual(submitted.state, "SYNTHESIS_JOB_STATE_QUEUED")
        direct_text_request = stub.SubmitSynthesizeSpeechJob.call_args_list[1].args[0]
        self.assertEqual(direct_text_request.speaker_artifact_id, "speaker-2")
        self.assertEqual(direct_text_request.text, "hello world")
        self.assertEqual(submitted_with_text.job_id, "syn-1")
        self.assertEqual(fetched.result.audio_artifact_id, "audio-1")
        self.assertEqual(fetched.result.audio_duration_seconds, 4.2)
        self.assertEqual(fetched.result.timeline_artifact_id, "timeline-1")
        self.assertEqual(fetched.result.chunk_count, 3)
        self.assertEqual(waited, waited_job)
        wait_for_job.assert_called_once()

    def test_synthesis_convenience_helper_waits_and_requires_result(self):
        with patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=object()):
            client = SynthesisClient(object())

        with (
            patch.object(client, "submit_synthesize_job", return_value=types.SimpleNamespace(job_id="syn-2")) as submit_job,
            patch.object(
                client,
                "wait_for_synthesis_job",
                return_value=types.SimpleNamespace(
                    result=types.SimpleNamespace(
                        audio_artifact_id="audio-2",
                        audio_duration_seconds=3.5,
                        timeline_artifact_id="timeline-2",
                        chunk_count=2,
                    )
                ),
            ) as wait_job,
        ):
            result = client.synthesize(
                speaker_artifact_id="speaker-2",
                text="hello world",
                timeout_seconds=8.0,
                poll_interval_seconds=0.5,
            )
        self.assertEqual(result.audio_artifact_id, "audio-2")
        submit_job.assert_called_once()
        wait_job.assert_called_once_with("syn-2", timeout_seconds=8.0, poll_interval_seconds=0.5)

        with (
            patch.object(client, "submit_synthesize_job", return_value=types.SimpleNamespace(job_id="syn-3")),
            patch.object(client, "wait_for_synthesis_job", return_value=types.SimpleNamespace(result=None)),
        ):
            with self.assertRaisesRegex(RuntimeError, "without a result payload"):
                client.synthesize(speaker_artifact_id="speaker-3", text="hello again")

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

    def test_wait_for_job_validates_inputs_and_times_out(self):
        from dictator.client._jobs import wait_for_job

        with self.assertRaisesRegex(ValueError, "timeout_seconds"):
            wait_for_job(lambda: types.SimpleNamespace(state="QUEUED"), timeout_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "poll_interval_seconds"):
            wait_for_job(
                lambda: types.SimpleNamespace(state="QUEUED"),
                timeout_seconds=1.0,
                poll_interval_seconds=0.0,
            )

        with (
            patch("dictator.client._jobs.time.monotonic", side_effect=[10.0, 11.1]),
            patch("dictator.client._jobs.time.sleep"),
        ):
            with self.assertRaisesRegex(TimeoutError, "did not complete"):
                wait_for_job(
                    lambda: types.SimpleNamespace(state="TRANSCRIPTION_JOB_STATE_QUEUED"),
                    timeout_seconds=1.0,
                    poll_interval_seconds=0.1,
                )

    def test_dictation_sync_fallback_and_error_paths(self):
        sync_stub = types.SimpleNamespace(
            SubmitTranscribeJob=MagicMock(
                side_effect=_FakeRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "transcription job manager is not configured",
                )
            ),
            Transcribe=MagicMock(
                return_value=types.SimpleNamespace(
                    text="fallback",
                    language_code="en",
                    words=[types.SimpleNamespace(content="fallback", start_seconds=0.0, end_seconds=0.4)],
                )
            ),
        )
        with (
            patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub", return_value=sync_stub),
            patch("dictator.client.dictation.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-sync")),
        ):
            client = DictationClient(object())
            result = client.dictate_bytes(b"abc", language_code="en")
        self.assertEqual(result.text, "fallback")
        self.assertEqual(result.artifact_id, "audio-sync")
        sync_stub.Transcribe.assert_called_once()

        with (
            patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub", return_value=object()),
            patch("dictator.client.dictation.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-3")),
        ):
            client = DictationClient(object())
        client._transcription_stub = types.SimpleNamespace(
            SubmitTranscribeJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="tx-3",
                    state=transcription_pb2.TRANSCRIPTION_JOB_STATE_QUEUED,
                )
            )
        )
        with (
            patch("dictator.client.dictation.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-3")),
            patch.object(client, "wait_for_dictation_job", return_value=types.SimpleNamespace(result=None)),
        ):
            with self.assertRaisesRegex(RuntimeError, "without a result payload"):
                client.dictate_bytes(b"abc")
        client._transcription_stub = types.SimpleNamespace(
            SubmitTranscribeJob=MagicMock(
                side_effect=_FakeRpcError(grpc.StatusCode.INTERNAL, "boom"),
            )
        )
        with patch("dictator.client.dictation.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-3")):
            with self.assertRaises(grpc.RpcError):
                client.dictate_bytes(b"abc")

    def test_diarization_sync_fallback_and_error_paths(self):
        diarization_struct = struct_pb2.Struct()
        diarization_struct.update({"text": "fallback", "speakers": [{"speaker": "S1"}]})
        sync_stub = types.SimpleNamespace(
            SubmitDiarizeAudioJob=MagicMock(
                side_effect=_FakeRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "diarization job manager is not configured",
                )
            ),
            DiarizeAudio=MagicMock(
                return_value=types.SimpleNamespace(
                    text="fallback",
                    language_code="en",
                    diarization=diarization_struct,
                    diarization_artifact_id="json-sync",
                )
            ),
        )
        with (
            patch("dictator.client.diarization.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.diarization.transcription_pb2_grpc.TranscriptionServiceStub", return_value=sync_stub),
            patch("dictator.client.diarization.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-sync")),
        ):
            client = DiarizationClient(object())
            result = client.diarize_bytes(
                b"abc",
                language_code="en",
                utterance_gap_seconds=0.5,
                persist_json_artifact=True,
            )
        self.assertEqual(result.diarization["text"], "fallback")
        self.assertEqual(result.diarization_artifact_id, "json-sync")
        sync_stub.DiarizeAudio.assert_called_once()

        with (
            patch("dictator.client.diarization.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.diarization.transcription_pb2_grpc.TranscriptionServiceStub", return_value=object()),
            patch("dictator.client.diarization.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-2")),
        ):
            client = DiarizationClient(object())
        client._transcription_stub = types.SimpleNamespace(
            SubmitDiarizeAudioJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="dia-2",
                    state=transcription_pb2.DIARIZATION_JOB_STATE_QUEUED,
                )
            )
        )
        with (
            patch("dictator.client.diarization.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-2")),
            patch.object(client, "wait_for_diarization_job", return_value=types.SimpleNamespace(result=None)),
        ):
            with self.assertRaisesRegex(RuntimeError, "without a result payload"):
                client.diarize_bytes(b"abc")
        client._transcription_stub = types.SimpleNamespace(
            SubmitDiarizeAudioJob=MagicMock(
                side_effect=_FakeRpcError(grpc.StatusCode.INTERNAL, "boom"),
            )
        )
        with patch("dictator.client.diarization.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-2")):
            with self.assertRaises(grpc.RpcError):
                client.diarize_bytes(b"abc")

    def test_subtitle_sync_fallback_and_error_paths(self):
        sync_stub = types.SimpleNamespace(
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
                    cues=[types.SimpleNamespace(content="fallback", start_seconds=0.0, end_seconds=0.4, item_count=1)],
                    srt_text="1\n00:00:00,000 --> 00:00:00,400\nfallback\n",
                    srt_artifact_id="srt-sync",
                )
            ),
        )
        with (
            patch("dictator.client.subtitles.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.subtitles.subtitle_pb2_grpc.SubtitleServiceStub", return_value=sync_stub),
            patch("dictator.client.subtitles.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-sync")),
        ):
            client = SubtitleClient(object())
            result = client.render_bytes(
                b"abc",
                language_code="en",
                source_text="fallback",
                granularity="words",
            )
        self.assertEqual(result.mode, "forced_alignment")
        self.assertEqual(result.srt_artifact_id, "srt-sync")
        sync_stub.RenderSubtitles.assert_called_once()

        with (
            patch("dictator.client.subtitles.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.subtitles.subtitle_pb2_grpc.SubtitleServiceStub", return_value=object()),
            patch("dictator.client.subtitles.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-2")),
        ):
            client = SubtitleClient(object())
        client._subtitle_stub = types.SimpleNamespace(
            SubmitRenderSubtitlesJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="sub-2",
                    state=subtitle_pb2.SUBTITLE_JOB_STATE_QUEUED,
                )
            )
        )
        with (
            patch("dictator.client.subtitles.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-2")),
            patch.object(client, "wait_for_subtitle_job", return_value=types.SimpleNamespace(result=None)),
        ):
            with self.assertRaisesRegex(RuntimeError, "without a result payload"):
                client.render_bytes(b"abc", source_text="fallback")
        client._subtitle_stub = types.SimpleNamespace(
            SubmitRenderSubtitlesJob=MagicMock(
                side_effect=_FakeRpcError(grpc.StatusCode.INTERNAL, "boom"),
            )
        )
        with patch("dictator.client.subtitles.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-2")):
            with self.assertRaises(grpc.RpcError):
                client.render_bytes(b"abc", source_text="fallback")

    def test_file_submit_helpers_forward_bytes(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            transcript = root / "source.txt"
            audio.write_bytes(b"audio")
            transcript.write_text("subtitle source", encoding="utf-8")

            with (
                patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
                patch("dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub", return_value=object()),
                patch("dictator.client.diarization.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
                patch("dictator.client.diarization.transcription_pb2_grpc.TranscriptionServiceStub", return_value=object()),
                patch("dictator.client.subtitles.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
                patch("dictator.client.subtitles.subtitle_pb2_grpc.SubtitleServiceStub", return_value=object()),
                patch("dictator.client.voice.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
                patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=object()),
            ):
                dictation_client = DictationClient(object())
                diarization_client = DiarizationClient(object())
                subtitle_client = SubtitleClient(object())
                reference_client = ReferenceSampleClient(object())

            with patch.object(dictation_client, "submit_dictate_bytes_job", return_value="dictation-ok") as submit_mock:
                self.assertEqual(dictation_client.submit_dictate_file_job(audio, language_code="en"), "dictation-ok")
            self.assertEqual(submit_mock.call_args.args[0], b"audio")
            self.assertEqual(submit_mock.call_args.kwargs["filename"], "audio.wav")

            with patch.object(diarization_client, "submit_diarize_bytes_job", return_value="diarization-ok") as submit_mock:
                self.assertEqual(diarization_client.submit_diarize_file_job(audio, language_code="en"), "diarization-ok")
            self.assertEqual(submit_mock.call_args.args[0], b"audio")
            self.assertEqual(submit_mock.call_args.kwargs["filename"], "audio.wav")

            with patch.object(subtitle_client, "submit_render_bytes_job", return_value="subtitle-ok") as submit_mock:
                self.assertEqual(
                    subtitle_client.submit_render_file_job(
                        audio,
                        language_code="en",
                        source_text_file=transcript,
                    ),
                    "subtitle-ok",
                )
            self.assertEqual(submit_mock.call_args.args[0], b"audio")
            self.assertEqual(submit_mock.call_args.kwargs["source_text"], "subtitle source")
            self.assertEqual(submit_mock.call_args.kwargs["source_text_name"], "source.txt")

            with patch.object(reference_client, "submit_extract_bytes_job", return_value="reference-ok") as submit_mock:
                self.assertEqual(reference_client.submit_extract_file_job(audio, language_code="en"), "reference-ok")
            self.assertEqual(submit_mock.call_args.args[0], b"audio")
            self.assertEqual(submit_mock.call_args.kwargs["filename"], "audio.wav")

    def test_reference_sample_missing_result(self):
        with (
            patch("dictator.client.voice.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.voice.voice_pb2_grpc.VoiceServiceStub", return_value=object()),
        ):
            client = ReferenceSampleClient(object())
        with (
            patch.object(
                client,
                "submit_extract_bytes_job",
                return_value=types.SimpleNamespace(job_id="ref-3", source_artifact_id="audio-3"),
            ),
            patch.object(client, "wait_for_reference_sample_job", return_value=types.SimpleNamespace(result=None)),
        ):
            with self.assertRaisesRegex(RuntimeError, "without a result payload"):
                client.extract_bytes(b"abc")
