from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc

from dictator.speech.v1 import alignment_pb2


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status_code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._status_code = status_code
        self._details = details

    def code(self):
        return self._status_code

    def details(self):
        return self._details


class AlignmentClientTests(unittest.TestCase):
    def test_alignment_job_helpers_and_async_convenience(self):
        from dictator.client.alignment import AlignmentClient

        stub = types.SimpleNamespace(
            SubmitAlignTranscriptJob=MagicMock(
                return_value=types.SimpleNamespace(
                    job_id="align-1",
                    state=alignment_pb2.ALIGNMENT_JOB_STATE_QUEUED,
                )
            ),
            GetAlignTranscriptJob=MagicMock(
                side_effect=[
                    types.SimpleNamespace(
                        job_id="align-1",
                        state=alignment_pb2.ALIGNMENT_JOB_STATE_RUNNING,
                        error_code="",
                        error_message="",
                        language_code="",
                        words=[],
                        srt_text="",
                        srt_artifact_id="",
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=0.0,
                    ),
                    types.SimpleNamespace(
                        job_id="align-1",
                        state=alignment_pb2.ALIGNMENT_JOB_STATE_SUCCEEDED,
                        error_code="",
                        error_message="",
                        language_code="en",
                        words=[
                            types.SimpleNamespace(content="hello", start_seconds=0.0, end_seconds=0.4),
                            types.SimpleNamespace(content="world", start_seconds=0.4, end_seconds=0.9),
                        ],
                        srt_text="1\n00:00:00,000 --> 00:00:00,900\nhello world\n",
                        srt_artifact_id="srt-1",
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    ),
                    types.SimpleNamespace(
                        job_id="align-2",
                        state=alignment_pb2.ALIGNMENT_JOB_STATE_SUCCEEDED,
                        error_code="",
                        error_message="",
                        language_code="en",
                        words=[types.SimpleNamespace(content="hi", start_seconds=0.0, end_seconds=0.2)],
                        srt_text="1\n00:00:00,000 --> 00:00:00,200\nhi\n",
                        srt_artifact_id="srt-2",
                        created_at_unix_seconds=1.0,
                        started_at_unix_seconds=2.0,
                        finished_at_unix_seconds=3.0,
                    ),
                ]
            ),
        )
        with (
            patch("dictator.client.alignment.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.alignment.alignment_pb2_grpc.AlignmentServiceStub", return_value=stub),
            patch(
                "dictator.client.alignment.upload_audio_artifact",
                side_effect=[
                    types.SimpleNamespace(artifact_id="audio-1"),
                    types.SimpleNamespace(artifact_id="audio-2"),
                ],
            ),
            patch("dictator.client._jobs.time.sleep"),
        ):
            client = AlignmentClient(object())
            submitted = client.submit_align_bytes_job(
                b"abc",
                transcript_text="hello world",
                language_code="en",
                remove_punctuation=True,
                include_srt_text=False,
            )
            finished = client.wait_for_alignment_job("align-1", poll_interval_seconds=0.01)
            result = client.align_bytes(
                b"xyz",
                transcript_text="hi",
                language_code="en",
                poll_interval_seconds=0.01,
            )
        self.assertEqual(submitted.source_artifact_id, "audio-1")
        submit_request = stub.SubmitAlignTranscriptJob.call_args_list[0].args[0]
        self.assertEqual(submit_request.transcript_text, "hello world")
        self.assertTrue(submit_request.remove_punctuation)
        self.assertFalse(submit_request.include_srt_text)
        self.assertEqual(finished.result.srt_artifact_id, "srt-1")
        self.assertEqual(finished.result.words[1]["content"], "world")
        self.assertEqual(result.source_artifact_id, "audio-2")
        self.assertEqual(result.srt_artifact_id, "srt-2")
        self.assertEqual(result.words[0]["content"], "hi")

    def test_alignment_convenience_falls_back_to_sync_rpc(self):
        from dictator.client.alignment import AlignmentClient

        sync_response = types.SimpleNamespace(
            language_code="en",
            words=[types.SimpleNamespace(content="fallback", start_seconds=0.0, end_seconds=0.5)],
            srt_text="1\n00:00:00,000 --> 00:00:00,500\nfallback\n",
            srt_artifact_id="srt-sync",
        )
        stub = types.SimpleNamespace(
            SubmitAlignTranscriptJob=MagicMock(
                side_effect=_FakeRpcError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "alignment job manager is not configured",
                )
            ),
            AlignTranscript=MagicMock(return_value=sync_response),
        )
        with (
            patch("dictator.client.alignment.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.alignment.alignment_pb2_grpc.AlignmentServiceStub", return_value=stub),
            patch("dictator.client.alignment.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio-sync")),
        ):
            client = AlignmentClient(object())
            result = client.align_bytes(
                b"abc",
                transcript_text="fallback",
                language_code="en",
            )
        self.assertEqual(result.source_artifact_id, "audio-sync")
        self.assertEqual(result.srt_artifact_id, "srt-sync")
        self.assertEqual(result.words[0]["content"], "fallback")
        stub.AlignTranscript.assert_called_once()

    def test_alignment_file_helper_and_source_validation(self):
        from dictator.client.alignment import AlignmentClient

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "audio.wav"
            transcript = root / "transcript.txt"
            audio.write_bytes(b"audio")
            transcript.write_text("hello from file", encoding="utf-8")

            with (
                patch("dictator.client.alignment.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
                patch("dictator.client.alignment.alignment_pb2_grpc.AlignmentServiceStub", return_value=object()),
            ):
                client = AlignmentClient(object())
            with patch.object(client, "align_bytes", return_value="ok") as align_bytes_mock:
                self.assertEqual(
                    client.align_file(audio, transcript_file=transcript, language_code="en"),
                    "ok",
                )
            self.assertEqual(
                align_bytes_mock.call_args.kwargs["transcript_text"],
                "hello from file",
            )

        with self.assertRaisesRegex(ValueError, "exactly one"):
            AlignmentClient._resolve_transcript_source(
                transcript_text=None,
                transcript_file=None,
                transcript_artifact_id="",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            AlignmentClient._resolve_transcript_source(
                transcript_text="hello",
                transcript_file=Path("transcript.txt"),
                transcript_artifact_id="artifact-1",
            )


if __name__ == "__main__":
    unittest.main()
