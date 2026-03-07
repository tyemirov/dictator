import tempfile
import time
from pathlib import Path
import unittest

import grpc

from dictator.alignment.models import AlignTranscriptResult, AlignedWord
from dictator.alignment.srt import build_srt
from dictator.client import DictationClient
from dictator.runtime import InflightLimiter, MetricsRegistry
from dictator.speech.v1 import (
    alignment_pb2,
    alignment_pb2_grpc,
    artifacts_pb2,
    artifacts_pb2_grpc,
    runtime_pb2,
    runtime_pb2_grpc,
    transcription_pb2,
    transcription_pb2_grpc,
)
from dictator.storage import LocalArtifactStore
from dictator.transport.grpc.config import ServerConfig
from dictator.transport.grpc.server import build_server
from dictator.transport.grpc.services import ServiceContext
from dictator.transcription.models import WordSegment


class FakeTranscriptionService:
    def transcribe_word_segments(
        self,
        audio,
        language=None,
        model_size="base",
        model=None,
        progress_cb=None,
    ):
        if model_size == "sleep":
            time.sleep(0.05)
        if progress_cb is not None:
            progress_cb(1.0)
        return [
            WordSegment("hello", 0.0, 0.4),
            WordSegment("world", 0.4, 0.9),
        ]


class FakeAlignmentService:
    def align(self, request):
        words = (
            AlignedWord("hello", 0.0, 0.4),
            AlignedWord("world", 0.4, 0.9),
        )
        srt_text = build_srt(words)
        if request.output_srt_path is not None:
            request.output_srt_path.write_text(srt_text, encoding="utf-8")
        return AlignTranscriptResult(
            audio_path=request.audio_path,
            language=request.language or "en",
            words=words,
            srt_text=srt_text,
            output_srt_path=request.output_srt_path,
        )


class FakeRuntime:
    def __init__(self):
        self.transcription_service = FakeTranscriptionService()
        self.alignment_service = FakeAlignmentService()

    def get_transcription_service(self):
        return self.transcription_service

    def get_alignment_service(self):
        return self.alignment_service

    def get_reference_extraction_service(self):
        raise NotImplementedError

    def get_synthesis_service(self):
        raise NotImplementedError

    def get_whisper_model(self, model_size: str):
        return object()

    def get_diarization_pipeline(self):
        return object()


class GrpcTransportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        artifact_root = Path(cls._tmpdir.name)
        cls._auth_metadata = (("x-dictator-token", "secret"),)
        service_context = ServiceContext(
            artifact_store=LocalArtifactStore(artifact_root),
            execution_runtime=FakeRuntime(),
            metrics=MetricsRegistry(),
            limiter=InflightLimiter(4),
            auth_token="secret",
            download_chunk_bytes=2,
        )
        cls.server = build_server(
            ServerConfig(artifact_root=artifact_root, auth_token="secret"),
            service_context=service_context,
        )
        port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(cls.channel).result(timeout=5)
        cls.artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(cls.channel)
        cls.transcription_stub = transcription_pb2_grpc.TranscriptionServiceStub(cls.channel)
        cls.alignment_stub = alignment_pb2_grpc.AlignmentServiceStub(cls.channel)
        cls.runtime_stub = runtime_pb2_grpc.RuntimeServiceStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.server.stop(None)
        cls._tmpdir.cleanup()

    def _upload_artifact(self, filename: str, payload: bytes) -> str:
        def request_iter():
            yield artifacts_pb2.UploadArtifactChunk(
                metadata=artifacts_pb2.UploadArtifactMetadata(
                    filename=filename,
                    media_type="application/octet-stream",
                )
            )
            for index in range(0, len(payload), 3):
                yield artifacts_pb2.UploadArtifactChunk(content=payload[index : index + 3])

        response = self.artifact_stub.UploadArtifact(request_iter(), metadata=self._auth_metadata)
        return response.artifact.artifact_id

    def test_auth_is_required(self):
        with self.assertRaises(grpc.RpcError) as exc:
            self.runtime_stub.GetMetrics(runtime_pb2.GetMetricsRequest())
        self.assertEqual(exc.exception.code(), grpc.StatusCode.UNAUTHENTICATED)

    def test_upload_download_transcribe_align_and_metrics(self):
        artifact_id = self._upload_artifact("sample.wav", b"abcdef")

        download_chunks = list(
            self.artifact_stub.DownloadArtifact(
                artifacts_pb2.DownloadArtifactRequest(artifact_id=artifact_id, chunk_size=2),
                metadata=self._auth_metadata,
            )
        )
        self.assertEqual(b"".join(chunk.content for chunk in download_chunks), b"abcdef")

        transcription = self.transcription_stub.Transcribe(
            transcription_pb2.TranscribeRequest(
                audio_artifact_id=artifact_id,
                language_code="en",
                model_size="base",
                include_word_segments=True,
            ),
            metadata=self._auth_metadata,
        )
        self.assertEqual(transcription.text, "hello world")
        self.assertEqual([word.content for word in transcription.words], ["hello", "world"])

        alignment = self.alignment_stub.AlignTranscript(
            alignment_pb2.AlignTranscriptRequest(
                audio_artifact_id=artifact_id,
                transcript_text="hello world",
                language_code="en",
                include_srt_text=True,
            ),
            metadata=self._auth_metadata,
        )
        self.assertEqual([word.content for word in alignment.words], ["hello", "world"])
        self.assertIn("00:00:00,000 --> 00:00:00,400", alignment.srt_text)
        self.assertTrue(alignment.srt_artifact_id)

        srt_chunks = list(
            self.artifact_stub.DownloadArtifact(
                artifacts_pb2.DownloadArtifactRequest(artifact_id=alignment.srt_artifact_id, chunk_size=64),
                metadata=self._auth_metadata,
            )
        )
        self.assertIn("hello", b"".join(chunk.content for chunk in srt_chunks).decode("utf-8"))

        metrics = self.runtime_stub.GetMetrics(runtime_pb2.GetMetricsRequest(), metadata=self._auth_metadata)
        self.assertGreaterEqual(metrics.requests_total, 5)
        self.assertGreaterEqual(metrics.requests_succeeded, 5)
        self.assertGreaterEqual(metrics.bytes_received, 6)

    def test_invalid_download_chunk_size_is_rejected(self):
        artifact_id = self._upload_artifact("sample.wav", b"abcdef")
        with self.assertRaises(grpc.RpcError) as exc:
            list(
                self.artifact_stub.DownloadArtifact(
                    artifacts_pb2.DownloadArtifactRequest(artifact_id=artifact_id, chunk_size=-1),
                    metadata=self._auth_metadata,
                )
            )
        self.assertEqual(exc.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    def test_deadline_exceeded_after_backend_work(self):
        artifact_id = self._upload_artifact("sample.wav", b"abcdef")
        with self.assertRaises(grpc.RpcError) as exc:
            self.transcription_stub.Transcribe(
                transcription_pb2.TranscribeRequest(
                    audio_artifact_id=artifact_id,
                    language_code="en",
                    model_size="sleep",
                ),
                metadata=self._auth_metadata,
                timeout=0.01,
            )
        self.assertIn(
            exc.exception.code(),
            {grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.CANCELLED},
        )

    def test_dictation_client_returns_llm_proxy_shape(self):
        client = DictationClient(self.channel, metadata=self._auth_metadata, chunk_bytes=2)
        result = client.dictate_bytes(
            b"abcdef",
            filename="audio.webm",
            model_size="base",
            language_code="en",
            include_word_segments=True,
        )
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.to_http_payload(), {"text": "hello world"})
        self.assertEqual([word["content"] for word in result.words], ["hello", "world"])


if __name__ == "__main__":
    unittest.main()
