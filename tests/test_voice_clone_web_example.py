from __future__ import annotations

import base64
from contextlib import contextmanager
import http.client
import json
import sys
import threading
import types
import unittest
from unittest import mock

import grpc

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())

from dictator.speech.v1 import artifacts_pb2, common_pb2, voice_pb2
from demo.voice_clone_web import app


class FakeUploadResponse:
    def __init__(self, artifact_id: str) -> None:
        self.artifact = type("Artifact", (), {"artifact_id": artifact_id})()


class FakeArtifactStub:
    def __init__(self, _channel) -> None:
        self.upload_calls = []
        self.download_calls = []

    def UploadArtifact(self, chunks, metadata=None):
        self.upload_calls.append((list(chunks), metadata))
        return FakeUploadResponse("source-artifact")

    def DownloadArtifact(self, request, metadata=None):
        self.download_calls.append((request, metadata))
        return iter(
            [
                artifacts_pb2.DownloadArtifactChunk(
                    artifact=common_pb2.ArtifactRef(
                        artifact_id=request.artifact_id,
                        filename="cloned.wav",
                        media_type="audio/wav",
                    ),
                    content=b"hello ",
                    offset=0,
                    eof=False,
                ),
                artifacts_pb2.DownloadArtifactChunk(
                    artifact=common_pb2.ArtifactRef(
                        artifact_id=request.artifact_id,
                        filename="cloned.wav",
                        media_type="audio/wav",
                    ),
                    content=b"world",
                    offset=6,
                    eof=True,
                ),
            ]
        )


class FakeVoiceStub:
    def __init__(self, _channel) -> None:
        self.extract_calls = []
        self.synthesize_calls = []

    def ExtractReferenceSample(self, request, metadata=None):
        self.extract_calls.append((request, metadata))
        return voice_pb2.ExtractReferenceSampleResponse(
            sample_artifact=common_pb2.ArtifactRef(artifact_id="sample-artifact")
        )

    def SynthesizeSpeech(self, request, metadata=None):
        self.synthesize_calls.append((request, metadata))
        return voice_pb2.SynthesizeSpeechResponse(
            audio_artifact=common_pb2.ArtifactRef(artifact_id="audio-artifact")
        )


class FakeSynthesisClient:
    def __init__(self, _channel, metadata=()) -> None:
        self.metadata = list(metadata)
        self.synthesize_calls = []
        self.submit_calls = []
        self.get_calls = []

    def synthesize(self, **kwargs):
        self.synthesize_calls.append(kwargs)
        return types.SimpleNamespace(audio_artifact_id="audio-artifact")

    def submit_synthesize_job(self, **kwargs):
        self.submit_calls.append(kwargs)
        return types.SimpleNamespace(
            job_id="deadbeef",
            state="SYNTHESIS_JOB_STATE_QUEUED",
            estimated_total_chunks=0,
            completed_chunks=0,
        )

    def get_synthesis_job(self, job_id):
        self.get_calls.append(job_id)
        return types.SimpleNamespace(
            job_id=job_id,
            state="SYNTHESIS_JOB_STATE_SUCCEEDED",
            error_code="",
            error_message="",
            estimated_total_chunks=4,
            completed_chunks=4,
            result=types.SimpleNamespace(audio_artifact_id="audio-artifact"),
        )


class FakeChannel:
    def __init__(self) -> None:
        self.closed = False

    def close(self):
        self.closed = True


class FakeRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAUTHENTICATED

    def details(self):
        return "missing token"


@contextmanager
def running_server(handler):
    server = app.ThreadingHTTPServer((app.DEFAULT_HOST, 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class VoiceCloneWebExampleTests(unittest.TestCase):
    def test_load_index_html_mentions_genesis(self):
        html = app.load_index_html()
        self.assertIn("Read classic passages in your own cloned voice", html)
        self.assertIn("I grew up near a busy street", html)
        self.assertIn("the one I use when I am truly delighted", html)
        self.assertIn("Qwen3-TTS is the only voice cloning engine in this demo", html)
        self.assertIn('option value="alice"', html)
        self.assertIn('option value="woods"', html)
        self.assertNotIn("Dictator gRPC URL", html)
        self.assertNotIn("Language Code", html)
        self.assertNotIn("Auth Token", html)
        self.assertIn("Record your voice sample", html)
        self.assertIn('class="sample-script"', html)
        self.assertIn("generatedAudio.load();", html)
        self.assertIn("clearGeneratedPlayback();", html)

    def test_parse_grpc_target_accepts_host_port_only(self):
        self.assertEqual(app.parse_grpc_target("localhost:50051"), "localhost:50051")

    def test_parse_grpc_target_rejects_schemes_and_paths(self):
        with self.assertRaisesRegex(app.ExampleRequestError, "host:port only"):
            app.parse_grpc_target("grpc://localhost:50051/path")
        with self.assertRaisesRegex(app.ExampleRequestError, "host:port only"):
            app.parse_grpc_target("https://voice.example:443")
        with self.assertRaisesRegex(app.ExampleRequestError, "host:port only"):
            app.parse_grpc_target("localhost")
        with self.assertRaisesRegex(app.ExampleRequestError, "host:port only"):
            app.parse_grpc_target("localhost:50051/extra")
        with self.assertRaisesRegex(app.ExampleRequestError, "required"):
            app.parse_grpc_target("")
        with self.assertRaisesRegex(app.ExampleRequestError, "host:port only"):
            app.parse_grpc_target("grpc:///")

    def test_resolve_bridge_target_uses_configured_bridge_target(self):
        self.assertEqual(app.resolve_bridge_target("dictator-grpc:50051"), "dictator-grpc:50051")
        with self.assertRaisesRegex(app.ExampleRequestError, "bridge target is not configured"):
            app.resolve_bridge_target("")

    def test_resolve_bridge_auth_token_uses_backend_configuration(self):
        self.assertEqual(app.resolve_bridge_auth_token("secret"), "secret")
        with self.assertRaisesRegex(app.ExampleRequestError, "bridge auth token is not configured"):
            app.resolve_bridge_auth_token("")

    def test_resolve_render_preset_uses_defaults_and_rejects_unknown_values(self):
        self.assertEqual(app.resolve_render_preset("alice").preset_id, "alice")
        self.assertEqual(app.resolve_render_preset("").preset_id, app.DEFAULT_RENDER_PRESET_ID)
        with self.assertRaisesRegex(app.ExampleRequestError, "Unknown reading selection"):
            app.resolve_render_preset("unknown")

    def test_build_auth_metadata_requires_token(self):
        self.assertEqual(app.build_auth_metadata("secret"), [("authorization", "Bearer secret")])
        with self.assertRaisesRegex(app.ExampleRequestError, "Auth token"):
            app.build_auth_metadata("")

    def test_create_channel_uses_insecure_grpc(self):
        insecure = object()
        with mock.patch("demo.voice_clone_web.app.grpc.insecure_channel", return_value=insecure) as insecure_channel:
            self.assertIs(app.create_channel("localhost:50051"), insecure)
        insecure_channel.assert_called_once_with("localhost:50051")

    def test_iter_upload_chunks_emits_metadata_then_payload(self):
        chunks = list(app.iter_upload_chunks("voice.webm", "audio/webm", b"abcdefgh"))
        self.assertEqual(chunks[0].metadata.filename, "voice.webm")
        self.assertEqual(chunks[0].metadata.media_type, "audio/webm")
        self.assertEqual(chunks[1].content, b"abcdefgh")

    def test_upload_artifact_returns_id(self):
        stub = FakeArtifactStub(None)
        artifact_id = app.upload_artifact(
            stub,
            filename="voice.webm",
            media_type="audio/webm",
            payload=b"data",
            metadata=[("authorization", "Bearer secret")],
        )
        self.assertEqual(artifact_id, "source-artifact")
        self.assertEqual(stub.upload_calls[0][1], [("authorization", "Bearer secret")])

    def test_download_artifact_collects_chunks(self):
        stub = FakeArtifactStub(None)
        result = app.download_artifact(
            stub,
            artifact_id="audio-artifact",
            metadata=[("authorization", "Bearer secret")],
        )
        self.assertEqual(result.filename, "cloned.wav")
        self.assertEqual(result.media_type, "audio/wav")
        self.assertEqual(result.content, b"hello world")
        self.assertEqual(stub.download_calls[0][0].artifact_id, "audio-artifact")

    def test_synthesize_selected_reading_calls_service_flow(self):
        fake_channel = FakeChannel()
        fake_artifacts = FakeArtifactStub(fake_channel)
        fake_synthesis_client = FakeSynthesisClient(fake_channel)

        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=fake_artifacts,
        ) as artifact_stub_ctor:
            with mock.patch(
                "demo.voice_clone_web.app.SynthesisClient",
                return_value=fake_synthesis_client,
            ) as synthesis_client_ctor:
                with mock.patch(
                    "demo.voice_clone_web.app.normalise_recorded_audio",
                    return_value=(b"wav-bytes", "voice-sample.wav", "audio/wav"),
                ) as normalise:
                    result = app.synthesize_selected_reading(
                        dictator_url="voice.example:443",
                        auth_token="secret",
                        audio_payload=b"sample-bytes",
                        audio_filename="voice.webm",
                        audio_media_type="audio/webm",
                        render_preset_id="alice",
                        channel_factory=lambda target: fake_channel,
                    )

        artifact_stub_ctor.assert_called_once_with(fake_channel)
        synthesis_client_ctor.assert_called_once_with(
            fake_channel,
            metadata=[("authorization", "Bearer secret")],
        )
        normalise.assert_called_once_with(
            b"sample-bytes",
            "voice.webm",
            "audio/webm",
        )
        upload_chunks, upload_metadata = fake_artifacts.upload_calls[0]
        self.assertEqual(upload_chunks[0].metadata.filename, "voice-sample.wav")
        self.assertEqual(upload_chunks[0].metadata.media_type, "audio/wav")
        self.assertEqual(upload_metadata, [("authorization", "Bearer secret")])
        synth_kwargs = fake_synthesis_client.synthesize_calls[0]
        self.assertEqual(synth_kwargs["speaker_artifact_id"], "source-artifact")
        self.assertEqual(synth_kwargs["synthesis_engine"], voice_pb2.SYNTHESIS_ENGINE_QWEN3)
        self.assertEqual(synth_kwargs["speaker_transcript_text"], app.VOICE_SAMPLE_TEXT)
        self.assertIn("Alice was beginning to get very tired", synth_kwargs["text"])
        self.assertEqual(result.content, b"hello world")
        self.assertEqual(result.filename, "alice-in-your-voice.wav")
        self.assertTrue(fake_channel.closed)

    def test_synthesize_selected_reading_uses_qwen3_sample_transcript(self):
        fake_channel = FakeChannel()
        fake_artifacts = FakeArtifactStub(fake_channel)
        fake_synthesis_client = FakeSynthesisClient(fake_channel)

        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=fake_artifacts,
        ):
            with mock.patch(
                "demo.voice_clone_web.app.SynthesisClient",
                return_value=fake_synthesis_client,
            ):
                with mock.patch(
                    "demo.voice_clone_web.app.normalise_recorded_audio",
                    return_value=(b"wav-bytes", "voice-sample.wav", "audio/wav"),
                ) as normalise:
                    app.synthesize_selected_reading(
                        dictator_url="dictator-grpc:50051",
                        auth_token="secret",
                        audio_payload=b"sample-bytes",
                        audio_filename="voice.webm",
                        audio_media_type="audio/webm",
                    )

        normalise.assert_called_once_with(b"sample-bytes", "voice.webm", "audio/webm")
        synth_kwargs = fake_synthesis_client.synthesize_calls[0]
        self.assertEqual(synth_kwargs["synthesis_engine"], voice_pb2.SYNTHESIS_ENGINE_QWEN3)
        self.assertEqual(synth_kwargs["speaker_transcript_text"], app.VOICE_SAMPLE_TEXT)

    def test_synthesize_selected_reading_requires_audio(self):
        with self.assertRaisesRegex(app.ExampleRequestError, "voice sample"):
            app.synthesize_selected_reading(
                dictator_url="localhost:50051",
                auth_token="secret",
                audio_payload=b"",
                audio_filename="voice.webm",
                audio_media_type="audio/webm",
            )

    def test_submit_and_get_selected_reading_job_expose_progress(self):
        fake_channel = FakeChannel()
        fake_artifacts = FakeArtifactStub(fake_channel)
        fake_synthesis_client = FakeSynthesisClient(fake_channel)

        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=fake_artifacts,
        ), mock.patch(
            "demo.voice_clone_web.app.SynthesisClient",
            return_value=fake_synthesis_client,
        ), mock.patch(
            "demo.voice_clone_web.app.normalise_recorded_audio",
            return_value=(b"wav-bytes", "voice-sample.wav", "audio/wav"),
        ):
            submitted = app.submit_selected_reading_job(
                dictator_url="dictator-grpc:50051",
                auth_token="secret",
                audio_payload=b"sample-bytes",
                audio_filename="voice.webm",
                audio_media_type="audio/webm",
                render_preset_id="woods",
            )
            status = app.get_selected_reading_job(
                dictator_url="dictator-grpc:50051",
                auth_token="secret",
                job_id="deadbeef",
            )

        self.assertEqual(submitted.job_id, "deadbeef")
        self.assertEqual(submitted.state, "SYNTHESIS_JOB_STATE_QUEUED")
        submit_kwargs = fake_synthesis_client.submit_calls[0]
        self.assertEqual(submit_kwargs["speaker_artifact_id"], "source-artifact")
        self.assertEqual(submit_kwargs["synthesis_engine"], voice_pb2.SYNTHESIS_ENGINE_QWEN3)
        self.assertEqual(submit_kwargs["speaker_transcript_text"], app.VOICE_SAMPLE_TEXT)
        self.assertIn("Whose woods these are I think I know.", submit_kwargs["text"])
        self.assertEqual(status.job_id, "deadbeef")
        self.assertEqual(status.estimated_total_chunks, 4)
        self.assertEqual(status.completed_chunks, 4)
        self.assertEqual(status.audio_artifact_id, "audio-artifact")
        self.assertEqual(status.progress_percent(), 1.0)

    def test_voice_clone_job_progress_percent_stops_short_before_success(self):
        job = app.VoiceCloneJob(
            job_id="deadbeef",
            state="SYNTHESIS_JOB_STATE_RUNNING",
            estimated_total_chunks=4,
            completed_chunks=4,
        )
        self.assertAlmostEqual(job.progress_percent(), 0.9)

    def test_voice_clone_job_progress_percent_shows_running_estimate(self):
        unknown_total_job = app.VoiceCloneJob(
            job_id="deadbeef",
            state="SYNTHESIS_JOB_STATE_RUNNING",
            estimated_total_chunks=0,
            completed_chunks=0,
        )
        self.assertAlmostEqual(unknown_total_job.progress_percent(), 0.1)

        single_chunk_job = app.VoiceCloneJob(
            job_id="deadbeef",
            state="SYNTHESIS_JOB_STATE_RUNNING",
            estimated_total_chunks=1,
            completed_chunks=0,
        )
        self.assertAlmostEqual(single_chunk_job.progress_percent(), 0.475)

        failed_job = app.VoiceCloneJob(
            job_id="deadbeef",
            state="SYNTHESIS_JOB_STATE_FAILED",
            estimated_total_chunks=4,
            completed_chunks=2,
        )
        self.assertAlmostEqual(failed_job.progress_percent(), 0.0)

    def test_submit_selected_reading_job_requires_audio(self):
        with self.assertRaisesRegex(app.ExampleRequestError, "voice sample"):
            app.submit_selected_reading_job(
                dictator_url="dictator-grpc:50051",
                auth_token="secret",
                audio_payload=b"",
                audio_filename="voice.webm",
                audio_media_type="audio/webm",
            )

    def test_download_selected_reading_job_audio_requires_ready_result(self):
        fake_channel = FakeChannel()
        fake_artifacts = FakeArtifactStub(fake_channel)
        fake_synthesis_client = FakeSynthesisClient(fake_channel)
        fake_synthesis_client.get_synthesis_job = lambda job_id: types.SimpleNamespace(
            job_id=job_id,
            state="SYNTHESIS_JOB_STATE_RUNNING",
            error_code="",
            error_message="",
            estimated_total_chunks=4,
            completed_chunks=3,
            result=None,
        )

        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=fake_artifacts,
        ), mock.patch(
            "demo.voice_clone_web.app.SynthesisClient",
            return_value=fake_synthesis_client,
        ):
            with self.assertRaisesRegex(app.ExampleRequestError, "not ready yet"):
                app.download_selected_reading_job_audio(
                    dictator_url="dictator-grpc:50051",
                    auth_token="secret",
                    job_id="deadbeef",
                )

    def test_download_selected_reading_job_audio_maps_failure_and_success(self):
        failed_client = FakeSynthesisClient(FakeChannel())
        failed_client.get_synthesis_job = lambda job_id: types.SimpleNamespace(
            job_id=job_id,
            state="SYNTHESIS_JOB_STATE_FAILED",
            error_code="dictator.jobs.failed",
            error_message="boom",
            estimated_total_chunks=4,
            completed_chunks=2,
            result=None,
        )
        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=FakeArtifactStub(None),
        ), mock.patch(
            "demo.voice_clone_web.app.SynthesisClient",
            return_value=failed_client,
        ):
            with self.assertRaisesRegex(app.ExampleRequestError, "boom"):
                app.download_selected_reading_job_audio(
                    dictator_url="dictator-grpc:50051",
                    auth_token="secret",
                    job_id="deadbeef",
                )

        success_channel = FakeChannel()
        success_artifacts = FakeArtifactStub(success_channel)
        success_client = FakeSynthesisClient(success_channel)
        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=success_artifacts,
        ), mock.patch(
            "demo.voice_clone_web.app.SynthesisClient",
            return_value=success_client,
        ):
            result = app.download_selected_reading_job_audio(
                dictator_url="dictator-grpc:50051",
                auth_token="secret",
                job_id="deadbeef",
                render_preset_id="alice",
            )
        self.assertEqual(result.filename, "alice-in-your-voice.wav")
        self.assertEqual(result.content, b"hello world")

    def test_decode_request_payload_validates_json_shape(self):
        self.assertEqual(app.decode_request_payload(b'{"ok": true}'), {"ok": True})
        with self.assertRaisesRegex(app.ExampleRequestError, "valid JSON"):
            app.decode_request_payload(b"not-json")
        with self.assertRaisesRegex(app.ExampleRequestError, "JSON object"):
            app.decode_request_payload(json.dumps([1, 2, 3]).encode("utf-8"))

    def test_decode_audio_base64_supports_data_urls(self):
        encoded = base64.b64encode(b"voice-bytes").decode("ascii")
        self.assertEqual(app.decode_audio_base64(encoded), b"voice-bytes")
        self.assertEqual(app.decode_audio_base64(f"data:audio/webm;base64,{encoded}"), b"voice-bytes")
        with self.assertRaisesRegex(app.ExampleRequestError, "Recorded audio is required"):
            app.decode_audio_base64("")
        with self.assertRaisesRegex(app.ExampleRequestError, "base64"):
            app.decode_audio_base64("***")

    def test_normalise_recorded_audio_transcodes_to_wav(self):
        def fake_audio_to_wav(src, dst, max_duration_seconds=None):
            self.assertEqual(src.read_bytes(), b"compressed-audio")
            self.assertIsNone(max_duration_seconds)
            dst.write_bytes(b"RIFFdemo")

        with mock.patch("demo.voice_clone_web.app.audio_to_wav", side_effect=fake_audio_to_wav):
            payload, filename, media_type = app.normalise_recorded_audio(
                b"compressed-audio",
                "voice-sample.webm",
                "audio/webm",
            )

        self.assertEqual(payload, b"RIFFdemo")
        self.assertEqual(filename, "voice-sample.wav")
        self.assertEqual(media_type, "audio/wav")

    def test_normalise_recorded_audio_can_cap_duration(self):
        def fake_audio_to_wav(src, dst, max_duration_seconds=None):
            self.assertEqual(src.read_bytes(), b"compressed-audio")
            self.assertEqual(max_duration_seconds, 12.5)
            dst.write_bytes(b"RIFFdemo")

        with mock.patch("demo.voice_clone_web.app.audio_to_wav", side_effect=fake_audio_to_wav):
            payload, filename, media_type = app.normalise_recorded_audio(
                b"compressed-audio",
                "voice-sample.webm",
                "audio/webm",
                max_duration_seconds=12.5,
            )

        self.assertEqual(payload, b"RIFFdemo")
        self.assertEqual(filename, "voice-sample.wav")
        self.assertEqual(media_type, "audio/wav")

    def test_normalise_recorded_audio_avoids_wav_source_collision(self):
        def fake_audio_to_wav(src, dst, max_duration_seconds=None):
            self.assertEqual(src.name, "voice-sample.wav")
            self.assertEqual(dst.name, "voice-sample-normalised.wav")
            dst.write_bytes(b"RIFFdemo")

        with mock.patch("demo.voice_clone_web.app.audio_to_wav", side_effect=fake_audio_to_wav):
            payload, filename, media_type = app.normalise_recorded_audio(
                b"compressed-audio",
                "voice-sample.wav",
                "audio/wav",
            )

        self.assertEqual(payload, b"RIFFdemo")
        self.assertEqual(filename, "voice-sample-normalised.wav")
        self.assertEqual(media_type, "audio/wav")

    def test_normalise_recorded_audio_reports_transcode_failure(self):
        with mock.patch("demo.voice_clone_web.app.audio_to_wav", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(app.ExampleRequestError, "could not be converted to WAV"):
                app.normalise_recorded_audio(b"compressed-audio", "voice-sample.webm", "audio/webm")

    def test_normalise_recorded_audio_requires_payload(self):
        with self.assertRaisesRegex(app.ExampleRequestError, "voice sample"):
            app.normalise_recorded_audio(b"", "voice-sample.webm", "audio/webm")

    def test_choose_download_filename_strips_paths(self):
        self.assertEqual(app.choose_download_filename("/tmp/voice.wav"), "voice.wav")
        self.assertEqual(app.choose_download_filename(""), app.DEFAULT_OUTPUT_FILENAME)

    def test_handler_serves_index(self):
        with running_server(app.build_handler(index_html="<html>ok</html>")) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("GET", "/")
            response = connection.getresponse()
            body = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertEqual(body, "<html>ok</html>")

    def test_handler_returns_not_found_for_unknown_get(self):
        with running_server(app.build_handler(index_html="<html>ok</html>")) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("GET", "/missing")
            response = connection.getresponse()
            response.read()
        self.assertEqual(response.status, 404)

    def test_handler_rejects_non_json(self):
        with running_server(app.build_handler(index_html="<html>ok</html>")) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("POST", "/api/clone", body=b"audio", headers={"Content-Type": "text/plain"})
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 415)
        self.assertEqual(payload["error"], "Use application/json.")

    def test_handler_returns_binary_audio(self):
        payload = {
            "renderPreset": "alice",
            "audioBase64": base64.b64encode(b"sample").decode("ascii"),
            "audioFilename": "voice.webm",
            "audioMediaType": "audio/webm",
        }

        def synthesizer(**kwargs):
            self.assertEqual(kwargs["dictator_url"], "dictator-grpc:50051")
            self.assertEqual(kwargs["auth_token"], "bridge-secret")
            self.assertEqual(kwargs["audio_payload"], b"sample")
            self.assertEqual(kwargs["render_preset_id"], "alice")
            return app.VoiceCloneResult("demo.wav", "audio/wav", b"rendered")

        with running_server(
            app.build_handler(
                index_html="<html>ok</html>",
                synthesizer=synthesizer,
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request(
                "POST",
                "/api/clone",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "audio/wav")
        self.assertIn('filename="demo.wav"', response.getheader("Content-Disposition"))
        self.assertEqual(body, b"rendered")

    def test_handler_returns_bad_request_on_validation_error(self):
        with running_server(
            app.build_handler(
                synthesizer=lambda **_: (_ for _ in ()).throw(app.ExampleRequestError("bad request")),
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request(
                "POST",
                "/api/clone",
                body=json.dumps({
                    "audioBase64": base64.b64encode(b"sample").decode("ascii"),
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 400)
        self.assertEqual(payload["error"], "bad request")

    def test_handler_maps_grpc_errors_to_bad_gateway(self):
        with running_server(
            app.build_handler(
                synthesizer=lambda **_: (_ for _ in ()).throw(FakeRpcError()),
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request(
                "POST",
                "/api/clone",
                body=json.dumps({
                    "audioBase64": base64.b64encode(b"sample").decode("ascii"),
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 502)
        self.assertIn("UNAUTHENTICATED", payload["error"])

    def test_handler_reports_missing_backend_auth_token(self):
        with running_server(
            app.build_handler(
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request(
                "POST",
                "/api/clone",
                body=json.dumps({
                    "audioBase64": base64.b64encode(b"sample").decode("ascii"),
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 400)
        self.assertEqual(payload["error"], "Dictator bridge auth token is not configured.")

    def test_handler_rejects_unknown_post(self):
        with running_server(app.build_handler(index_html="<html>ok</html>")) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("POST", "/missing", body=b"{}", headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            response.read()
        self.assertEqual(response.status, 404)

    def test_handler_supports_async_clone_job_routes(self):
        submitted_jobs = []
        looked_up_jobs = []
        downloaded_jobs = []

        def fake_submitter(**kwargs):
            submitted_jobs.append(kwargs)
            return app.VoiceCloneJob(
                job_id="deadbeef",
                state="SYNTHESIS_JOB_STATE_QUEUED",
                estimated_total_chunks=0,
                completed_chunks=0,
            )

        def fake_getter(**kwargs):
            looked_up_jobs.append(kwargs)
            return app.VoiceCloneJob(
                job_id=kwargs["job_id"],
                state="SYNTHESIS_JOB_STATE_RUNNING",
                estimated_total_chunks=4,
                completed_chunks=2,
            )

        def fake_downloader(**kwargs):
            downloaded_jobs.append(kwargs)
            return app.VoiceCloneResult(
                filename="alice-in-your-voice.wav",
                media_type="audio/wav",
                content=b"RIFF",
            )

        with running_server(
            app.build_handler(
                index_html="<html>ok</html>",
                job_submitter=fake_submitter,
                job_getter=fake_getter,
                job_audio_downloader=fake_downloader,
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request(
                "POST",
                "/api/clone-jobs",
                body=json.dumps(
                    {
                        "renderPreset": "alice",
                        "audioBase64": base64.b64encode(b"sample").decode("ascii"),
                        "audioFilename": "voice.webm",
                        "audioMediaType": "audio/webm",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            submit_response = connection.getresponse()
            submit_payload = json.loads(submit_response.read().decode("utf-8"))
            self.assertEqual(submit_response.status, 202)
            self.assertEqual(submit_payload["jobId"], "deadbeef")

            connection.request("GET", "/api/clone-jobs/deadbeef")
            status_response = connection.getresponse()
            status_payload = json.loads(status_response.read().decode("utf-8"))
            self.assertEqual(status_response.status, 200)
            self.assertEqual(status_payload["completedChunks"], 2)
            self.assertEqual(status_payload["estimatedTotalChunks"], 4)
            self.assertAlmostEqual(status_payload["progressPercent"], 0.56875)

            connection.request("GET", "/api/clone-jobs/deadbeef/audio?renderPreset=alice")
            audio_response = connection.getresponse()
            self.assertEqual(audio_response.status, 200)
            self.assertEqual(audio_response.read(), b"RIFF")
            self.assertIn('filename="alice-in-your-voice.wav"', audio_response.getheader("Content-Disposition"))
            connection.close()

        self.assertEqual(submitted_jobs[0]["dictator_url"], "dictator-grpc:50051")
        self.assertEqual(looked_up_jobs[0]["job_id"], "deadbeef")
        self.assertEqual(downloaded_jobs[0]["render_preset_id"], "alice")

    def test_handler_maps_async_job_lookup_errors(self):
        with running_server(
            app.build_handler(
                job_getter=lambda **_: (_ for _ in ()).throw(app.ExampleRequestError("bad job lookup")),
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("GET", "/api/clone-jobs/deadbeef")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(payload["error"], "bad job lookup")

        with running_server(
            app.build_handler(
                job_getter=lambda **_: (_ for _ in ()).throw(FakeRpcError()),
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("GET", "/api/clone-jobs/deadbeef")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
        self.assertEqual(response.status, 502)
        self.assertIn("UNAUTHENTICATED", payload["error"])

    def test_handler_maps_async_job_audio_errors(self):
        with running_server(
            app.build_handler(
                job_audio_downloader=lambda **_: (_ for _ in ()).throw(app.ExampleRequestError("not ready")),
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("GET", "/api/clone-jobs/deadbeef/audio")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
        self.assertEqual(response.status, 409)
        self.assertEqual(payload["error"], "not ready")

        with running_server(
            app.build_handler(
                job_audio_downloader=lambda **_: (_ for _ in ()).throw(FakeRpcError()),
                default_dictator_url="dictator-grpc:50051",
                default_auth_token="bridge-secret",
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request("GET", "/api/clone-jobs/deadbeef/audio")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
        self.assertEqual(response.status, 502)
        self.assertIn("UNAUTHENTICATED", payload["error"])

    def test_build_parser_and_main_delegate_to_serve(self):
        parser = app.build_parser()
        args = parser.parse_args(["--host", "0.0.0.0", "--port", "9090"])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9090)
        with mock.patch("demo.voice_clone_web.app.serve") as serve:
            result = app.main(["--host", "0.0.0.0", "--port", "9090"])
        serve.assert_called_once_with(host="0.0.0.0", port=9090)
        self.assertEqual(result, 0)

    def test_serve_runs_and_closes_server(self):
        events = []

        class FakeServer:
            def __init__(self, address, handler):
                events.append(("init", address, handler.__name__))

            def serve_forever(self):
                events.append(("serve_forever",))
                raise KeyboardInterrupt

            def server_close(self):
                events.append(("server_close",))

        with mock.patch("demo.voice_clone_web.app.ThreadingHTTPServer", FakeServer):
            with mock.patch("builtins.print") as printer:
                app.serve(host="127.0.0.2", port=8181)
        self.assertEqual(events[0][0], "init")
        self.assertEqual(events[0][1], ("127.0.0.2", 8181))
        self.assertEqual(events[1], ("serve_forever",))
        self.assertEqual(events[2], ("server_close",))
        printer.assert_called_once_with("Voice clone demo listening on http://127.0.0.2:8181")
