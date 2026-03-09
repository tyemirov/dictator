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
        self.assertIn('data-engine="xtts"', html)
        self.assertIn('data-engine="qwen3"', html)
        self.assertIn('data-engine="cosyvoice3"', html)
        self.assertIn('option value="alice"', html)
        self.assertIn('option value="woods"', html)
        self.assertNotIn("Dictator gRPC URL", html)
        self.assertNotIn("Language Code", html)
        self.assertNotIn("Auth Token", html)
        self.assertIn("Record your voice sample", html)
        self.assertIn('class="sample-script"', html)

    def test_parse_grpc_target_accepts_plain_and_secure_urls(self):
        self.assertEqual(app.parse_grpc_target("localhost:50051"), app.GrpcTarget("localhost:50051", False))
        self.assertEqual(app.parse_grpc_target("grpcs://voice.example:443"), app.GrpcTarget("voice.example:443", True))

    def test_parse_grpc_target_rejects_bad_paths_and_schemes(self):
        with self.assertRaisesRegex(app.ExampleRequestError, "must point"):
            app.parse_grpc_target("grpc://localhost:50051/path")
        with self.assertRaisesRegex(app.ExampleRequestError, "must use"):
            app.parse_grpc_target("ftp://localhost:50051")
        with self.assertRaisesRegex(app.ExampleRequestError, "must not include"):
            app.parse_grpc_target("localhost:50051/extra")
        with self.assertRaisesRegex(app.ExampleRequestError, "required"):
            app.parse_grpc_target("")
        with self.assertRaisesRegex(app.ExampleRequestError, "include a host"):
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

    def test_resolve_synthesis_engine_uses_defaults_and_rejects_unknown_values(self):
        self.assertEqual(app.resolve_synthesis_engine("qwen3").engine_id, "qwen3")
        self.assertEqual(app.resolve_synthesis_engine("cosyvoice3").engine_id, "cosyvoice3")
        self.assertEqual(app.resolve_synthesis_engine("").engine_id, app.DEFAULT_SYNTHESIS_ENGINE_ID)
        self.assertEqual(
            app.resolve_synthesis_engine("qwen3").speaker_transcript_text,
            app.VOICE_SAMPLE_TEXT,
        )
        self.assertEqual(
            app.resolve_synthesis_engine("cosyvoice3").speaker_transcript_text,
            app.VOICE_SAMPLE_TEXT,
        )
        self.assertIsNone(app.resolve_synthesis_engine("xtts").speaker_transcript_text)
        with self.assertRaisesRegex(app.ExampleRequestError, "Unknown synthesis engine"):
            app.resolve_synthesis_engine("other")

    def test_build_auth_metadata_requires_token(self):
        self.assertEqual(app.build_auth_metadata("secret"), [("authorization", "Bearer secret")])
        with self.assertRaisesRegex(app.ExampleRequestError, "Auth token"):
            app.build_auth_metadata("")

    def test_create_channel_selects_secure_and_insecure(self):
        insecure = object()
        secure = object()
        with mock.patch("demo.voice_clone_web.app.grpc.insecure_channel", return_value=insecure) as insecure_channel:
            with mock.patch(
                "demo.voice_clone_web.app.grpc.secure_channel",
                return_value=secure,
            ) as secure_channel:
                with mock.patch(
                    "demo.voice_clone_web.app.grpc.ssl_channel_credentials",
                    return_value="creds",
                ) as ssl_credentials:
                    self.assertIs(app.create_channel(app.GrpcTarget("localhost:50051", False)), insecure)
                    self.assertIs(app.create_channel(app.GrpcTarget("voice.example:443", True)), secure)
        insecure_channel.assert_called_once_with("localhost:50051")
        ssl_credentials.assert_called_once_with()
        secure_channel.assert_called_once_with("voice.example:443", "creds")

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
        fake_voice = FakeVoiceStub(fake_channel)

        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=fake_artifacts,
        ) as artifact_stub_ctor:
            with mock.patch(
                "demo.voice_clone_web.app.voice_pb2_grpc.VoiceServiceStub",
                return_value=fake_voice,
            ) as voice_stub_ctor:
                with mock.patch(
                    "demo.voice_clone_web.app.normalise_recorded_audio",
                    return_value=(b"wav-bytes", "voice-sample.wav", "audio/wav"),
                ) as normalise:
                    result = app.synthesize_selected_reading(
                        dictator_url="https://voice.example:443",
                        auth_token="secret",
                        audio_payload=b"sample-bytes",
                        audio_filename="voice.webm",
                        audio_media_type="audio/webm",
                        render_preset_id="alice",
                        channel_factory=lambda target: fake_channel,
                    )

        artifact_stub_ctor.assert_called_once_with(fake_channel)
        voice_stub_ctor.assert_called_once_with(fake_channel)
        normalise.assert_called_once_with(b"sample-bytes", "voice.webm", "audio/webm")
        upload_chunks, upload_metadata = fake_artifacts.upload_calls[0]
        self.assertEqual(upload_chunks[0].metadata.filename, "voice-sample.wav")
        self.assertEqual(upload_chunks[0].metadata.media_type, "audio/wav")
        self.assertEqual(upload_metadata, [("authorization", "Bearer secret")])
        self.assertEqual(fake_voice.extract_calls, [])
        synth_request, synth_metadata = fake_voice.synthesize_calls[0]
        self.assertEqual(synth_request.speaker_artifact_id, "source-artifact")
        self.assertEqual(synth_request.synthesis_engine, voice_pb2.SYNTHESIS_ENGINE_XTTS)
        self.assertEqual(synth_request.speaker_transcript_text, "")
        self.assertIn("Alice was beginning to get very tired", synth_request.text)
        self.assertEqual(synth_metadata, [("authorization", "Bearer secret")])
        self.assertEqual(result.content, b"hello world")
        self.assertEqual(result.filename, "alice-in-your-voice.wav")
        self.assertTrue(fake_channel.closed)

    def test_synthesize_selected_reading_uses_sample_transcript_for_qwen3(self):
        fake_channel = FakeChannel()
        fake_artifacts = FakeArtifactStub(fake_channel)
        fake_voice = FakeVoiceStub(fake_channel)

        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=fake_artifacts,
        ):
            with mock.patch(
                "demo.voice_clone_web.app.voice_pb2_grpc.VoiceServiceStub",
                return_value=fake_voice,
            ):
                with mock.patch(
                    "demo.voice_clone_web.app.normalise_recorded_audio",
                    return_value=(b"wav-bytes", "voice-sample.wav", "audio/wav"),
                ):
                    app.synthesize_selected_reading(
                        dictator_url="dictator-grpc:50051",
                        auth_token="secret",
                        audio_payload=b"sample-bytes",
                        audio_filename="voice.webm",
                        audio_media_type="audio/webm",
                        synthesis_engine_id="qwen3",
                    )

        synth_request, _ = fake_voice.synthesize_calls[0]
        self.assertEqual(synth_request.synthesis_engine, voice_pb2.SYNTHESIS_ENGINE_QWEN3)
        self.assertEqual(synth_request.speaker_transcript_text, app.VOICE_SAMPLE_TEXT)

    def test_synthesize_selected_reading_uses_sample_transcript_for_cosyvoice3(self):
        fake_channel = FakeChannel()
        fake_artifacts = FakeArtifactStub(fake_channel)
        fake_voice = FakeVoiceStub(fake_channel)

        with mock.patch(
            "demo.voice_clone_web.app.artifacts_pb2_grpc.ArtifactServiceStub",
            return_value=fake_artifacts,
        ):
            with mock.patch(
                "demo.voice_clone_web.app.voice_pb2_grpc.VoiceServiceStub",
                return_value=fake_voice,
            ):
                with mock.patch(
                    "demo.voice_clone_web.app.normalise_recorded_audio",
                    return_value=(b"wav-bytes", "voice-sample.wav", "audio/wav"),
                ):
                    app.synthesize_selected_reading(
                        dictator_url="dictator-grpc:50051",
                        auth_token="secret",
                        audio_payload=b"sample-bytes",
                        audio_filename="voice.webm",
                        audio_media_type="audio/webm",
                        synthesis_engine_id="cosyvoice3",
                    )

        synth_request, _ = fake_voice.synthesize_calls[0]
        self.assertEqual(synth_request.synthesis_engine, voice_pb2.SYNTHESIS_ENGINE_COSYVOICE3)
        self.assertEqual(synth_request.speaker_transcript_text, app.VOICE_SAMPLE_TEXT)

    def test_synthesize_selected_reading_requires_audio(self):
        with self.assertRaisesRegex(app.ExampleRequestError, "voice sample"):
            app.synthesize_selected_reading(
                dictator_url="localhost:50051",
                auth_token="secret",
                audio_payload=b"",
                audio_filename="voice.webm",
                audio_media_type="audio/webm",
            )

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
        def fake_audio_to_wav(src, dst):
            self.assertEqual(src.read_bytes(), b"compressed-audio")
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
            "synthesisEngine": "qwen3",
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
            self.assertEqual(kwargs["synthesis_engine_id"], "qwen3")
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
