from __future__ import annotations

import base64
from contextlib import contextmanager
import http.client
import json
import threading
import unittest
from unittest import mock

import grpc

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
        self.assertIn("Read Genesis in your own cloned voice", html)
        self.assertIn("Eleven benevolent elephants", html)
        self.assertNotIn("Dictator gRPC URL", html)
        self.assertNotIn("Language Code", html)
        self.assertIn("Record your voice sample", html)

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

    def test_synthesize_genesis_reading_calls_service_flow(self):
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
                result = app.synthesize_genesis_reading(
                    dictator_url="https://voice.example:443",
                    auth_token="secret",
                    audio_payload=b"sample-bytes",
                    audio_filename="voice.webm",
                    audio_media_type="audio/webm",
                    channel_factory=lambda target: fake_channel,
                )

        artifact_stub_ctor.assert_called_once_with(fake_channel)
        voice_stub_ctor.assert_called_once_with(fake_channel)
        extract_request, extract_metadata = fake_voice.extract_calls[0]
        self.assertEqual(extract_request.source_artifact_id, "source-artifact")
        self.assertEqual(extract_request.language_code, "en")
        self.assertEqual(extract_metadata, [("authorization", "Bearer secret")])
        synth_request, synth_metadata = fake_voice.synthesize_calls[0]
        self.assertEqual(synth_request.speaker_artifact_id, "sample-artifact")
        self.assertIn("In the beginning God created", synth_request.text)
        self.assertEqual(synth_metadata, [("authorization", "Bearer secret")])
        self.assertEqual(result.content, b"hello world")
        self.assertTrue(fake_channel.closed)

    def test_synthesize_genesis_reading_requires_audio(self):
        with self.assertRaisesRegex(app.ExampleRequestError, "voice sample"):
            app.synthesize_genesis_reading(
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
            "authToken": "secret",
            "languageCode": "en",
            "audioBase64": base64.b64encode(b"sample").decode("ascii"),
            "audioFilename": "voice.webm",
            "audioMediaType": "audio/webm",
        }

        def synthesizer(**kwargs):
            self.assertEqual(kwargs["dictator_url"], "dictator-grpc:50051")
            self.assertEqual(kwargs["audio_payload"], b"sample")
            return app.VoiceCloneResult("demo.wav", "audio/wav", b"rendered")

        with running_server(
            app.build_handler(
                index_html="<html>ok</html>",
                synthesizer=synthesizer,
                default_dictator_url="dictator-grpc:50051",
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
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request(
                "POST",
                "/api/clone",
                body=json.dumps({
                    "authToken": "secret",
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
            )
        ) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port)
            connection.request(
                "POST",
                "/api/clone",
                body=json.dumps({
                    "authToken": "secret",
                    "audioBase64": base64.b64encode(b"sample").decode("ascii"),
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 502)
        self.assertIn("UNAUTHENTICATED", payload["error"])

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
