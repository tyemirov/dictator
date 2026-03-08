from __future__ import annotations

import importlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import grpc
from google.protobuf import struct_pb2

from dictator.client.diarization import DiarizationClient
from dictator.client.dictation import DictationClient
from dictator.diarization.models import DiarizedUtterance, DiarizedWord
from dictator.transport.grpc import config as grpc_config
from dictator.transport.grpc.config import ServerConfig
from dictator.transport.grpc.server import serve as serve_grpc


class ClientConfigServerCoverageTests(unittest.TestCase):
    def test_dictation_client_file_and_conflict_paths(self):
        channel = object()
        with (
            patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch("dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub", return_value=object()),
        ):
            client = DictationClient(channel)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(b"abc")
            with patch.object(client, "dictate_bytes", return_value="ok") as dictate_bytes_mock:
                self.assertEqual(client.dictate_file(audio_path, media_type="audio/wav"), "ok")
            dictate_bytes_mock.assert_called_once()
            self.assertEqual(dictate_bytes_mock.call_args.kwargs["filename"], "audio.wav")

        with self.assertRaisesRegex(ValueError, "cannot both be set"):
            DictationClient._resolve_autodetect(language_code="en", autodetect_language=True)

    def test_diarization_client_file_and_gap_field_paths(self):
        response = types.SimpleNamespace(
            text="hello",
            language_code="en",
            diarization=struct_pb2.Struct(),
            diarization_artifact_id="json1",
        )
        with (
            patch("dictator.client.diarization.artifacts_pb2_grpc.ArtifactServiceStub", return_value=object()),
            patch(
                "dictator.client.diarization.transcription_pb2_grpc.TranscriptionServiceStub",
                return_value=types.SimpleNamespace(DiarizeAudio=MagicMock(return_value=response)),
            ),
            patch("dictator.client.diarization.upload_audio_artifact", return_value=types.SimpleNamespace(artifact_id="audio1")),
            patch("dictator.client.diarization.MessageToDict", return_value={"text": "hello"}),
        ):
            client = DiarizationClient(object())
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = Path(tmpdir) / "audio.wav"
                audio_path.write_bytes(b"abc")
                file_result = client.diarize_file(audio_path, autodetect_language=True)
                self.assertEqual(file_result.source_artifact_id, "audio1")

            result = client.diarize_bytes(
                b"abc",
                filename="audio.wav",
                media_type="audio/wav",
                autodetect_language=True,
                utterance_gap_seconds=1.25,
            )
            self.assertEqual(result.diarization_artifact_id, "json1")
            request = client._transcription_stub.DiarizeAudio.call_args.args[0]
            self.assertTrue(request.HasField("utterance_gap_seconds"))
            self.assertAlmostEqual(request.utterance_gap_seconds, 1.25)

    def test_diarized_models_and_utterance_without_words(self):
        word = DiarizedWord("hello", 0.0, 0.4, "S1")
        self.assertEqual(
            word.to_legacy_dict(),
            {"content": "hello", "speaker": "S1", "start": 0.0, "end": 0.4},
        )
        utterance = DiarizedUtterance("S1", 0.0, 0.4, "hello", (word,))
        self.assertEqual(
            utterance.to_json_dict(include_words=False),
            {"speaker": "S1", "start": 0.0, "end": 0.4, "text": "hello"},
        )

    def test_config_private_helpers_and_overlay(self):
        self.assertEqual(grpc_config._parse_scalar(""), "")
        self.assertEqual(grpc_config._parse_scalar("'quoted'"), "quoted")
        self.assertEqual(grpc_config._parse_scalar("true"), True)
        self.assertEqual(grpc_config._parse_scalar("false"), False)
        self.assertEqual(grpc_config._parse_scalar("12"), 12)
        self.assertEqual(grpc_config._parse_scalar("1.5"), 1.5)
        self.assertEqual(grpc_config._parse_scalar("abc"), "abc")
        with self.assertRaisesRegex(ValueError, "positive"):
            grpc_config._parse_positive_int("0", "port")

        self.assertEqual(grpc_config._resolve_env_placeholders(5, {}), 5)
        self.assertEqual(grpc_config._resolve_env_placeholders("${X}", {"X": "ok"}), "ok")
        with self.assertRaisesRegex(ValueError, "missing required env var X"):
            grpc_config._resolve_env_placeholders("${X}", {})

        config = ServerConfig().overlay(host="", auth_token="", artifact_root="~/artifacts")
        self.assertEqual(config.host, grpc_config.DEFAULT_HOST)
        self.assertEqual(config.auth_token, None)
        self.assertEqual(config.artifact_root, Path("~/artifacts").expanduser())

    def test_load_config_mapping_edge_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            config_file.write_text(
                "grpc:\n"
                "  host: 127.0.0.1\n"
                "  port: 5000\n"
                "\n"
                "# comment\n"
                "other:\n"
                "  ignored: true\n",
                encoding="utf-8",
            )
            self.assertEqual(
                grpc_config._load_config_mapping(config_file, {}),
                {"host": "127.0.0.1", "port": 5000},
            )
            config_file.write_text(" bad: yes\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid indentation"):
                grpc_config._load_config_mapping(config_file, {})
            config_file.write_text("badline\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid config line"):
                grpc_config._load_config_mapping(config_file, {})

    def test_server_serve_calls_build_start_and_wait(self):
        fake_server = MagicMock()
        with (
            patch("dictator.transport.grpc.server.build_server", return_value=fake_server) as build_mock,
            patch("dictator.transport.grpc.server.logging.info") as info_mock,
        ):
            config = ServerConfig(host="127.0.0.1", port=50051)
            serve_grpc(config)
        build_mock.assert_called_once_with(config)
        fake_server.add_insecure_port.assert_called_once_with("127.0.0.1:50051")
        fake_server.start.assert_called_once()
        fake_server.wait_for_termination.assert_called_once()
        info_mock.assert_called_once()

    def test_serve_cli_raises_when_token_missing(self):
        import serve

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            config_file.write_text("grpc:\n  host: 127.0.0.1\n", encoding="utf-8")
            with (
                patch("sys.argv", ["serve.py", "--config", str(config_file)]),
                patch("serve.serve"),
                patch.dict("dictator.transport.grpc.config.os.environ", {}, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "auth token must be configured"):
                    serve.main()


if __name__ == "__main__":
    unittest.main()
