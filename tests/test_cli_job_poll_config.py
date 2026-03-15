from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from dictator.transport.grpc.config import ServerConfig

import align
import dictate
import subtitle


class CliJobPollConfigTests(unittest.TestCase):
    def test_server_config_loads_job_wait_defaults_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yml"
            config_file.write_text(
                "grpc:\n"
                "  host: 127.0.0.1\n"
                "  port: 5000\n"
                "  job_wait_timeout_seconds: 12.5\n"
                "  job_poll_interval_seconds: 0.25\n",
                encoding="utf-8",
            )
            config = ServerConfig.from_sources(config_file=config_file, env={})
        self.assertEqual(config.job_wait_timeout_seconds, 12.5)
        self.assertEqual(config.job_poll_interval_seconds, 0.25)

    def test_dictate_main_uses_job_polling_values_from_config(self):
        base_config = ServerConfig(
            host="0.0.0.0",
            port=50051,
            auth_token="secret",
            job_wait_timeout_seconds=12.5,
            job_poll_interval_seconds=0.25,
        )
        fake_result = type(
            "Result",
            (),
            {
                "text": "hello",
                "words": (),
                "to_http_payload": lambda self: {"text": "hello"},
            },
        )()
        channel_cm = MagicMock()
        channel_cm.__enter__.return_value = "channel"
        with (
            patch("sys.argv", ["dictate.py", "--input", "audio.wav"]),
            patch("dictate.ServerConfig.from_sources", return_value=base_config),
            patch("dictate.grpc.insecure_channel", return_value=channel_cm),
            patch("dictate.DictationClient") as client_cls,
            patch("builtins.print") as print_mock,
        ):
            client_cls.return_value.dictate_file.return_value = fake_result
            dictate.main()
        self.assertEqual(json.loads(print_mock.call_args.args[0]), {"text": "hello"})
        self.assertEqual(client_cls.return_value.dictate_file.call_args.kwargs["timeout_seconds"], 12.5)
        self.assertEqual(client_cls.return_value.dictate_file.call_args.kwargs["poll_interval_seconds"], 0.25)

    def test_subtitle_main_uses_job_polling_values_from_config(self):
        base_config = ServerConfig(
            host="0.0.0.0",
            port=50051,
            auth_token="secret",
            job_wait_timeout_seconds=45.0,
            job_poll_interval_seconds=0.5,
        )
        fake_result = type(
            "Result",
            (),
            {
                "language_code": "en",
                "mode": "transcription",
                "srt_artifact_id": "artifact-1",
                "srt_text": "1\n00:00:00,000 --> 00:00:00,400\nhello\n",
            },
        )()
        channel_cm = MagicMock()
        channel_cm.__enter__.return_value = "channel"
        with (
            patch("sys.argv", ["subtitle.py", "--input", "audio.wav"]),
            patch("subtitle.ServerConfig.from_sources", return_value=base_config),
            patch("subtitle.grpc.insecure_channel", return_value=channel_cm),
            patch("subtitle.SubtitleClient") as client_cls,
            patch("builtins.print"),
        ):
            client_cls.return_value.render_file.return_value = fake_result
            subtitle.main()
        self.assertEqual(client_cls.return_value.render_file.call_args.kwargs["timeout_seconds"], 45.0)
        self.assertEqual(client_cls.return_value.render_file.call_args.kwargs["poll_interval_seconds"], 0.5)

    def test_align_main_uses_job_polling_values_from_config(self):
        base_config = ServerConfig(
            host="0.0.0.0",
            port=50051,
            auth_token="secret",
            job_wait_timeout_seconds=90.0,
            job_poll_interval_seconds=2.0,
        )
        fake_result = type(
            "Result",
            (),
            {
                "language_code": "en",
                "words": (1, 2),
                "srt_artifact_id": "artifact-2",
                "srt_text": "1\n00:00:00,000 --> 00:00:00,400\nhello world\n",
            },
        )()
        channel_cm = MagicMock()
        channel_cm.__enter__.return_value = "channel"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.wav"
            text_path = root / "transcript.txt"
            output_path = root / "out.srt"
            input_path.write_bytes(b"wav")
            text_path.write_text("hello world", encoding="utf-8")
            with (
                patch("sys.argv", ["align.py", "--input", str(input_path), "--text", str(text_path), "--output", str(output_path)]),
                patch("align.ServerConfig.from_sources", return_value=base_config),
                patch("align.grpc.insecure_channel", return_value=channel_cm),
                patch("align.AlignmentClient") as client_cls,
            ):
                client_cls.return_value.align_file.return_value = fake_result
                align.main()
        self.assertEqual(client_cls.return_value.align_file.call_args.kwargs["timeout_seconds"], 90.0)
        self.assertEqual(client_cls.return_value.align_file.call_args.kwargs["poll_interval_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
