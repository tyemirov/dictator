import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from dictator.transport.grpc.config import ServerConfig
import subtitle


class SubtitleCliTests(unittest.TestCase):
    def test_default_target_uses_localhost_for_wildcard_bind(self):
        target = subtitle._default_target_from_config(
            ServerConfig(host="0.0.0.0", port=50051)
        )
        self.assertEqual(target, "127.0.0.1:50051")

    def test_default_target_uses_explicit_host(self):
        target = subtitle._default_target_from_config(
            ServerConfig(host="10.0.0.7", port=55001)
        )
        self.assertEqual(target, "10.0.0.7:55001")

    def test_main_prints_srt_to_stdout(self):
        base_config = ServerConfig(host="0.0.0.0", port=50051, auth_token="secret")
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
            patch("builtins.print") as print_mock,
        ):
            client_cls.return_value.render_file.return_value = fake_result
            subtitle.main()
        client_cls.return_value.render_file.assert_called_once()
        self.assertEqual(print_mock.call_args.kwargs["end"], "")
        self.assertEqual(print_mock.call_args.args[0], fake_result.srt_text)

    def test_main_writes_output_file_and_prints_metadata(self):
        base_config = ServerConfig(host="127.0.0.1", port=50051, auth_token="secret")
        fake_result = type(
            "Result",
            (),
            {
                "language_code": "en",
                "mode": "forced_alignment",
                "srt_artifact_id": "artifact-2",
                "srt_text": "1\n00:00:00,000 --> 00:00:00,400\nhello world\n",
            },
        )()
        channel_cm = MagicMock()
        channel_cm.__enter__.return_value = "channel"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "transcript.txt"
            output = root / "out.srt"
            transcript.write_text("hello world", encoding="utf-8")
            with (
                patch(
                    "sys.argv",
                    [
                        "subtitle.py",
                        "--input",
                        "audio.wav",
                        "--language",
                        "en",
                        "--granularity",
                        "sentences",
                        "--group-size",
                        "2",
                        "--source-text-file",
                        str(transcript),
                        "--source-text-name",
                        "scene.txt",
                        "--output",
                        str(output),
                    ],
                ),
                patch("subtitle.ServerConfig.from_sources", return_value=base_config),
                patch("subtitle.grpc.insecure_channel", return_value=channel_cm),
                patch("subtitle.SubtitleClient") as client_cls,
                patch("builtins.print") as print_mock,
            ):
                client_cls.return_value.render_file.return_value = fake_result
                subtitle.main()
            payload = json.loads(print_mock.call_args.args[0])
            self.assertEqual(payload["mode"], "forced_alignment")
            self.assertEqual(payload["srtArtifactId"], "artifact-2")
            self.assertEqual(payload["output"], str(output))
            self.assertEqual(output.read_text(encoding="utf-8"), fake_result.srt_text)
            self.assertEqual(
                client_cls.return_value.render_file.call_args.kwargs["source_text_file"],
                transcript,
            )
            self.assertEqual(
                client_cls.return_value.render_file.call_args.kwargs["source_text_name"],
                "scene.txt",
            )
            self.assertFalse(
                client_cls.return_value.render_file.call_args.kwargs["autodetect_language"]
            )


if __name__ == "__main__":
    unittest.main()
