import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import serve


class ServeCliTests(unittest.TestCase):
    def test_main_loads_config_file_as_single_service_config_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            config_file.write_text(
                "\n".join(
                    [
                        "grpc:",
                        "  host: 127.0.0.1",
                        "  auth_token: ${DICTATOR_GRPC_AUTH_TOKEN}",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("sys.argv", ["serve.py", "--config", str(config_file)]), patch(
                "serve.serve"
            ) as serve_mock, patch("serve.logging.info") as logging_info_mock, patch.dict(
                "dictator.transport.grpc.config.os.environ",
                {"DICTATOR_GRPC_AUTH_TOKEN": "secret"},
                clear=True,
            ):
                serve.main()

        serve_mock.assert_called_once()
        loaded_config = serve_mock.call_args.args[0]
        self.assertEqual(loaded_config.auth_token, "secret")
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "loading gRPC config from %s (exists=%s)"
                for call in logging_info_mock.call_args_list
            )
        )

    def test_main_rejects_removed_cli_overrides(self):
        with self.assertRaises(SystemExit):
            with patch("sys.argv", ["serve.py", "--host", "127.0.0.1"]):
                serve.main()


if __name__ == "__main__":
    unittest.main()
