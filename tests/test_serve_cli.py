import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import serve


class ServeCliTests(unittest.TestCase):
    def test_main_logs_config_paths_and_starts_with_loaded_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            env_file = root / ".env"
            config_file.write_text(
                "\n".join(
                    [
                        "grpc:",
                        "  host: 127.0.0.1",
                        "  port: 55001",
                        "  artifact_root: .dictator-artifacts",
                        "  auth_token: ${DICTATOR_GRPC_AUTH_TOKEN}",
                    ]
                ),
                encoding="utf-8",
            )
            env_file.write_text(
                "\n".join(
                    [
                        "DICTATOR_GRPC_AUTH_TOKEN=secret",
                        "HF_TOKEN=hf-test-token",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with (
                patch(
                    "sys.argv",
                    [
                        "serve.py",
                        "--config",
                        str(config_file),
                        "--env-file",
                        str(env_file),
                    ],
                ),
                patch("serve.serve") as serve_mock,
                patch("serve.logging.info") as logging_info_mock,
                patch.dict("serve.os.environ", {}, clear=True),
            ):
                serve.main()
                exported_hf_token = serve.os.environ["HF_TOKEN"]

        serve_mock.assert_called_once()
        loaded_config = serve_mock.call_args.args[0]
        self.assertEqual(loaded_config.port, 55001)
        self.assertEqual(loaded_config.auth_token, "secret")
        self.assertEqual(exported_hf_token, "hf-test-token")
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "loading gRPC config from %s (exists=%s), env file %s (exists=%s)"
                for call in logging_info_mock.call_args_list
            )
        )
        self.assertTrue(
            any(
                call.args
                and call.args[0] == "loaded %d env vars into process environment"
                and call.args[1] == 2
                for call in logging_info_mock.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
