import unittest
from pathlib import Path
import tempfile

from dictator.transport.grpc.config import ServerConfig


class ServerConfigTests(unittest.TestCase):
    def test_from_env_overrides_defaults(self):
        config = ServerConfig.from_env(
            {
                "DICTATOR_GRPC_HOST": "127.0.0.1",
                "DICTATOR_GRPC_PORT": "55001",
                "DICTATOR_GRPC_MAX_WORKERS": "8",
                "DICTATOR_GRPC_MAX_MESSAGE_BYTES": "1234",
                "DICTATOR_GRPC_MAX_INFLIGHT": "6",
                "DICTATOR_GRPC_SYNTHESIS_JOB_WORKERS": "2",
                "DICTATOR_GRPC_MAX_PENDING_SYNTHESIS_JOBS": "16",
                "DICTATOR_GRPC_DOWNLOAD_CHUNK_BYTES": "2048",
                "DICTATOR_GRPC_ARTIFACT_ROOT": "~/dictator-artifacts",
                "DICTATOR_GRPC_AUTH_TOKEN": "secret",
            }
        )

        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 55001)
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.max_message_bytes, 1234)
        self.assertEqual(config.max_inflight, 6)
        self.assertEqual(config.synthesis_job_workers, 2)
        self.assertEqual(config.max_pending_synthesis_jobs, 16)
        self.assertEqual(config.download_chunk_bytes, 2048)
        self.assertEqual(config.artifact_root, Path("~/dictator-artifacts").expanduser())
        self.assertEqual(config.auth_token, "secret")

    def test_from_sources_reads_config_file_with_placeholder_substitution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            config_file.write_text(
                "\n".join(
                    [
                        "grpc:",
                        "  host: 127.0.0.1",
                        "  port: 55001",
                        "  max_workers: 8",
                        "  max_message_bytes: 1234",
                        "  max_inflight: 6",
                        "  synthesis_job_workers: 2",
                        "  max_pending_synthesis_jobs: 16",
                        "  download_chunk_bytes: 2048",
                        "  artifact_root: ~/dictator-artifacts",
                        "  auth_token: ${DICTATOR_GRPC_AUTH_TOKEN}",
                    ]
                ),
                encoding="utf-8",
            )
            config = ServerConfig.from_sources(
                config_file=config_file,
                env={"DICTATOR_GRPC_AUTH_TOKEN": "secret"},
            )

        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 55001)
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.max_message_bytes, 1234)
        self.assertEqual(config.max_inflight, 6)
        self.assertEqual(config.synthesis_job_workers, 2)
        self.assertEqual(config.max_pending_synthesis_jobs, 16)
        self.assertEqual(config.download_chunk_bytes, 2048)
        self.assertEqual(config.artifact_root, Path("~/dictator-artifacts").expanduser())
        self.assertEqual(config.auth_token, "secret")

    def test_from_sources_ignores_environment_overrides_not_declared_in_file(self):
        config = ServerConfig.from_sources(
            config_file=None,
            env={
                "DICTATOR_GRPC_HOST": "127.0.0.1",
                "DICTATOR_GRPC_PORT": "55001",
                "DICTATOR_GRPC_AUTH_TOKEN": "secret",
            },
        )

        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 50051)
        self.assertIsNone(config.auth_token)

    def test_from_sources_raises_for_missing_placeholder_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            config_file.write_text(
                "\n".join(
                    [
                        "grpc:",
                        "  auth_token: ${DICTATOR_GRPC_AUTH_TOKEN}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "DICTATOR_GRPC_AUTH_TOKEN"):
                ServerConfig.from_sources(
                    config_file=config_file,
                    env={},
                )

    def test_from_sources_reads_placeholder_from_supplied_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            config_file.write_text(
                "\n".join(
                    [
                        "grpc:",
                        "  auth_token: ${DICTATOR_GRPC_AUTH_TOKEN}",
                    ]
                ),
                encoding="utf-8",
            )

            config = ServerConfig.from_sources(
                config_file=config_file,
                env={"DICTATOR_GRPC_AUTH_TOKEN": "secret"},
            )

        self.assertEqual(config.auth_token, "secret")


if __name__ == "__main__":
    unittest.main()
