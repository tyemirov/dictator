import unittest
from pathlib import Path

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
        self.assertEqual(config.download_chunk_bytes, 2048)
        self.assertEqual(config.artifact_root, Path("~/dictator-artifacts").expanduser())
        self.assertEqual(config.auth_token, "secret")


if __name__ == "__main__":
    unittest.main()
