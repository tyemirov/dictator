import tempfile
from pathlib import Path
import unittest

from dictator.transport.grpc.config import ServerConfig
from dictator.transport.grpc.server import build_server


class GrpcServerTests(unittest.TestCase):
    def test_build_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = build_server(ServerConfig(artifact_root=Path(tmpdir)))
            self.assertIsNotNone(server)
            server.stop(None)


if __name__ == "__main__":
    unittest.main()
