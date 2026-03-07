import tempfile
from pathlib import Path
import unittest

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from dictator.transport.grpc.config import ServerConfig
from dictator.transport.grpc.server import build_server


class GrpcServerTests(unittest.TestCase):
    def test_build_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = build_server(ServerConfig(artifact_root=Path(tmpdir)))
            self.assertIsNotNone(server)
            server.stop(None)

    def test_build_server_marks_aggregate_health_as_serving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = build_server(ServerConfig(artifact_root=Path(tmpdir)))
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    grpc.channel_ready_future(channel).result(timeout=2)
                    response = health_pb2_grpc.HealthStub(channel).Check(
                        health_pb2.HealthCheckRequest(service="")
                    )
                self.assertEqual(
                    response.status,
                    health_pb2.HealthCheckResponse.SERVING,
                )
            finally:
                server.stop(None)


if __name__ == "__main__":
    unittest.main()
