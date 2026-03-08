from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dictator.transport.grpc.config import (
    ServerConfig,
    _load_config_mapping,
    _parse_positive_int,
    _parse_scalar,
    _resolve_env_placeholders,
)
from dictator.transport.grpc.server import serve


class GrpcConfigServerCoverageTests(unittest.TestCase):
    def test_config_helper_functions_cover_error_branches(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _parse_positive_int("0", "port")
        self.assertEqual(_parse_scalar(""), "")
        self.assertEqual(_parse_scalar(" 'hello' "), "hello")
        self.assertIs(_parse_scalar("true"), True)
        self.assertIs(_parse_scalar("FALSE"), False)
        self.assertEqual(_parse_scalar("123"), 123)
        self.assertEqual(_parse_scalar("1.5"), 1.5)
        self.assertEqual(_parse_scalar("text"), "text")
        self.assertEqual(_resolve_env_placeholders(12, {}), 12)
        self.assertEqual(_resolve_env_placeholders("${TOKEN}", {"TOKEN": "secret"}), "secret")
        with self.assertRaisesRegex(ValueError, "missing required env var TOKEN"):
            _resolve_env_placeholders("${TOKEN}", {})

    def test_config_mapping_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yml"
            config_file.write_text("host: 127.0.0.1\nport: 50052\nunknown: keep\n", encoding="utf-8")
            self.assertEqual(_load_config_mapping(config_file, {}), {"host": "127.0.0.1", "port": 50052})

            invalid_indent = root / "bad-indent.yml"
            invalid_indent.write_text("grpc:\n   host: 127.0.0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid indentation"):
                _load_config_mapping(invalid_indent, {})

            invalid_line = root / "bad-line.yml"
            invalid_line.write_text("grpc\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid config line"):
                _load_config_mapping(invalid_line, {})

            self.assertEqual(_load_config_mapping(root / "missing.yml", {}), {})

    def test_server_config_overlay_and_serve(self):
        config = ServerConfig().overlay(host="", port=55001, auth_token=" ")
        self.assertEqual(config.host, ServerConfig().host)
        self.assertEqual(config.port, 55001)
        self.assertIsNone(config.auth_token)

        fake_server = Mock()
        with patch("dictator.transport.grpc.server.build_server", return_value=fake_server) as build_server, patch("dictator.transport.grpc.server.logging.info") as log_info:
            serve(ServerConfig(host="127.0.0.1", port=55001))
        build_server.assert_called_once()
        fake_server.add_insecure_port.assert_called_once_with("127.0.0.1:55001")
        fake_server.start.assert_called_once()
        fake_server.wait_for_termination.assert_called_once()
        log_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
