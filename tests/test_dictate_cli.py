import unittest

from dictator.transport.grpc.config import ServerConfig
import dictate
from dictator.client import DictationClient


class DictateCliTests(unittest.TestCase):
    def test_default_target_uses_localhost_for_wildcard_bind(self):
        target = dictate._default_target_from_config(
            ServerConfig(host="0.0.0.0", port=50051)
        )

        self.assertEqual(target, "127.0.0.1:50051")

    def test_default_target_uses_explicit_host(self):
        target = dictate._default_target_from_config(
            ServerConfig(host="10.0.0.5", port=55001)
        )

        self.assertEqual(target, "10.0.0.5:55001")

    def test_dictation_client_autodetect_resolution_defaults_from_empty_language(self):
        self.assertTrue(
            DictationClient._resolve_autodetect(
                language_code="",
                autodetect_language=None,
            )
        )

    def test_dictation_client_autodetect_resolution_rejects_missing_language_mode(self):
        with self.assertRaisesRegex(ValueError, "language_code or autodetect_language"):
            DictationClient._resolve_autodetect(
                language_code="",
                autodetect_language=False,
            )


if __name__ == "__main__":
    unittest.main()
