
import unittest
import tempfile
from pathlib import Path
from dictator.transport.grpc.config import (
    _strip_inline_comment,
    _validate_config_mapping,
    _lookup_mapping_value,
    _load_config_mapping,
)

class ExtraConfigCoverageTests(unittest.TestCase):
    def test_strip_inline_comment_edge_cases(self):
        # Line 197: if not raw_value
        self.assertEqual(_strip_inline_comment(""), "")
        
        # Lines 201-204: quote toggling
        self.assertEqual(_strip_inline_comment("'quoted # comment'"), "'quoted # comment'")
        self.assertEqual(_strip_inline_comment('"double # quoted"'), '"double # quoted"')
        
        # Line 206: comment after quote
        self.assertEqual(_strip_inline_comment('"quoted" # comment'), '"quoted"')

    def test_validate_config_mapping_edge_cases(self):
        # Line 234: unknown config key
        with self.assertRaisesRegex(ValueError, "unknown config key server.unknown"):
            _validate_config_mapping({"server": {"unknown": 1}}, schema={"server": {"listen": {}}})
            
        # Lines 238-239: mapping expected but got scalar
        schema = {"server": {"listen": {"host": str}}}
        with self.assertRaisesRegex(ValueError, "config key server.listen must be a mapping"):
            _validate_config_mapping({"server": {"listen": "not-a-mapping"}}, schema=schema)
            
        # Line 242: scalar expected but got mapping
        with self.assertRaisesRegex(ValueError, "config key server.listen.host must be a scalar"):
             _validate_config_mapping({"server": {"listen": {"host": {"nested": 1}}}}, schema=schema)

        # Line 249: multiple types in schema (e.g. (int, float))
        # Trigger the "else" branch of the expected label generation
        # We need a schema where child_schema is not a type (it's a tuple of types)
        # AND value is not an instance of any of those types.
        schema = {"execution": {"jobs": {"wait_timeout_seconds": (int, float)}}}
        with self.assertRaisesRegex(ValueError, "config key execution.jobs.wait_timeout_seconds must be int or float, got str"):
            _validate_config_mapping({"execution": {"jobs": {"wait_timeout_seconds": "high"}}, "server": {}}, schema=schema)

        # Line 255-258: strict bool check for single type (int)
        schema = {"server": {"listen": {"port": int}}}
        with self.assertRaisesRegex(ValueError, "config key server.listen.port must be int, got bool"):
            _validate_config_mapping({"server": {"listen": {"port": True}}}, schema=schema)

        # Line 263-267: strict bool check for tuple of types (int, float)
        schema = {"execution": {"jobs": {"wait_timeout_seconds": (int, float)}}}
        with self.assertRaisesRegex(ValueError, "config key execution.jobs.wait_timeout_seconds must be int or float, got bool"):
            _validate_config_mapping({"execution": {"jobs": {"wait_timeout_seconds": True}}}, schema=schema)

    def test_lookup_mapping_value_edge_cases(self):
        # Line 275: current not a dict or key not in current
        mapping = {"server": {"listen": {"host": "0.0.0.0"}}}
        val, found = _lookup_mapping_value(mapping, ("server", "listen", "port"))
        self.assertFalse(found)
        
        val, found = _lookup_mapping_value(mapping, ("server", "not-here", "host"))
        self.assertFalse(found)

    def test_load_config_mapping_empty(self):
        # Line 315: if not root
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "empty.yml"
            config_file.write_text("# just a comment\n\n", encoding="utf-8")
            self.assertEqual(_load_config_mapping(config_file, {}), {})

if __name__ == "__main__":
    unittest.main()
