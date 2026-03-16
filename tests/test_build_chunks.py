import sys
import types

# Provide dummy modules so importing main doesn't require heavy deps
sys.modules['ffmpeg'] = types.ModuleType('ffmpeg')
sys.modules['soundfile'] = types.ModuleType('soundfile')
sys.modules['torch'] = types.ModuleType('torch')

import unittest

from dictator.synthesis.text import clean, join_synthesis_units, parse_length, split_into_sentences


class TestSynthesisTextUtils(unittest.TestCase):
    def test_clean_normalizes_and_cleans_whitespace(self):
        self.assertEqual(clean("A\x00  B\nC"), "A B C")
        self.assertEqual(clean("Though;\nHe"), "Though; He")

    def test_split_into_sentences_keeps_terminal_punctuation(self):
        self.assertEqual(
            split_into_sentences("Hello. Again? Last!"),
            ["Hello.", "Again?", "Last!"],
        )

    def test_join_synthesis_units_joins_with_strong_separators_and_rejects_empty(self):
        self.assertEqual(join_synthesis_units(("Hello.", "Again?")), "Hello.\n\nAgain?")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            join_synthesis_units(())

    def test_parse_length_supports_units(self):
        self.assertEqual(parse_length("90s"), 90.0)
        self.assertEqual(parse_length("2m"), 120.0)
        self.assertEqual(parse_length("1.5h"), 5400.0)
        with self.assertRaisesRegex(ValueError, "--length"):
            parse_length("later")


if __name__ == "__main__":
    unittest.main()
