import unittest

from duration import parse_duration


class TestDuration(unittest.TestCase):
    def test_duration_seconds_suffix(self):
        self.assertEqual(parse_duration("60s"), 60.0)

    def test_duration_minutes_suffix(self):
        self.assertEqual(parse_duration("1m"), 60.0)

    def test_duration_no_suffix(self):
        self.assertEqual(parse_duration("15"), 15.0)


if __name__ == "__main__":
    unittest.main()
