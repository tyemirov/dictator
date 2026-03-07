import time
import unittest

from dictator.runtime import run_with_timeout


class TimeoutHelperTests(unittest.TestCase):
    def test_returns_result_when_in_time(self):
        result = run_with_timeout(1.0, "quick", lambda: 42)
        self.assertEqual(result, 42)

    def test_raises_timeout_error(self):
        with self.assertRaisesRegex(TimeoutError, "slow exceeded 0.01s"):
            run_with_timeout(0.01, "slow", time.sleep, 0.05)


if __name__ == "__main__":
    unittest.main()
