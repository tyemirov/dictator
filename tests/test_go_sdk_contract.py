import json
from pathlib import Path
import subprocess
import sys
import unittest


class GoSDKContractTests(unittest.TestCase):
    def test_declared_sdk_compiles_in_separate_consumer(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "scripts/verify_go_sdk.py", "--source", "local"],
            cwd=repository_root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        result = json.loads(completed.stdout)
        self.assertEqual(result["module"], "github.com/tyemirov/dictator/sdk/go/dictatorspeechv1")
        self.assertEqual(result["source"], "local")
        self.assertEqual(result["fields"], {
            "preset_speaker": "baya",
            "text_format": "SYNTHESIS_TEXT_FORMAT_SSML",
            "input_audio_usage": {
                "sample_count": 32001,
                "sample_rate_hz": 16000,
                "response_count": 9,
            },
        })


if __name__ == "__main__":
    unittest.main()
