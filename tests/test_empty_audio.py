import sys
import types
import importlib
import unittest
from unittest.mock import patch

import numpy as np


class TestEmptyAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = {
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
            "whisper": types.SimpleNamespace(),
        }
        cls.patcher = patch.dict(sys.modules, modules)
        cls.patcher.start()
        import whisper_service as ws
        cls.ws = importlib.reload(ws)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_empty_audio_raises(self):
        pcm = np.array([], dtype=np.int16)
        dummy_model = types.SimpleNamespace(transcribe=lambda *a, **k: {"segments": []})
        with self.assertRaisesRegex(ValueError, "audio array is empty"):
            self.ws.transcribe_words(pcm, model=dummy_model)


if __name__ == "__main__":
    unittest.main()

