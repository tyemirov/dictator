import sys
import types
from pathlib import Path
import unittest


# stub heavy dependencies so main imports without optional packages
torch = types.ModuleType('torch')
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules['torch'] = torch

tts_api = types.ModuleType('TTS.api')
class DummyTTS:
    def __init__(self, *args, **kwargs):
        pass
    def to(self, device):
        return self
    def tts_to_file(self, **kwargs):
        return None
tts_api.TTS = DummyTTS
tts_pkg = types.ModuleType('TTS')
tts_pkg.api = tts_api
sys.modules['TTS'] = tts_pkg
sys.modules['TTS.api'] = tts_api

sf = types.ModuleType('soundfile')
class _Info:
    frames = 24000
    samplerate = 24000
def info(path):
    return _Info()
sf.info = info
sys.modules['soundfile'] = sf

# ffmpeg is imported but unused in this test
sys.modules['ffmpeg'] = types.ModuleType('ffmpeg')

import importlib
import main as _main
main = importlib.reload(_main)


class TestSynthesise(unittest.TestCase):
    def test_synthesise_empty_chunks_raises_value_error(self):
        with self.assertRaises(ValueError):
            main.synthesise(Path('speaker.wav'), [], 10.0, 'en')

    def test_synthesise_returns_timeline(self):
        paths, segments = main.synthesise(Path('speaker.wav'), ['one', 'two'], None, 'en')
        self.assertEqual(len(paths), 2)
        expected = [
            {'text': 'one', 'start': 0.0, 'end': 1.0},
            {'text': 'two', 'start': 1.0, 'end': 2.0},
        ]
        self.assertEqual(segments, expected)
        for seg in segments:
            self.assertEqual(set(seg.keys()), {'text', 'start', 'end'})


if __name__ == "__main__":
    unittest.main()
