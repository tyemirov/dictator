import sys
import types
from pathlib import Path
import importlib
import unittest
from unittest.mock import patch, MagicMock


class DummyTTS:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def tts_to_file(self, **kwargs):
        return None


class SynthesiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sf_module = types.SimpleNamespace(
            info=lambda path: types.SimpleNamespace(frames=24000, samplerate=24000)
        )

        cls.patcher = patch.dict(
            sys.modules,
            {
                'torch': types.SimpleNamespace(
                    cuda=types.SimpleNamespace(is_available=lambda: False)
                ),
                'TTS': types.SimpleNamespace(api=types.SimpleNamespace(TTS=DummyTTS)),
                'TTS.api': types.SimpleNamespace(TTS=DummyTTS),
                'soundfile': sf_module,
                'ffmpeg': MagicMock(),
            },
        )
        cls.patcher.start()
        import main as _main

        cls.main = importlib.reload(_main)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_synthesise_empty_chunks_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.main.synthesise(Path('speaker.wav'), [], 10.0, 'en')

    def test_synthesise_returns_timeline(self):
        paths, segments = self.main.synthesise(Path('speaker.wav'), ['one', 'two'], None, 'en')
        self.assertEqual(len(paths), 2)
        self.assertEqual(
            segments,
            [
                {'content': 'one', 'start': 0.0, 'end': 1.0},
                {'content': 'two', 'start': 1.0, 'end': 2.0},
            ],
        )
