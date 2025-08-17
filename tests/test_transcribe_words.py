import sys
import types
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch


class DummyModel:
    def __init__(self, result):
        self._result = result

    def transcribe(self, audio_path, language=None, word_timestamps=False, verbose=False):
        self.called_with = {
            'audio_path': audio_path,
            'language': language,
            'word_timestamps': word_timestamps,
            'verbose': verbose,
        }
        return self._result


class TranscribeWordsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # stub heavy dependencies before importing service
        modules = {
            'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
            'whisper': types.SimpleNamespace(),
        }
        cls.patcher = patch.dict(sys.modules, modules)
        cls.patcher.start()
        import whisper_service as ws
        cls.ws = importlib.reload(ws)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_transcribe_words_uses_model_output(self):
        dummy_result = {
            'segments': [
                {'words': [
                    {'word': 'Hello', 'start': 0.0, 'end': 0.4},
                    {'word': 'world', 'start': 0.4, 'end': 0.9},
                ]}
            ]
        }
        model = DummyModel(dummy_result)
        segments = self.ws.transcribe_words(Path('dummy.wav'), 'en', model=model)
        self.assertEqual(
            segments,
            [
                {'content': 'Hello', 'start': 0.0, 'end': 0.4, 'probability': None},
                {'content': 'world', 'start': 0.4, 'end': 0.9, 'probability': None},
            ],
        )
        self.assertEqual(model.called_with['language'], 'en')
        self.assertTrue(model.called_with['word_timestamps'])
        self.assertFalse(model.called_with['verbose'])


if __name__ == '__main__':
    unittest.main()
