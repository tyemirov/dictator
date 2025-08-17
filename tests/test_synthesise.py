import sys
import types
from pathlib import Path

import pytest

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


def test_synthesise_empty_chunks_raises_value_error():
    with pytest.raises(ValueError):
        main.synthesise(Path('speaker.wav'), [], 10.0, 'en')


def test_synthesise_returns_timeline():
    paths, segments = main.synthesise(Path('speaker.wav'), ['one', 'two'], None, 'en')
    assert len(paths) == 2
    assert segments == [
        {'text': 'one', 'start': 0.0, 'end': 1.0},
        {'text': 'two', 'start': 1.0, 'end': 2.0},
    ]
