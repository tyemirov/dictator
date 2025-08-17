import sys
import types
from pathlib import Path

import pytest

# stub heavy dependencies so main imports without optional packages
if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch

if 'TTS' not in sys.modules:
    tts_api = types.ModuleType('TTS.api')
    class DummyTTS:
        def to(self, device):
            return self
        def tts_to_file(self, **kwargs):
            return None
    tts_api.TTS = DummyTTS
    tts_pkg = types.ModuleType('TTS')
    tts_pkg.api = tts_api
    sys.modules['TTS'] = tts_pkg
    sys.modules['TTS.api'] = tts_api

if 'soundfile' not in sys.modules:
    sf = types.ModuleType('soundfile')
    class _Info:
        frames = 0
        samplerate = 1
    def info(path):
        return _Info()
    sf.info = info
    sys.modules['soundfile'] = sf

# ffmpeg is imported but unused in this test
if 'ffmpeg' not in sys.modules:
    sys.modules['ffmpeg'] = types.ModuleType('ffmpeg')

import main


def test_synthesise_empty_chunks_raises_value_error():
    with pytest.raises(ValueError):
        main.synthesise(Path('speaker.wav'), [], 10.0, 'en')
