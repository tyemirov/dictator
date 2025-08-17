import sys
import types
import importlib
import unittest
from unittest.mock import patch

import numpy as np


def _stub_modules():
    torch_stub = types.ModuleType("torch")
    torch_stub.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda *_args, **_kwargs: None

    whisper_stub = types.ModuleType("whisper")
    whisper_stub.load_model = lambda *args, **kwargs: None

    ffmpeg_stub = types.ModuleType("ffmpeg")
    ffmpeg_stub.input = lambda *args, **kwargs: None

    librosa_stub = types.ModuleType("librosa")
    librosa_stub.feature = types.SimpleNamespace(rms=lambda y: np.array([[0.0]]))

    pyannote_stub = types.ModuleType("pyannote")
    pyannote_audio_stub = types.ModuleType("pyannote.audio")

    class DummyPipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyPipeline()

        def to(self, device):
            return self

    pyannote_audio_stub.Pipeline = DummyPipeline
    pyannote_stub.audio = pyannote_audio_stub

    return {
        "torch": torch_stub,
        "whisper": whisper_stub,
        "ffmpeg": ffmpeg_stub,
        "librosa": librosa_stub,
        "pyannote": pyannote_stub,
        "pyannote.audio": pyannote_audio_stub,
    }


class TestEmptyAudio(unittest.TestCase):
    def test_empty_audio_raises(self):
        modules = _stub_modules()
        with patch.dict(sys.modules, modules):
            sys.modules.pop("extract", None)
            extract = importlib.import_module("extract")
            pcm = np.array([], dtype=np.int16)
            with self.assertRaisesRegex(ValueError, "total_duration must be positive"):
                extract.transcribe_with_whisper(pcm, "tiny", 0.5, 0)
        sys.modules.pop("extract", None)


if __name__ == "__main__":
    unittest.main()

