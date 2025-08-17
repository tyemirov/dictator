import sys
import types
import importlib
import numpy as np
import pytest


def import_extract_with_stubs(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    whisper_stub = types.ModuleType("whisper")
    whisper_stub.load_model = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "whisper", whisper_stub)

    ffmpeg_stub = types.ModuleType("ffmpeg")
    ffmpeg_stub.input = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ffmpeg", ffmpeg_stub)

    librosa_stub = types.ModuleType("librosa")
    librosa_stub.feature = types.SimpleNamespace(rms=lambda y: np.array([[0.0]]))
    monkeypatch.setitem(sys.modules, "librosa", librosa_stub)

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
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_stub)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_stub)

    return importlib.import_module("extract")


def test_empty_audio_raises(monkeypatch):
    extract = import_extract_with_stubs(monkeypatch)
    pcm = np.array([], dtype=np.int16)
    with pytest.raises(ValueError, match="total_duration must be positive"):
        extract.transcribe_with_whisper(pcm, "tiny", 0.5, 0)
