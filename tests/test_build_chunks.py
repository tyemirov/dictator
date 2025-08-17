import sys
import types

# Provide dummy modules so importing main doesn't require heavy deps
sys.modules['ffmpeg'] = types.ModuleType('ffmpeg')
sys.modules['soundfile'] = types.ModuleType('soundfile')
sys.modules['torch'] = types.ModuleType('torch')

tts_mod = types.ModuleType('TTS')
api_mod = types.ModuleType('TTS.api')
class DummyTTS: ...
api_mod.TTS = DummyTTS
sys.modules['TTS'] = tts_mod
sys.modules['TTS.api'] = api_mod

from main import build_chunks


def test_chinese_text_respects_byte_budget():
    text = "你好世界" * 10
    budget = 15
    chunks = build_chunks(text, budget)
    assert len(chunks) == 1
    assert len(chunks[0].encode('utf-8')) <= budget


def test_emoji_text_respects_byte_budget():
    text = "😀" * 10
    budget = 21
    chunks = build_chunks(text, budget)
    assert len(chunks) == 1
    assert len(chunks[0].encode('utf-8')) <= budget
