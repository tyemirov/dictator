from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())

from dictator.audio import ffmpeg_ops


class _FakeAudioStream:
    def __init__(self, name="stream", run_result=(b"", b"")):
        self.name = name
        self.filters = []
        self.output_args = None
        self.overwrite_called = False
        self.run_result = run_result

    @property
    def audio(self):
        return self

    def filter(self, name, *args, **kwargs):
        self.filters.append((name, args, kwargs))
        return self

    def output(self, *args, **kwargs):
        self.output_args = (args, kwargs)
        return self

    def overwrite_output(self):
        self.overwrite_called = True
        return self

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self.run_result


class FfmpegOpsTests(unittest.TestCase):
    def test_decode_pcm_decodes_mono_pcm(self):
        pcm = np.array([1, -2, 3], dtype=np.int16).tobytes()
        stream = MagicMock()
        stream.output.return_value.run.return_value = (pcm, b"")
        with patch.object(ffmpeg_ops.ffmpeg, "input", return_value=stream, create=True):
            decoded = ffmpeg_ops.decode_pcm(Path("sample.wav"), sample_rate=22050)

        self.assertEqual(decoded.tolist(), [1, -2, 3])
        stream.output.assert_called_once_with("pipe:", format="s16le", ac=1, ar=22050)

    def test_audio_to_wav_builds_expected_ffmpeg_pipeline(self):
        stream = MagicMock()
        stream.output.return_value.overwrite_output.return_value.run.return_value = None
        with patch.object(ffmpeg_ops.ffmpeg, "input", return_value=stream, create=True):
            ffmpeg_ops.audio_to_wav(Path("sample.webm"), Path("sample.wav"), target_sample_rate=24000)

        stream.output.assert_called_once_with("sample.wav", ar=24000, ac=1, acodec="pcm_s16le")

    def test_mp3_to_wav_delegates_to_audio_to_wav(self):
        with patch.object(ffmpeg_ops, "audio_to_wav") as audio_to_wav:
            ffmpeg_ops.mp3_to_wav(Path("sample.mp3"), Path("sample.wav"), target_sample_rate=24000)

        audio_to_wav.assert_called_once_with(Path("sample.mp3"), Path("sample.wav"), target_sample_rate=24000)

    def test_normalised_concat_stream_applies_concat_norm_and_cap(self):
        inputs = [_FakeAudioStream("a"), _FakeAudioStream("b")]
        concat_stream = _FakeAudioStream("concat")
        with (
            patch.object(ffmpeg_ops.ffmpeg, "input", side_effect=inputs, create=True),
            patch.object(ffmpeg_ops.ffmpeg, "concat", return_value=concat_stream, create=True) as concat_mock,
        ):
            result = ffmpeg_ops._normalised_concat_stream([Path("a.wav"), Path("b.wav")], cap=3.0)

        self.assertIs(result, concat_stream)
        concat_mock.assert_called_once()
        self.assertEqual(concat_stream.filters[0][0], "dynaudnorm")
        self.assertEqual(concat_stream.filters[1], ("atrim", (), {"duration": 3.0}))

    def test_concat_normalise_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ffmpeg_ops.concat_normalise([], Path("out.wav"), None)

    def test_concat_normalise_applies_volume_when_peak_detected(self):
        detect_stream = _FakeAudioStream(run_result=(b"", b"max_volume: -3.0 dB"))
        audio_stream = _FakeAudioStream()
        with patch.object(ffmpeg_ops, "_normalised_concat_stream", side_effect=[detect_stream, audio_stream]):
            ffmpeg_ops.concat_normalise([Path("a.wav")], Path("out.wav"), cap=2.0, target_sample_rate=24000)

        self.assertIn(("volume", ("2.0dB",), {}), audio_stream.filters)
        self.assertEqual(audio_stream.output_args[1]["ar"], 24000)
        self.assertTrue(audio_stream.overwrite_called)

    def test_concat_normalise_skips_volume_when_peak_missing_or_negative_infinity(self):
        for stderr in (b"", b"max_volume: -inf dB"):
            detect_stream = _FakeAudioStream(run_result=(b"", stderr))
            audio_stream = _FakeAudioStream()
            with patch.object(ffmpeg_ops, "_normalised_concat_stream", side_effect=[detect_stream, audio_stream]):
                ffmpeg_ops.concat_normalise([Path("a.wav")], Path("out.wav"), cap=None)
            self.assertNotIn("volume", [name for name, _, _ in audio_stream.filters])

    def test_trim_and_normalise_raises_when_volumedetect_has_no_match(self):
        input_stream = _FakeAudioStream(run_result=(b"", b"no peak info"))
        with patch.object(ffmpeg_ops.ffmpeg, "input", return_value=input_stream, create=True):
            with self.assertRaisesRegex(RuntimeError, "failed to find max_volume"):
                ffmpeg_ops.trim_and_normalise(Path("in.wav"), Path("out.wav"), 0.0, 1.0)

    def test_trim_and_normalise_handles_negative_infinity_and_numeric_peaks(self):
        input_stream = _FakeAudioStream(run_result=(b"", b"max_volume: -inf dB"))
        second_stream = _FakeAudioStream()
        with patch.object(ffmpeg_ops.ffmpeg, "input", side_effect=[input_stream, second_stream], create=True):
            max_volume, gain_db = ffmpeg_ops.trim_and_normalise(Path("in.wav"), Path("out.wav"), 1.0, 2.0)
        self.assertEqual((max_volume, gain_db), ("-inf", 0.0))
        self.assertIn(("volume", (1.0,), {}), second_stream.filters)

        input_stream = _FakeAudioStream(run_result=(b"", b"max_volume: -6.0 dB"))
        second_stream = _FakeAudioStream()
        with patch.object(ffmpeg_ops.ffmpeg, "input", side_effect=[input_stream, second_stream], create=True):
            max_volume, gain_db = ffmpeg_ops.trim_and_normalise(Path("in.wav"), Path("out.wav"), 1.0, 2.0)
        self.assertEqual(max_volume, "-6.0")
        self.assertAlmostEqual(gain_db, 5.0)
        self.assertAlmostEqual(second_stream.filters[0][1][0], 10 ** (5.0 / 20))


if __name__ == "__main__":
    unittest.main()
