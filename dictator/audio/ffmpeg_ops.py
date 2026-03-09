"""ffmpeg-backed audio transforms used by the service core."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import ffmpeg

from .constants import PCM_SAMPLE_RATE, TARGET_SAMPLE_RATE


def decode_pcm(source_path: Path, sample_rate: int = PCM_SAMPLE_RATE):
    """Decode an audio file to mono int16 PCM at the requested sample rate."""
    import numpy as np

    buffer, _ = (
        ffmpeg.input(str(source_path))
        .output("pipe:", format="s16le", ac=1, ar=sample_rate)
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(buffer, dtype=np.int16)


def audio_to_wav(
    src: Path,
    dst: Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """Convert an audio input into mono PCM WAV."""
    (
        ffmpeg.input(str(src))
        .output(str(dst), ar=target_sample_rate, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )


def mp3_to_wav(
    src: Path,
    dst: Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """Backward-compatible wrapper for converting audio into mono PCM WAV."""
    audio_to_wav(src, dst, target_sample_rate=target_sample_rate)


def _normalised_concat_stream(
    inputs: Sequence[Path],
    cap: float | None,
):
    streams = [ffmpeg.input(str(file_path)) for file_path in inputs]
    audio = ffmpeg.concat(*[stream.audio for stream in streams], v=0, a=1)
    audio = audio.filter("dynaudnorm")
    if cap is not None:
        audio = audio.filter("atrim", duration=cap)
    return audio


def concat_normalise(
    inputs: Sequence[Path],
    dst: Path,
    cap: float | None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """Concat WAV inputs and peak-normalise the result to -1 dBFS."""
    if not inputs:
        raise ValueError("inputs must not be empty")

    detect = (
        _normalised_concat_stream(inputs, cap)
        .filter("volumedetect")
        .output("null", f="null")
        .overwrite_output()
    )
    _, stderr = detect.run(capture_stdout=True, capture_stderr=True)
    match = re.search(r"max_volume: (-?inf|[-\d.]+) dB", stderr.decode())
    if not match or match.group(1) == "-inf":
        gain_db = 0.0
    else:
        gain_db = -1.0 - float(match.group(1))

    audio = _normalised_concat_stream(inputs, cap)
    if abs(gain_db) > 1e-3:
        audio = audio.filter("volume", f"{gain_db}dB")
    (
        audio.output(str(dst), ar=target_sample_rate, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )


def trim_and_normalise(
    src: Path,
    dst: Path,
    start_seconds: float,
    duration_seconds: float,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[str, float]:
    """Trim a region and peak-normalise it to -1 dBFS."""
    _, stderr = (
        ffmpeg.input(str(src), ss=start_seconds, t=duration_seconds)
        .filter("aresample", str(target_sample_rate))
        .filter("aformat", channel_layouts="mono")
        .filter("volumedetect")
        .output("-", f="null")
        .run(capture_stdout=True, capture_stderr=True)
    )
    match = re.search(r"max_volume:\s*(-?inf|-?\d+(?:\.\d+)?)\s*dB", stderr.decode())
    if not match:
        raise RuntimeError("volumedetect failed to find max_volume")
    max_volume_str = match.group(1)
    if max_volume_str == "-inf":
        gain_db = 0.0
    else:
        gain_db = -1.0 - float(max_volume_str)
    volume_factor = 10 ** (gain_db / 20)
    (
        ffmpeg.input(str(src), ss=start_seconds, t=duration_seconds)
        .filter("volume", volume_factor)
        .output(str(dst), acodec="pcm_s16le", ac=1, ar=str(target_sample_rate))
        .overwrite_output()
        .run(quiet=True)
    )
    return max_volume_str, gain_db
