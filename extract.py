#!/usr/bin/env python3
"""
Whisper GPU speech-clip extractor, limited to the single dominant speaker.
• Outputs 24 kHz mono WAV, peak-normalised to –1 dBFS.
• Picks the window (default 20 s) where the dominant speaker speaks most.
• Uses pyannote.audio’s pretrained speaker-diarization pipeline on GPU if available.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import tempfile
import time
import re
from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Iterator, List, Dict, Optional

import ffmpeg
import numpy as np
import torch
import whisper
import librosa
from pyannote.audio import Pipeline
from duration import parse_duration

SAMPLE_RATE = 16_000
TARGET_SR = 24_000
WIN_SEC = 20.0
STRIDE_SEC = 1.0
MAX_SPEECH_RATE = 4.0
MAX_CENTROID_HZ = 4_000  # skip overly bright segments
MIN_CENTROID_HZ = 500    # skip overly muffled segments
PUBLIC_SIZES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
DIARIZATION_MODEL = "pyannote/speaker-diarization@2.1"
PRE_ROLL_SEC = 0.2
POST_ROLL_SEC = 0.2


@contextmanager
def timed(tag: str) -> Iterator[None]:
    t0 = time.perf_counter()
    logging.info("START %s", tag)
    yield
    logging.info("DONE  %s  (Δ = %.1fs)", tag, time.perf_counter() - t0)


@contextmanager
def timeout(seconds: int, task_name: str) -> Iterator[None]:
    if seconds <= 0:
        yield
        return

    def _timeout_handler(_sig, _frame):
        raise TimeoutError(f"{task_name} exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def decode_pcm(source_path: Path) -> np.ndarray:
    buffer, _ = (
        ffmpeg.input(str(source_path))
        .output("pipe:", format="s16le", ac=1, ar=SAMPLE_RATE)
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(buffer, dtype=np.int16)


def spectral_centroid(samples: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(samples.astype(float)))
    if spec.sum() == 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    return float((spec * freqs).sum() / spec.sum())


def snr(samples: np.ndarray) -> float:
    samples_float = samples.astype(float)
    noise_floor = np.percentile(np.abs(samples_float), 20)
    return samples_float.std() / (noise_floor + 1e-6)


def pitch_variation(samples: np.ndarray) -> float:
    """Estimate RMS energy spread for a window.

    Uses librosa to compute frame-wise RMS and returns the standard
    deviation, which serves as a proxy for dynamic range.
    """
    y = samples.astype(float) / 32768.0
    rms = librosa.feature.rms(y=y)
    return float(rms.std())


def load_whisper_model(model_size: str) -> whisper.Whisper:
    if model_size not in PUBLIC_SIZES:
        raise ValueError(f"--model must be one of {sorted(PUBLIC_SIZES)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with timed(f"load_whisper_model {model_size} ({device})"):
        return whisper.load_model(
            model_size,
            device=device,
            download_root=str(Path.home() / ".cache" / "whisper"),
        )


def transcribe_with_whisper(
        pcm_array: np.ndarray,
        model_size: str,
        minimum_confidence: float,
        total_duration: float,
        language: Optional[str] = None,
) -> List[Dict]:
    model = load_whisper_model(model_size)
    word_list: List[Dict] = []

    def log_progress(segment_end: float) -> None:
        percent = int(100 * segment_end / total_duration)
        if percent - log_progress.last_logged >= 5:
            logging.info("  progress %d %%", percent)
            log_progress.last_logged = percent

    log_progress.last_logged = 0
    with timed("transcribe"):
        transcribe_kwargs = {"word_timestamps": True, "verbose": False}
        if language is not None:
            transcribe_kwargs["language"] = language
        result = model.transcribe(
            pcm_array.astype(np.float32) / 32768,
            **transcribe_kwargs,
        )
        for segment in result["segments"]:
            word_list.extend(segment["words"])
            log_progress(segment["end"])
    confident_words = [w for w in word_list if w["probability"] >= minimum_confidence]
    if not confident_words:
        raise RuntimeError("no words above confidence threshold")
    return confident_words


def load_diarization_pipeline() -> Pipeline:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with timed("load_diarization_pipeline"):
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
        pipeline.to(device)
        return pipeline


def apply_diarization_filter(
        words: List[Dict],
        diarization_pipeline: Pipeline,
        audio_file: Path,
) -> List[Dict]:
    with timed("prepare_diarization_audio"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_wav_path = Path(tmp_file.name)
        ffmpeg.input(str(audio_file)) \
            .output(str(temp_wav_path), ac=1, ar=SAMPLE_RATE, format="wav") \
            .overwrite_output() \
            .run(quiet=True)
    with timed("diarization"):
        diarization_result = diarization_pipeline({
            "uri": temp_wav_path.stem,
            "audio": str(temp_wav_path)
        })
    speaker_durations = Counter()
    for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
        speaker_durations[speaker_label] += (turn.end - turn.start)
    dominant_speaker = speaker_durations.most_common(1)[0][0]
    logging.info("dominant speaker: %s", dominant_speaker)
    filtered_words: List[Dict] = []
    for word in words:
        word_start = word["start"]
        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
            if turn.start <= word_start < turn.end:
                if speaker_label == dominant_speaker:
                    word["speaker"] = speaker_label
                    filtered_words.append(word)
                break
    try:
        temp_wav_path.unlink()
    except Exception:
        pass
    if not filtered_words:
        raise RuntimeError("no words from dominant speaker")
    return filtered_words


def choose_window(
        pcm_array: np.ndarray,
        speaker_words: List[Dict],
        duration: float,
        max_speech_rate: float = MAX_SPEECH_RATE,
        max_centroid: float = MAX_CENTROID_HZ,
        min_centroid: float = MIN_CENTROID_HZ,
) -> float:
    """Return the start time of the highest-quality window.

    Windows are skipped if their spectral centroid lies outside ``min_centroid``
    and ``max_centroid`` to avoid overly muffled or overly bright segments.
    """
    best_word_count, best_quality_score, best_window_start = -1, -1.0, 0.0
    track_length = len(pcm_array) / SAMPLE_RATE
    with timed("window_search"):
        position = 0.0
        while position + duration <= track_length:
            chunk = pcm_array[int(position * SAMPLE_RATE):int((position + duration) * SAMPLE_RATE)]
            centroid = spectral_centroid(chunk)
            if centroid > max_centroid or centroid < min_centroid:
                position += STRIDE_SEC
                continue
            words_in_window = [w for w in speaker_words if position <= w["start"] < position + duration]
            word_count = len(words_in_window)
            if word_count == 0:
                position += STRIDE_SEC
                continue
            if word_count / duration > max_speech_rate:
                position += STRIDE_SEC
                continue
            average_confidence = sum(
                w["probability"] for w in speaker_words if position <= w["start"] < position + duration
            ) / word_count
            variation = pitch_variation(chunk)
            quality_score = average_confidence * snr(chunk) * (1.0 + variation)
            if (word_count > best_word_count) or (
                    word_count == best_word_count and quality_score > best_quality_score
            ):
                best_word_count, best_quality_score, best_window_start = word_count, quality_score, position
            position += STRIDE_SEC
    if best_word_count < 0:
        raise RuntimeError("no suitable window found")
    logging.info("chosen window: %d words, quality %.2f", best_word_count, best_quality_score)
    return best_window_start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="medium", choices=sorted(PUBLIC_SIZES))
    parser.add_argument(
        "--duration",
        type=parse_duration,
        default=WIN_SEC,
        help="window length, e.g. '20', '60s', '1m'",
    )
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--language", help="ISO language code (e.g. 'en')")
    parser.add_argument(
        "--max-speech-rate",
        type=float,
        default=MAX_SPEECH_RATE,
        help="discard windows faster than this many words/s",
    )
    parser.add_argument(
        "--timeouts",
        nargs=3,
        metavar=("decode", "transcribe", "trim"),
        type=int,
        default=(60, 3600, 60),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.output.exists() and not args.force:
        if input(f"{args.output} exists. Overwrite? [y/N]: ").lower() != "y":
            sys.exit(0)
    with timeout(args.timeouts[0], "decode"):
        pcm = decode_pcm(args.input)
    total_track_seconds = len(pcm) / SAMPLE_RATE
    logging.info("Track length %s", timedelta(seconds=int(total_track_seconds)))
    diarization_pipeline = load_diarization_pipeline()
    with timeout(args.timeouts[1], "transcription"):
        raw_words = transcribe_with_whisper(
            pcm, args.model, args.min_confidence, total_track_seconds, args.language
        )
    dominant_speaker_words = apply_diarization_filter(raw_words, diarization_pipeline, args.input)
    window_start = choose_window(
        pcm,
        dominant_speaker_words,
        args.duration,
        max_speech_rate=args.max_speech_rate,
    )
    window_end = window_start + args.duration
    logging.info(
        "window %s → %s",
        timedelta(seconds=round(window_start)),
        timedelta(seconds=round(window_end)),
    )
    window_words = [
        w for w in dominant_speaker_words if window_start <= w["start"] < window_end
    ]
    window_words.sort(key=lambda w: w["start"])
    if not window_words:
        raise RuntimeError("no words found in chosen window")
    first_word_start = window_words[0]["start"]
    last_word_end = window_words[-1]["end"]
    trim_start = max(0.0, first_word_start - PRE_ROLL_SEC)
    trim_end = min(total_track_seconds, last_word_end + POST_ROLL_SEC)
    logging.info(
        "trim %s → %s",
        timedelta(seconds=round(trim_start)),
        timedelta(seconds=round(trim_end)),
    )
    with timeout(args.timeouts[2], "trim"), timed("trim"):
        trim_duration = trim_end - trim_start
        _, err = (
            ffmpeg.input(str(args.input), ss=trim_start, t=trim_duration)
            .filter("aresample", str(TARGET_SR))
            .filter("aformat", channel_layouts="mono")
            .filter("volumedetect")
            .output("-", f="null")
            .run(capture_stdout=True, capture_stderr=True)
        )
        match = re.search(r"max_volume:\s*(-?inf|-?\d+(?:\.\d+)?)\s*dB", err.decode())
        if not match:
            raise RuntimeError("volumedetect failed to find max_volume")
        max_volume_str = match.group(1)
        if max_volume_str == "-inf":
            gain_db = 0.0
        else:
            max_volume_db = float(max_volume_str)
            gain_db = -1.0 - max_volume_db
        volume_factor = 10 ** (gain_db / 20)
        logging.info(
            "peak %s dBFS, applying %+0.1f dB gain",
            max_volume_str,
            gain_db,
        )
        (
            ffmpeg.input(str(args.input), ss=trim_start, t=trim_duration)
            .filter("volume", volume_factor)
            .output(str(args.output), acodec="pcm_s16le", ac=1, ar=str(TARGET_SR))
            .overwrite_output()
            .run(quiet=True)
        )
    logging.info("✓ Saved → %s", args.output)


if __name__ == "__main__":
    main()
