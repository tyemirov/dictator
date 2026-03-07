#!/usr/bin/env python3
"""
Whisper speech-clip extractor limited to the dominant speaker.

- Outputs 24 kHz mono WAV, peak-normalised to -1 dBFS
- Picks the window where the dominant speaker speaks most cleanly
- Uses pyannote speaker diarization when available
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import logging
import sys
from pathlib import Path

from dictator.audio.ffmpeg_ops import decode_pcm, trim_and_normalise
from dictator.extraction import (
    DIARIZATION_MODEL,
    MAX_CENTROID_HZ,
    MAX_SPEECH_RATE,
    MIN_CENTROID_HZ,
    POST_ROLL_SEC,
    PRE_ROLL_SEC,
    PUBLIC_SIZES,
    SAMPLE_RATE,
    STRIDE_SEC,
    TARGET_SR,
    WIN_SEC,
    apply_diarization_filter,
    choose_window,
    compute_trim_bounds,
    load_diarization_pipeline,
    pitch_variation,
    snr,
    spectral_centroid,
)
from dictator.extraction.service import timed
from dictator.runtime import run_with_timeout
from dictator.transcription.service import load_whisper_model, transcribe_words
from duration import parse_duration


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
    parser.add_argument(
        "--min-centroid",
        type=float,
        default=MIN_CENTROID_HZ,
        help="discard windows below this spectral centroid (Hz)",
    )
    parser.add_argument(
        "--max-centroid",
        type=float,
        default=MAX_CENTROID_HZ,
        help="discard windows above this spectral centroid (Hz)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.output.exists() and not args.force:
        if input(f"{args.output} exists. Overwrite? [y/N]: ").lower() != "y":
            sys.exit(0)

    pcm = run_with_timeout(args.timeouts[0], "decode", decode_pcm, args.input)
    total_track_seconds = len(pcm) / SAMPLE_RATE
    logging.info("Track length %s", timedelta(seconds=int(total_track_seconds)))

    diarization_pipeline = load_diarization_pipeline()
    model = load_whisper_model(args.model)

    def log_progress(segment_end: float) -> None:
        percent = int(100 * segment_end / total_track_seconds)
        if percent - log_progress.last_logged >= 5:
            logging.info("  progress %d %%", percent)
            log_progress.last_logged = percent

    log_progress.last_logged = 0
    with timed("transcribe"):
        raw_words = run_with_timeout(
            args.timeouts[1],
            "transcription",
            transcribe_words,
            pcm,
            language=args.language,
            model=model,
            progress_cb=log_progress,
        )
    if not raw_words:
        raise RuntimeError("no words transcribed")

    dominant_speaker_words = apply_diarization_filter(raw_words, diarization_pipeline, args.input)
    window_start = choose_window(
        pcm,
        dominant_speaker_words,
        args.duration,
        max_speech_rate=args.max_speech_rate,
        max_centroid=args.max_centroid,
        min_centroid=args.min_centroid,
    )
    window_end = window_start + args.duration
    logging.info(
        "window %s -> %s",
        timedelta(seconds=round(window_start)),
        timedelta(seconds=round(window_end)),
    )

    window_words = [
        word for word in dominant_speaker_words if window_start <= float(word["start"]) < window_end
    ]
    trim_start, trim_end = compute_trim_bounds(total_track_seconds, window_words)
    logging.info(
        "trim %s -> %s",
        timedelta(seconds=round(trim_start)),
        timedelta(seconds=round(trim_end)),
    )

    with timed("trim"):
        max_volume_str, gain_db = run_with_timeout(
            args.timeouts[2],
            "trim",
            trim_and_normalise,
            args.input,
            args.output,
            trim_start,
            trim_end - trim_start,
        )
    logging.info("peak %s dBFS, applying %+0.1f dB gain", max_volume_str, gain_db)
    logging.info("saved -> %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
