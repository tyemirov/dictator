#!/usr/bin/env python3
"""
Whisper GPU speech-clip extractor
• Outputs 24 kHz mono WAV, peak-normalised to –1 dBFS.
• Picks the window (default 10 s) with the largest number of confidently
  recognised words; ties resolved by avg_conf × SNR.
• Works on GPU even if cuDNN is absent (PyTorch fallback kernels).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Iterator, List, Dict

import ffmpeg
import numpy as np
import torch
import whisper

# ───────── configuration ────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
TARGET_SR = 24_000
WIN_SEC = 20.0          # user-requested window length
STRIDE_SEC = 1.0
CENTROID_HZ = 4_000
PUBLIC_SIZES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}

# ───────── utility wrappers ─────────────────────────────────────────────────
@contextmanager
def timed(tag: str) -> Iterator[None]:
    t0 = time.perf_counter()
    logging.info("START %s", tag)
    yield
    logging.info("DONE  %s  (Δ = %.1fs)", tag, time.perf_counter() - t0)


@contextmanager
def timeout(sec: int, task: str) -> Iterator[None]:
    if sec <= 0:
        yield
        return

    def _handler(_s, _f):
        raise TimeoutError(f"{task} exceeded {sec}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(sec)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)

# ───────── low-level audio helpers ─────────────────────────────────────────-
def decode_pcm(src: Path) -> np.ndarray:
    buf, _ = (
        ffmpeg.input(str(src))
        .output("pipe:", format="s16le", ac=1, ar=SAMPLE_RATE)
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(buf, dtype=np.int16)


def spectral_centroid(samples: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(samples.astype(float)))
    if spec.sum() == 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    return float((spec * freqs).sum() / spec.sum())


def snr(samples: np.ndarray) -> float:
    samples_f = samples.astype(float)
    noise = np.percentile(np.abs(samples_f), 20)
    return samples_f.std() / (noise + 1e-6)

# ───────── Whisper helpers ──────────────────────────────────────────────────
def load_whisper(size: str) -> whisper.Whisper:
    if size not in PUBLIC_SIZES:
        raise ValueError(f"--model must be one of {sorted(PUBLIC_SIZES)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with timed(f"load_model {size} ({device})"):
        return whisper.load_model(
            size,
            device=device,
            download_root=str(Path.home() / ".cache" / "whisper"),
        )


def transcribe(
        pcm: np.ndarray,
        size: str,
        min_conf: float,
        total_sec: float,
) -> List[Dict]:
    model = load_whisper(size)
    words: List[Dict] = []

    def heartbeat(seg_end: float) -> None:
        pct = int(100 * seg_end / total_sec)
        if pct - heartbeat.last >= 5:
            logging.info("  progress %d %%", pct)
            heartbeat.last = pct
    heartbeat.last = 0    # type: ignore[attr-defined]

    with timed("transcribe"):
        result = model.transcribe(
            pcm.astype(np.float32) / 32768,
            word_timestamps=True,
            verbose=False,
            )
        for seg in result["segments"]:
            words.extend(seg["words"])
            heartbeat(seg["end"])

    confident = [w for w in words if w["probability"] >= min_conf]
    if not confident:
        raise RuntimeError("no words above confidence threshold")
    return confident

# ───────── window search ────────────────────────────────────────────────────
def choose_window(
        pcm: np.ndarray,
        words: List[Dict],
        duration: float,
        min_centroid: float = CENTROID_HZ,
) -> float:
    best_count, best_quality, best_start = -1, -1.0, 0.0
    track_len = len(pcm) / SAMPLE_RATE

    with timed("window_search"):
        pos = 0.0
        while pos + duration <= track_len:
            chunk = pcm[int(pos * SAMPLE_RATE) : int((pos + duration) * SAMPLE_RATE)]
            if spectral_centroid(chunk) > min_centroid:
                pos += STRIDE_SEC
                continue

            cnt = sum(pos <= w["start"] < pos + duration for w in words)
            if cnt == 0:
                pos += STRIDE_SEC
                continue

            avg_conf = (
                    sum(w["probability"] for w in words if pos <= w["start"] < pos + duration)
                    / cnt
            )
            quality = avg_conf * snr(chunk)

            if (cnt > best_count) or (cnt == best_count and quality > best_quality):
                best_count, best_quality, best_start = cnt, quality, pos
            pos += STRIDE_SEC

    if best_count < 0:
        raise RuntimeError("no suitable window found")

    logging.info("chosen window: %d words, quality %.2f", best_count, best_quality)
    return best_start

# ───────── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="medium", choices=sorted(PUBLIC_SIZES))
    parser.add_argument("--duration", type=float, default=WIN_SEC)
    parser.add_argument("--min-confidence", type=float, default=0.80)
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

    # ---------- decode ------------------------------------------------------
    with timeout(args.timeouts[0], "decode"):
        pcm = decode_pcm(args.input)
    total_sec = len(pcm) / SAMPLE_RATE
    logging.info("Track length %s", timedelta(seconds=int(total_sec)))

    # ---------- transcribe --------------------------------------------------
    with timeout(args.timeouts[1], "transcription"):
        words = transcribe(pcm, args.model, args.min_confidence, total_sec)

    # ---------- pick window -------------------------------------------------
    start = choose_window(pcm, words, args.duration)
    logging.info(
        "window %s → %s",
        timedelta(seconds=round(start)),
        timedelta(seconds=round(start + args.duration)),
    )

    # ---------- trim & normalise -------------------------------------------
    with timeout(args.timeouts[2], "trim"), timed("trim"):
        (
            ffmpeg.input(str(args.input), ss=start, t=args.duration)
            .filter("volume", 0.891250938)         # static –1 dBFS peak
            .output(
                str(args.output),
                acodec="pcm_s16le",
                ac=1,
                ar=str(TARGET_SR),
            )
            .overwrite_output()
            .run(quiet=True)
        )
    logging.info("✓ Saved → %s", args.output)


if __name__ == "__main__":
    main()
