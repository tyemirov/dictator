#!/usr/bin/env python3
"""
XTTS-v2 long-form voice cloning

• 24 kHz mono output, peak-normalised (dynaudnorm → –1 dBFS)
• ffmpeg-python only – no subprocess()
• smart text splitting (≤ 240 chars so XTTS never truncates)
• optional --length NN[s|m|h] => stop at the last full sentence that fits
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import ffmpeg
import soundfile as sf
import torch
from TTS.api import TTS

MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
TARGET_SR = 24_000
MAX_CHARS = 240  # safe per-chunk limit for XTTS

CTRL_REMOVE = {c: None for c in range(32)} | {127: None}  # strip ASCII control


# ───────────── text helpers ──────────────────────────────────────────────
def clean(text: str) -> str:
    """Unicode-normalise + collapse whitespace + strip controls."""
    cleaned = unicodedata.normalize("NFKC", text.translate(CTRL_REMOVE))
    return " ".join(cleaned.split())


def smart_split(text: str, limit: int = MAX_CHARS) -> List[str]:
    """
    Split text ≤ `limit` characters but only on:
      1. sentence breaks [.?!]⎵
      2. phrase breaks [,;:–]⎵
      3. word breaks (fallback)

    Returned list has no empty strings and no part exceeds `limit`.
    """
    sentence_re = re.compile(r"([.!?])\s+")
    phrase_re = re.compile(r"([,;:–])\s+")

    chunks: List[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer:
            chunks.append(buffer.strip())
            buffer = ""

    for segment in sentence_re.split(text):
        if not segment:
            continue
        if len(buffer) + len(segment) <= limit:
            buffer += segment
        else:
            for phrase in phrase_re.split(segment):
                if not phrase:
                    continue
                if len(buffer) + len(phrase) <= limit:
                    buffer += phrase
                else:
                    for word in phrase.split():
                        if len(buffer) + len(word) + 1 <= limit:
                            buffer = f"{buffer} {word}".strip()
                        else:
                            flush()
                            buffer = word
    flush()
    return chunks


def parse_length(spec: Optional[str]) -> Optional[float]:
    """'90s' → 90.0, '2m' → 120.0, '1.5h' → 5400.0."""
    if not spec:
        return None
    match = re.fullmatch(r"\s*([\d.]+)\s*([smhSMH])\s*", spec)
    if not match:
        raise ValueError("--length must look like '90s', '2m' or '1.5h'")
    value, unit = match.groups()
    return float(value) * {"s": 1, "m": 60, "h": 3600}[unit.lower()]


# ───────────── ffmpeg helpers ────────────────────────────────────────────
def mp3_to_wav(src_mp3: Path, dst_wav: Path) -> None:
    (ffmpeg
     .input(str(src_mp3))
     .output(str(dst_wav), ar=TARGET_SR, ac=1, acodec="pcm_s16le")
     .overwrite_output()
     .run(quiet=True))


def concat_normalise(src_wavs: List[Path], dst: Path,
                     cap: Optional[float]) -> None:
    """Concat → dynaudnorm → –1 dBFS → 24 kHz mono."""
    streams = [ffmpeg.input(str(w)) for w in src_wavs]
    audio = ffmpeg.concat(*[s.audio for s in streams], v=0, a=1)
    audio = audio.filter("dynaudnorm").filter("volume", 0.891250938)
    out = audio.output(str(dst), ar=TARGET_SR, ac=1, acodec="pcm_s16le",
                       t=str(cap) if cap else None
                       ).overwrite_output()
    out.run(quiet=True)


# ───────────── synthesis ────────────────────────────────────────────────
def synthesise(speaker: Path, pieces: List[str],
               cap: Optional[float]) -> List[Path]:
    """Generate WAV chunks until `cap` seconds is reached (or all text)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(MODEL_ID).to(device)

    tmp_dir = Path("_tts_chunks")
    tmp_dir.mkdir(exist_ok=True)
    wavs: List[Path] = []

    elapsed = 0.0
    for idx, text in enumerate(pieces):
        if cap and elapsed >= cap:
            break
        wav_path = tmp_dir / f"{idx:04d}.wav"
        tts.tts_to_file(text=text, speaker_wav=str(speaker),
                        language="en", file_path=str(wav_path))

        info = sf.info(wav_path)
        duration = info.frames / info.samplerate

        # if adding this chunk would exceed --length, discard it and stop
        if cap and elapsed + duration > cap:
            wav_path.unlink()
            logging.warning("sentence %.1fs is longer than remaining cap "
                            "(%.1fs) – skipped", duration, cap - elapsed)
            break

        wavs.append(wav_path)
        elapsed += duration
        logging.info("chunk %03d   %.1fs   cumulative %.1fs",
                     idx, duration, elapsed)

    if cap and not wavs:
        logging.error("first sentence (%.1fs) exceeds --length – nothing done",
                      duration)
        return []

    return wavs


# ───────────── CLI ───────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True, help="reference WAV / MP3")
    p.add_argument("--text", required=True, help="UTF-8 plain-text file")
    p.add_argument("--output", required=True, help="destination WAV")
    p.add_argument("--length", help="limit final duration, e.g. 3m or 180s")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
                        datefmt="%H:%M:%S")

    out = Path(args.output)
    if out.exists() and not args.force:
        if input(f"{out} exists – overwrite? [y/N] ").lower() != "y":
            return

    sample = Path(args.sample)
    ref_wav = sample
    if sample.suffix.lower() == ".mp3":
        ref_wav = sample.with_suffix(".wav")
        mp3_to_wav(sample, ref_wav)
        logging.info("converted MP3 → WAV: %s", ref_wav)

    raw_text = Path(args.text).read_text(encoding="utf-8")
    pieces = smart_split(clean(raw_text))
    logging.info("XTTS calls: %d (≤%d chars each)", len(pieces), MAX_CHARS)

    cap = parse_length(args.length)
    wav_chunks = synthesise(ref_wav, pieces, cap)
    if not wav_chunks:
        return

    concat_normalise(wav_chunks, out, cap)

    # tidy up
    for w in wav_chunks:
        w.unlink()
    Path("_tts_chunks").rmdir()

    logging.info("✓ saved → %s", out)


if __name__ == "__main__":
    main()
