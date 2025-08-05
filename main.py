#!/usr/bin/env python3
"""
XTTS-v2 long-form voice cloning (GPU preferred)

• 24 kHz mono output, peak-normalised to −1 dBFS
• ffmpeg-python only – never spawns subprocesses
• Smart byte-budget splitter: ≤ 800 UTF-8 bytes.
  ⇒ typically 6-10 sentences per chunk, no “light.” orphans, and never hits
  XTTS’ ≈ 250-code-point truncation limit
• --length NN[s|m|h] trims on the last *complete* sentence that fits
• --language CODE selects the TTS language (e.g. en, ru)
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
BYTE_BUDGET = 800  # safe for XTTS-v2  (≈ 25-30 s of English speech)

CTRL_REMOVE = {c: None for c in range(32)} | {127: None}  # strip ASCII control chars


def clean(text: str) -> str:
    """Unicode-normalise, strip controls, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text.translate(CTRL_REMOVE))
    return " ".join(text.split())


def split_into_sentences(text: str) -> List[str]:
    """Return sentences *with* their terminal punctuation."""
    return re.split(r"(?<=[.!?])\s+", text)


def fits_xtts(chunk: str, budget: int = BYTE_BUDGET) -> bool:
    """True if `chunk` is within the UTF-8 byte budget accepted by XTTS."""
    return len(chunk.encode("utf-8")) <= budget


def build_chunks(text: str, budget: int = BYTE_BUDGET) -> List[str]:
    """
    Greedy byte-budget splitter.

    • Add sentences until the next would overflow `budget`.
    • Never output empty chunks.
    • Final pass merges very short tails (< 80 bytes) into the previous chunk.
    """
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    buf = ""

    for sent in sentences:
        candidate = f"{buf} {sent}".strip() if buf else sent
        if fits_xtts(candidate, budget):
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = sent if fits_xtts(sent, budget) else sent[:budget]
    if buf:
        chunks.append(buf)

    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk.encode("utf-8")) < 80 and fits_xtts(
                f"{merged[-1]} {chunk}", budget
        ):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def parse_length(spec: Optional[str]) -> Optional[float]:
    """'90s' → 90.0, '2m' → 120.0, '1.5h' → 5400.0."""
    if not spec:
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*([smhSMH])\s*", spec)
    if not m:
        raise ValueError("--length must look like '90s', '2m' or '1.5h'")
    value, unit = m.groups()
    return float(value) * {"s": 1, "m": 60, "h": 3600}[unit.lower()]


def mp3_to_wav(src: Path, dst: Path) -> None:
    (
        ffmpeg.input(str(src))
        .output(str(dst), ar=TARGET_SR, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )


def concat_normalise(inputs: List[Path], dst: Path, cap: Optional[float]) -> None:
    """Concat → dynaudnorm → −1 dBFS → 24 kHz mono WAV."""
    streams = [ffmpeg.input(str(f)) for f in inputs]
    audio = ffmpeg.concat(*[s.audio for s in streams], v=0, a=1)
    audio = audio.filter("dynaudnorm").filter("volume", 0.891250938)
    out_kwargs = dict(ar=TARGET_SR, ac=1, acodec="pcm_s16le")
    if cap is not None:
        out_kwargs["t"] = str(cap)
    (
        audio.output(str(dst), **out_kwargs)
        .overwrite_output()
        .run(quiet=True)
    )


def synthesise(
        speaker_wav: Path, chunks: List[str], cap: Optional[float], language_code: str
) -> List[Path]:
    """Generate WAV chunks until `cap` seconds have been produced."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(MODEL_ID).to(device)

    tmp_dir = Path("_tts_chunks")
    tmp_dir.mkdir(exist_ok=True)
    wav_paths: List[Path] = []

    elapsed = 0.0
    for idx, chunk in enumerate(chunks):
        if cap and elapsed >= cap:
            break

        path = tmp_dir / f"{idx:04d}.wav"
        tts.tts_to_file(
            text=chunk,
            speaker_wav=str(speaker_wav),
            language=language_code,
            file_path=str(path),
        )

        info = sf.info(path)
        dur = info.frames / info.samplerate

        if cap and elapsed + dur > cap:
            path.unlink()
            logging.warning(
                "Sentence %.1fs longer than remaining cap (%.1fs) – skipped",
                dur,
                cap - elapsed,
                )
            break

        wav_paths.append(path)
        elapsed += dur
        logging.info("chunk %03d  %.1f s  (cumulative %.1f s)", idx, dur, elapsed)

    if cap and not wav_paths:
        logging.error("First sentence (%.1fs) exceeds --length – nothing generated", dur)
    return wav_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="reference WAV/MP3")
    parser.add_argument("--text", required=True, help="UTF-8 plain-text file")
    parser.add_argument("--output", required=True, help="destination WAV")
    parser.add_argument("--length", help="max duration (e.g. 3m or 180s)")
    parser.add_argument(
        "--language", default="en", help="TTS language code (e.g. 'en', 'ru')"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    language_code = args.language

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        if input(f"{out_path} exists – overwrite? [y/N] ").lower() != "y":
            return

    sample_path = Path(args.sample)
    ref_wav = sample_path
    if sample_path.suffix.lower() == ".mp3":
        ref_wav = sample_path.with_suffix(".wav")
        mp3_to_wav(sample_path, ref_wav)
        logging.info("converted MP3 → WAV: %s", ref_wav)

    raw_text = Path(args.text).read_text(encoding="utf-8")
    clean_text = clean(raw_text)
    text_chunks = build_chunks(clean_text)
    logging.info("XTTS calls: %d  (≤%d UTF-8 bytes each)", len(text_chunks), BYTE_BUDGET)

    cap_seconds = parse_length(args.length)
    wav_chunks = synthesise(ref_wav, text_chunks, cap_seconds, language_code)
    if not wav_chunks:
        return

    concat_normalise(wav_chunks, out_path, cap_seconds)

    # tidy up
    for f in wav_chunks:
        f.unlink()
    Path("_tts_chunks").rmdir()

    logging.info("✓ saved → %s", out_path)


if __name__ == "__main__":
    main()
