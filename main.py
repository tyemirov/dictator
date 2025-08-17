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


def trim_utf8(text: str, budget: int) -> str:
    """Trim ``text`` to at most ``budget`` UTF-8 bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    return encoded[:budget].decode("utf-8", errors="ignore")


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
            buf = sent if fits_xtts(sent, budget) else trim_utf8(sent, budget)
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
    """Concat → dynaudnorm → peak normalise to −1 dBFS → 24 kHz mono WAV."""

    def normalised_streams() -> ffmpeg.nodes.FilterableStream:
        streams = [ffmpeg.input(str(f)) for f in inputs]
        audio = ffmpeg.concat(*[s.audio for s in streams], v=0, a=1)
        audio = audio.filter("dynaudnorm")
        if cap is not None:
            audio = audio.filter("atrim", duration=cap)
        return audio

    # First pass: detect max volume after dynamic normalisation
    detect = normalised_streams().filter("volumedetect").output("null", f="null").overwrite_output()
    _, stderr = detect.run(capture_stdout=True, capture_stderr=True)
    m = re.search(r"max_volume: (-?inf|[-\d.]+) dB", stderr.decode())
    if not m or m.group(1) == "-inf":
        gain_db = 0.0
    else:
        gain_db = -1.0 - float(m.group(1))

    # Second pass: apply gain and output
    audio = normalised_streams()
    if abs(gain_db) > 1e-3:
        audio = audio.filter("volume", f"{gain_db}dB")
    (
        audio.output(str(dst), ar=TARGET_SR, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )


def synthesise(
        speaker_wav: Path,
        chunks: List[str],
        cap: Optional[float],
        language_code: str,
) -> tuple[List[Path], List[dict]]:
    """Generate WAV chunks until `cap` seconds have been produced.

    Returns a tuple ``(paths, segments)`` where ``paths`` is the list of
    temporary WAV files and ``segments`` describes the timeline as
    ``{"content": chunk, "start": float, "end": float}``.
    """

    if not chunks:
        raise ValueError("No text chunks provided")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(MODEL_ID).to(device)

    tmp_dir = Path("_tts_chunks")
    tmp_dir.mkdir(exist_ok=True)
    wav_paths: List[Path] = []
    segments: List[dict] = []

    elapsed = 0.0
    dur = 0.0
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
        segments.append({"content": chunk, "start": elapsed, "end": elapsed + dur})
        elapsed += dur
        logging.info("chunk %03d  %.1f s  (cumulative %.1f s)", idx, dur, elapsed)

    if cap and not wav_paths:
        logging.error("First sentence (%.1fs) exceeds --length – nothing generated", dur)
        raise ValueError("No chunks fit within the length cap")
    return wav_paths, segments


def transcribe_words(wav_path: Path, language_code: str, model=None) -> List[dict]:
    """Use Whisper to extract word-level timestamps from ``wav_path``.

    Parameters
    ----------
    wav_path:
        Path to the audio file to transcribe.
    language_code:
        ISO language code hint for Whisper.
    model:
        Optional preloaded Whisper model. Supplying this allows tests to
        inject a stub implementation and avoids loading real weights.
    """

    if model is None:
        import whisper  # lazy import so tests can stub easily
        model = whisper.load_model("base")

    result = model.transcribe(
        str(wav_path), language=language_code, word_timestamps=True, verbose=False
    )

    words: List[dict] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            words.append(
                {
                    "content": word.get("word", "").strip(),
                    "start": word.get("start"),
                    "end": word.get("end"),
                }
            )
    return words

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="reference WAV/MP3")
    parser.add_argument("--text", required=True, help="UTF-8 plain-text file")
    parser.add_argument("--output", required=True, help="destination WAV")
    parser.add_argument("--length", help="max duration (e.g. 3m or 180s)")
    parser.add_argument(
        "--language", default="en", help="TTS language code (e.g. 'en', 'ru')"
    )
    parser.add_argument(
        "--speech",
        help="write JSON timeline alongside audio",
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
    pcm, sr = sf.read(ref_wav, dtype="int16")
    track_length = len(pcm) / sr if sr else 0
    if track_length <= 0:
        raise RuntimeError(f"reference audio {ref_wav} is empty")

    raw_text = Path(args.text).read_text(encoding="utf-8")
    clean_text = clean(raw_text)
    text_chunks = build_chunks(clean_text)
    logging.info("XTTS calls: %d  (≤%d UTF-8 bytes each)", len(text_chunks), BYTE_BUDGET)

    cap_seconds = parse_length(args.length)
    wav_chunks, _ = synthesise(ref_wav, text_chunks, cap_seconds, language_code)
    if not wav_chunks:
        return

    concat_normalise(wav_chunks, out_path, cap_seconds)

    if args.speech:
        import json

        word_segments = transcribe_words(out_path, language_code)
        timeline = {
            "textSegments": word_segments,
            "imageCues": [],
            "voices": [
                {
                    "id": sample_path.stem,
                    "label": sample_path.stem,
                    "file": str(sample_path),
                }
            ],
        }

        Path(args.speech).write_text(
            json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # tidy up
    for f in wav_chunks:
        f.unlink()
    Path("_tts_chunks").rmdir()

    logging.info("✓ saved → %s", out_path)


if __name__ == "__main__":
    main()
