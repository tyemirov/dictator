#!/usr/bin/env python3
"""Long-form voice cloning CLI."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile

from dictator.synthesis.models import SynthesisEngine, SynthesisRequest
from dictator.synthesis.text import clean, parse_length


def main() -> None:
    from dictator.audio.ffmpeg_ops import concat_normalise, mp3_to_wav
    from dictator.synthesis.service import SpeechSynthesisService, cleanup_synthesis_result
    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="reference WAV/MP3")
    parser.add_argument("--text", required=True, help="UTF-8 plain-text file")
    parser.add_argument("--output", required=True, help="destination WAV")
    parser.add_argument("--length", help="max duration (e.g. 3m or 180s)")
    parser.add_argument(
        "--language", default="en", help="TTS language code (e.g. 'en', 'ru')"
    )
    parser.add_argument(
        "--sample-text",
        required=True,
        help="reference transcript for the sample audio",
    )
    parser.add_argument(
        "--speech",
        help="write JSON timeline alongside audio",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        if input(f"{out_path} exists - overwrite? [y/N] ").lower() != "y":
            return

    sample_path = Path(args.sample)
    ref_wav = sample_path
    temp_ref_wav: Path | None = None
    result = None

    try:
        if sample_path.suffix.lower() == ".mp3":
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                prefix="dictator_ref_",
                delete=False,
            ) as handle:
                temp_ref_wav = Path(handle.name)
            mp3_to_wav(sample_path, temp_ref_wav)
            ref_wav = temp_ref_wav
            logging.info("converted MP3 -> WAV: %s", ref_wav)

        info = sf.info(ref_wav)
        track_length = info.frames / info.samplerate if info.samplerate else 0
        if track_length <= 0:
            raise RuntimeError(f"reference audio {ref_wav} is empty")

        raw_text = Path(args.text).read_text(encoding="utf-8")
        clean_text = clean(raw_text)
        logging.info("engine: %s", SynthesisEngine.QWEN3.value)

        cap_seconds = parse_length(args.length)
        result = SpeechSynthesisService().synthesise_text(
            SynthesisRequest(
                engine=SynthesisEngine.QWEN3,
                speaker_wav=ref_wav,
                text=clean_text,
                language_code=args.language,
                cap_seconds=cap_seconds,
                speaker_transcript_text=args.sample_text.strip(),
            )
        )
        if not result.wav_paths:
            return

        concat_normalise(result.wav_paths, out_path, cap_seconds)

        if args.speech:
            from dictator.transcription.service import transcribe_words

            word_segments = transcribe_words(out_path, args.language)
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
                json.dumps(timeline, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logging.info("saved -> %s", out_path)
    finally:
        if result is not None:
            cleanup_synthesis_result(result)
        if temp_ref_wav is not None:
            temp_ref_wav.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    main()
