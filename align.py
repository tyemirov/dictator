#!/usr/bin/env python3
"""Force-align audio to an existing transcript and emit SRT."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dictator.alignment import AlignTranscriptRequest, AlignmentService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="input audio or video")
    parser.add_argument("--text", required=True, type=Path, help="transcript text or SRT")
    parser.add_argument("--output", required=True, type=Path, help="destination SRT")
    parser.add_argument("--language", default="", help="language code; auto-detected when omitted")
    parser.add_argument("--device", default="auto", help="alignment device: auto, cpu, cuda")
    parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="strip punctuation before alignment",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output.exists() and not args.force:
        if input(f"{args.output} exists - overwrite? [y/N] ").lower() != "y":
            return

    transcript_text = args.text.read_text(encoding="utf-8")
    result = AlignmentService().align(
        AlignTranscriptRequest(
            audio_path=args.input,
            transcript_text=transcript_text,
            language=args.language,
            remove_punctuation=args.remove_punctuation,
            device=args.device,
            transcript_source_name=args.text.name,
            output_srt_path=args.output,
        )
    )
    logging.info("aligned %d word(s) in language %s", len(result.words), result.language)
    logging.info("saved -> %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
