"""SRT rendering helpers."""

from __future__ import annotations

import math
from typing import Sequence

from .models import AlignedWord


def srt_timestamp_from_seconds(seconds: float, rounding: str) -> int:
    if rounding == "ceil":
        return int(math.ceil(seconds * 1000))
    return int(math.floor(seconds * 1000))


def format_srt_timestamp(milliseconds: int) -> str:
    total_seconds, millis = divmod(milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(words: Sequence[AlignedWord]) -> str:
    lines: list[str] = []
    for index, word in enumerate(words, start=1):
        start_ms = srt_timestamp_from_seconds(word.start_seconds, "floor")
        end_ms = srt_timestamp_from_seconds(word.end_seconds, "ceil")
        lines.append(str(index))
        lines.append(
            f"{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}"
        )
        lines.append(word.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
