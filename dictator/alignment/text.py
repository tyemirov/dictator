"""Transcript normalization for forced alignment."""

from __future__ import annotations

import re
import unicodedata

from dictator.runtime.errors import ValidationError

from .models import SUPPORTED_LANGUAGE_CODES

INVALID_CONFIG_CODE = "dictator.alignment.input.invalid_config"
INVALID_LANGUAGE_CODE = "dictator.alignment.input.invalid_language"
SRT_TIME_RANGE_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$"
)


def is_srt_text(source_name: str, text_value: str) -> bool:
    if source_name.lower().endswith(".srt"):
        return True
    return any(
        SRT_TIME_RANGE_PATTERN.fullmatch(line.strip())
        for line in text_value.splitlines()
        if line.strip()
    )


def sanitize_srt_text(text_value: str) -> str:
    cleaned_lines: list[str] = []
    for line in text_value.replace("\ufeff", "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if SRT_TIME_RANGE_PATTERN.fullmatch(stripped):
            continue
        cleaned_lines.append(stripped)
    return " ".join(cleaned_lines)


def normalize_transcript(text_value: str, source_name: str = "transcript.txt") -> str:
    sanitized = sanitize_srt_text(text_value) if is_srt_text(source_name, text_value) else text_value
    normalized = " ".join(sanitized.replace("\ufeff", "").split())
    if not normalized:
        raise ValidationError(INVALID_CONFIG_CODE, "input text contains no words")
    return normalized


def remove_punctuation_from_transcript(text_value: str) -> str:
    replaced: list[str] = []
    for character in text_value:
        if unicodedata.category(character).startswith("P"):
            replaced.append(" ")
        else:
            replaced.append(character)
    return "".join(replaced)


def normalize_transcript_for_alignment(
    text_value: str,
    source_name: str,
    remove_punctuation: bool,
) -> str:
    normalized = normalize_transcript(text_value, source_name)
    if not remove_punctuation:
        return normalized
    stripped = " ".join(remove_punctuation_from_transcript(normalized).split())
    if not stripped:
        raise ValidationError(
            INVALID_CONFIG_CODE,
            "input text contains no words after punctuation removal",
        )
    return stripped


def detect_default_language(transcript: str) -> str:
    if any("\u0400" <= character <= "\u04ff" for character in transcript):
        return "ru"
    return "en"


def normalize_language_value(raw_value: str, default_language: str) -> str:
    normalized = raw_value.strip().lower()
    if not normalized:
        return default_language
    if normalized not in SUPPORTED_LANGUAGE_CODES:
        raise ValidationError(
            INVALID_LANGUAGE_CODE,
            f"unsupported language: {normalized!r}",
        )
    return normalized


def is_punctuation_token(text_value: str) -> bool:
    return bool(text_value) and not any(character.isalnum() for character in text_value)


def strip_punctuation_from_token(text_value: str) -> str:
    kept: list[str] = []
    for character in text_value:
        if unicodedata.category(character).startswith("P"):
            continue
        kept.append(character)
    return "".join(kept).strip()
