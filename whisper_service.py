"""Compatibility wrapper for the packaged Whisper transcription service."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_whisper_model(model_size: str = "base"):
    from dictator.transcription.service import load_whisper_model as _load_whisper_model

    return _load_whisper_model(model_size)


def transcribe_word_segments(
    audio,
    language: Optional[str] = None,
    model: Optional[object] = None,
    progress_cb=None,
):
    from dictator.transcription.service import transcribe_word_segments as _transcribe_word_segments

    return _transcribe_word_segments(
        audio,
        language=language,
        model=model,
        progress_cb=progress_cb,
    )


def transcribe_words(
    audio,
    language: Optional[str] = None,
    model: Optional[object] = None,
    progress_cb=None,
):
    from dictator.transcription.service import transcribe_words as _transcribe_words

    return _transcribe_words(
        audio,
        language=language,
        model=model,
        progress_cb=progress_cb,
    )


def transcribe_text(
    audio,
    language: Optional[str] = None,
    model: Optional[object] = None,
    progress_cb=None,
) -> str:
    from dictator.transcription.service import transcribe_text as _transcribe_text

    return _transcribe_text(
        audio,
        language=language,
        model=model,
        progress_cb=progress_cb,
    )


__all__ = [
    "load_whisper_model",
    "transcribe_text",
    "transcribe_word_segments",
    "transcribe_words",
]
