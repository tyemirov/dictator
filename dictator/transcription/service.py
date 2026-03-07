"""Whisper-backed transcription helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from .models import WordSegment

ProgressCallback = Callable[[float], None]
AudioInput = Any


def load_whisper_model(model_size: str = "base", cache_dir: Path | None = None):
    """Load a Whisper model on CPU or GPU depending on availability."""
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    download_root = cache_dir or (Path.home() / ".cache" / "whisper")
    return whisper.load_model(
        model_size,
        device=device,
        download_root=str(download_root),
    )


def _coerce_audio_input(audio: AudioInput) -> str | object:
    if isinstance(audio, Path):
        return str(audio)
    import numpy as np

    if audio.size == 0:
        raise ValueError("audio array is empty")
    return audio.astype(np.float32) / 32768.0


def transcribe_word_segments(
    audio: AudioInput,
    language: Optional[str] = None,
    model: Optional[object] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[WordSegment]:
    """Transcribe audio and return typed word segments."""
    if model is None:
        model = load_whisper_model("base")

    kwargs = {"word_timestamps": True, "verbose": False}
    if language is not None:
        kwargs["language"] = language
    result = model.transcribe(_coerce_audio_input(audio), **kwargs)

    words: list[WordSegment] = []
    for segment in result.get("segments", []):
        if progress_cb and "end" in segment:
            progress_cb(segment["end"])
        for word in segment.get("words", []):
            words.append(
                WordSegment(
                    text=word.get("word", "").strip(),
                    start_seconds=word.get("start"),
                    end_seconds=word.get("end"),
                )
            )
    return words


def serialise_word_segments(words: Iterable[WordSegment]) -> list[dict[str, float | str | None]]:
    """Convert typed segments into the legacy dict structure."""
    return [word.to_legacy_dict() for word in words]


def transcribe_words(
    audio: AudioInput,
    language: Optional[str] = None,
    model: Optional[object] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[dict[str, float | str | None]]:
    """Compatibility wrapper returning the legacy word payload."""
    return serialise_word_segments(
        transcribe_word_segments(
            audio,
            language=language,
            model=model,
            progress_cb=progress_cb,
        )
    )


def transcribe_text(
    audio: AudioInput,
    language: Optional[str] = None,
    model: Optional[object] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """Transcribe audio and return plain text."""
    return " ".join(
        word.text
        for word in transcribe_word_segments(
            audio,
            language=language,
            model=model,
            progress_cb=progress_cb,
        )
        if word.text
    )
