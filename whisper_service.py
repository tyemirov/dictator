from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

import numpy as np
import torch


def load_whisper_model(model_size: str = "base"):
    """Load a Whisper model on CPU or GPU depending on availability."""
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(
        model_size,
        device=device,
        download_root=str(Path.home() / ".cache" / "whisper"),
    )


def transcribe_words(
    audio: Union[Path, np.ndarray],
    language: Optional[str] = None,
    model: Optional[object] = None,
    min_confidence: float = 0.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> List[Dict]:
    """Transcribe ``audio`` and return word-level timestamps.

    ``audio`` may be a file :class:`Path` or a numpy ``ndarray`` of int16 PCM
    samples. If ``model`` is ``None`` a base Whisper model is loaded.
    ``progress_cb`` is called with each segment's end time in seconds.
    """
    if model is None:
        model = load_whisper_model("base")

    if isinstance(audio, Path):
        audio_input = str(audio)
    else:
        if audio.size == 0:
            raise ValueError("audio array is empty")
        audio_input = audio.astype(np.float32) / 32768

    kwargs = {"word_timestamps": True, "verbose": False}
    if language is not None:
        kwargs["language"] = language
    result = model.transcribe(audio_input, **kwargs)

    words: List[Dict] = []
    for segment in result.get("segments", []):
        if progress_cb and "end" in segment:
            progress_cb(segment["end"])
        for word in segment.get("words", []):
            prob = word.get("probability")
            if prob is not None and prob < min_confidence:
                continue
            words.append(
                {
                    "content": word.get("word", "").strip(),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "probability": prob,
                }
            )
    return words
