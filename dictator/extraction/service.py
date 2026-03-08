"""Dominant-speaker reference extraction service."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import time
from pathlib import Path
from typing import Callable, Iterator
import warnings

import librosa
import numpy as np
import torch

from dictator.audio.constants import PCM_SAMPLE_RATE, TARGET_SAMPLE_RATE
from dictator.audio.ffmpeg_ops import decode_pcm, trim_and_normalise
from dictator.diarization import (
    assign_words_to_speakers,
    dominant_speaker_label,
    run_diarization,
)
from dictator.transcription.service import load_whisper_model, transcribe_words

from .models import ReferenceExtractionRequest, ReferenceExtractionResult

SAMPLE_RATE = PCM_SAMPLE_RATE
TARGET_SR = TARGET_SAMPLE_RATE
WIN_SEC = 20.0
STRIDE_SEC = 1.0
MAX_SPEECH_RATE = 4.0
MAX_CENTROID_HZ = 4_000
MIN_CENTROID_HZ = 500
PUBLIC_SIZES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization"
DIARIZATION_MODEL_REVISION = "2.1"
DIARIZATION_MODEL = f"{DIARIZATION_MODEL_ID}@{DIARIZATION_MODEL_REVISION}"
DIARIZATION_TOKEN_ENV = "HF_TOKEN"
PRE_ROLL_SEC = 0.2
POST_ROLL_SEC = 0.2


@contextmanager
def timed(tag: str) -> Iterator[None]:
    started_at = time.perf_counter()
    logging.info("START %s", tag)
    yield
    logging.info("DONE  %s  (delta = %.1fs)", tag, time.perf_counter() - started_at)


def configure_torch_runtime() -> None:
    """Apply runtime tweaks lazily so imports stay side-effect light."""
    # Legacy pyannote 2.1 checkpoints require full pickle loads with Torch 2.6+.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
            cuda_backend.matmul.allow_tf32 = True
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pyannote.audio.utils.reproducibility",
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
    warnings.filterwarnings(
        "ignore",
        message=".*torchaudio._backend.list_audio_backends.*",
    )


def spectral_centroid(samples: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(samples.astype(float)))
    if spec.sum() == 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    return float((spec * freqs).sum() / spec.sum())


def snr(samples: np.ndarray) -> float:
    samples_float = samples.astype(float)
    noise_floor = np.percentile(np.abs(samples_float), 20)
    return samples_float.std() / (noise_floor + 1e-6)


def pitch_variation(samples: np.ndarray) -> float:
    """Estimate RMS energy spread for a window."""
    y = samples.astype(float) / 32768.0
    rms = librosa.feature.rms(y=y)
    return float(rms.std())


def require_diarization_token() -> str:
    """Return the Hugging Face token needed to load the diarization pipeline."""
    token = (os.getenv(DIARIZATION_TOKEN_ENV, "") or "").strip()
    if not token:
        raise RuntimeError(
            f"{DIARIZATION_TOKEN_ENV} environment variable is required to load {DIARIZATION_MODEL}"
        )
    return token


def load_diarization_pipeline() -> object:
    """Load the pyannote diarization pipeline."""
    configure_torch_runtime()
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = require_diarization_token()
    with timed("load_diarization_pipeline"):
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=token,
        )
        pipeline.to(device)
        return pipeline


def apply_diarization_filter(
    words: list[dict[str, object]],
    diarization_pipeline: object,
    audio_file: Path,
) -> list[dict[str, object]]:
    """Keep only words spoken by the dominant diarized speaker."""
    with timed("diarization"):
        speaker_segments = run_diarization(diarization_pipeline, audio_file)
    dominant_speaker = dominant_speaker_label(speaker_segments)
    logging.info("dominant speaker: %s", dominant_speaker)

    diarized_words = assign_words_to_speakers(words, speaker_segments)
    filtered_words: list[dict[str, object]] = []
    for word in diarized_words:
        if word.speaker == dominant_speaker:
            filtered_words.append(word.to_legacy_dict())
    if not filtered_words:
        raise RuntimeError("no words from dominant speaker")
    return filtered_words


def choose_window(
    pcm_array: np.ndarray,
    speaker_words: list[dict[str, object]],
    duration: float,
    max_speech_rate: float = MAX_SPEECH_RATE,
    max_centroid: float = MAX_CENTROID_HZ,
    min_centroid: float = MIN_CENTROID_HZ,
) -> float:
    """Return the start time of the highest-quality window."""
    best_score, best_word_count, best_window_start = -1.0, -1, 0.0
    track_length = len(pcm_array) / SAMPLE_RATE
    if duration > track_length:
        raise RuntimeError(
            f"requested duration {duration:.1f}s exceeds track length {track_length:.1f}s"
        )
    with timed("window_search"):
        position = 0.0
        while position + duration <= track_length:
            chunk = pcm_array[int(position * SAMPLE_RATE) : int((position + duration) * SAMPLE_RATE)]
            centroid = spectral_centroid(chunk)
            if centroid > max_centroid or centroid < min_centroid:
                position += STRIDE_SEC
                continue
            words_in_window = [
                word for word in speaker_words if position <= float(word["start"]) < position + duration
            ]
            word_count = len(words_in_window)
            if word_count == 0:
                position += STRIDE_SEC
                continue
            if word_count / duration > max_speech_rate:
                position += STRIDE_SEC
                continue
            variation = pitch_variation(chunk)
            quality_score = snr(chunk) * (1.0 + variation)
            score = word_count * quality_score
            if score > best_score:
                best_score, best_word_count, best_window_start = score, word_count, position
            position += STRIDE_SEC
    if best_score < 0:
        raise RuntimeError("no suitable window found")
    logging.info("chosen window: %d words, score %.2f", best_word_count, best_score)
    return best_window_start


def compute_trim_bounds(
    total_track_seconds: float,
    window_words: list[dict[str, object]],
    pre_roll_seconds: float = PRE_ROLL_SEC,
    post_roll_seconds: float = POST_ROLL_SEC,
) -> tuple[float, float]:
    """Expand a word window slightly so the clip starts and ends naturally."""
    if not window_words:
        raise RuntimeError("no words found in chosen window")
    window_words.sort(key=lambda word: float(word["start"]))
    first_word_start = float(window_words[0]["start"])
    last_word_end = float(window_words[-1]["end"])
    trim_start = max(0.0, first_word_start - pre_roll_seconds)
    trim_end = min(total_track_seconds, last_word_end + post_roll_seconds)
    return trim_start, trim_end


class ReferenceExtractionService:
    """Service layer for extracting a clean dominant-speaker clip."""

    def extract(
        self,
        request: ReferenceExtractionRequest,
        progress_cb: Callable[[float], None] | None = None,
        model: object | None = None,
        diarization_pipeline: object | None = None,
    ) -> ReferenceExtractionResult:
        configure_torch_runtime()
        pcm = decode_pcm(request.input_path)
        total_track_seconds = len(pcm) / SAMPLE_RATE
        diarization_pipeline = diarization_pipeline or load_diarization_pipeline()
        model = model or load_whisper_model(request.model_size)

        with timed("transcribe"):
            raw_words = transcribe_words(
                pcm,
                language=request.language,
                model=model,
                progress_cb=progress_cb,
            )
        if not raw_words:
            raise RuntimeError("no words transcribed")

        dominant_speaker_words = apply_diarization_filter(
            raw_words,
            diarization_pipeline,
            request.input_path,
        )
        window_start = choose_window(
            pcm,
            dominant_speaker_words,
            request.duration_seconds,
            max_speech_rate=request.max_speech_rate,
            max_centroid=request.max_centroid_hz,
            min_centroid=request.min_centroid_hz,
        )
        window_end = window_start + request.duration_seconds
        window_words = [
            word
            for word in dominant_speaker_words
            if window_start <= float(word["start"]) < window_end
        ]
        trim_start, trim_end = compute_trim_bounds(total_track_seconds, window_words)

        if request.output_path is not None:
            with timed("trim"):
                trim_and_normalise(
                    request.input_path,
                    request.output_path,
                    trim_start,
                    trim_end - trim_start,
                )

        return ReferenceExtractionResult(
            raw_words=tuple(raw_words),
            dominant_speaker_words=tuple(dominant_speaker_words),
            window_start_seconds=window_start,
            window_end_seconds=window_end,
            trim_start_seconds=trim_start,
            trim_end_seconds=trim_end,
            output_path=request.output_path,
        )
