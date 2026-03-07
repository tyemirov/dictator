"""XTTS-backed speech synthesis service."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from dictator.audio.constants import TARGET_SAMPLE_RATE

from .models import SpeechSegment, SynthesisResult

MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
TARGET_SR = TARGET_SAMPLE_RATE


class XTTSBackend:
    """Lazy XTTS model wrapper so transports can reuse a warm model."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._tts = None

    def load(self):
        if self._tts is None:
            import torch
            from TTS.api import TTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tts = TTS(self.model_id).to(device)
        return self._tts

    def synthesise_to_file(
        self,
        text: str,
        speaker_wav: Path,
        language_code: str,
        output_path: Path,
    ) -> None:
        self.load().tts_to_file(
            text=text,
            speaker_wav=str(speaker_wav),
            language=language_code,
            file_path=str(output_path),
        )


class SpeechSynthesisService:
    """Service layer for request-safe chunked XTTS synthesis."""

    def __init__(self, backend: XTTSBackend | None = None) -> None:
        self.backend = backend or XTTSBackend()

    def synthesise(
        self,
        speaker_wav: Path,
        chunks: Sequence[str],
        cap_seconds: float | None,
        language_code: str,
    ) -> SynthesisResult:
        if not chunks:
            raise ValueError("No text chunks provided")

        temp_dir = Path(tempfile.mkdtemp(prefix="dictator_tts_"))
        wav_paths: list[Path] = []
        segments: list[SpeechSegment] = []
        elapsed = 0.0
        previous_duration = 0.0

        for index, chunk in enumerate(chunks):
            if cap_seconds is not None and elapsed >= cap_seconds:
                break

            wav_path = temp_dir / f"{index:04d}.wav"
            self.backend.synthesise_to_file(
                text=chunk,
                speaker_wav=speaker_wav,
                language_code=language_code,
                output_path=wav_path,
            )

            import soundfile as sf

            info = sf.info(wav_path)
            duration_seconds = info.frames / info.samplerate
            previous_duration = duration_seconds

            if cap_seconds is not None and elapsed + duration_seconds > cap_seconds:
                wav_path.unlink(missing_ok=True)
                logging.warning(
                    "Sentence %.1fs longer than remaining cap (%.1fs) - skipped",
                    duration_seconds,
                    cap_seconds - elapsed,
                )
                break

            wav_paths.append(wav_path)
            segments.append(
                SpeechSegment(
                    text=chunk,
                    start_seconds=elapsed,
                    end_seconds=elapsed + duration_seconds,
                )
            )
            elapsed += duration_seconds
            logging.info(
                "chunk %03d  %.1f s  (cumulative %.1f s)",
                index,
                duration_seconds,
                elapsed,
            )

        if cap_seconds is not None and not wav_paths:
            logging.error(
                "First sentence (%.1fs) exceeds --length - nothing generated",
                previous_duration,
            )
            raise ValueError("No chunks fit within the length cap")

        return SynthesisResult(
            temp_dir=temp_dir,
            wav_paths=tuple(wav_paths),
            segments=tuple(segments),
        )


def cleanup_synthesis_result(result: SynthesisResult) -> None:
    """Remove a synthesis result's temporary directory."""
    shutil.rmtree(result.temp_dir, ignore_errors=True)


def synthesise(
    speaker_wav: Path,
    chunks: Sequence[str],
    cap: float | None,
    language_code: str,
) -> tuple[list[Path], list[dict[str, float | str]]]:
    """Compatibility wrapper returning temporary chunk paths and legacy timeline dicts."""
    result = SpeechSynthesisService().synthesise(
        speaker_wav=speaker_wav,
        chunks=chunks,
        cap_seconds=cap,
        language_code=language_code,
    )
    return list(result.wav_paths), [segment.to_legacy_dict() for segment in result.segments]
