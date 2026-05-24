"""Shared synthesis request preparation and execution workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from dictator.runtime import ValidationError
from dictator.storage import ArtifactAudioMetadata, ArtifactRecord, LocalArtifactStore

from .models import (
    DEFAULT_SYNTHESIS_AUDIO_FORMAT,
    SILERO_RU_SYNTHESIS_AUDIO_FORMAT,
    SynthesisAudioFormat,
    SynthesisEngine,
    SynthesisRequest,
    SynthesisTextFormat,
)


@dataclass(frozen=True)
class PreparedSynthesisRequest:
    """Validated synthesis request resolved against stored artifacts."""

    speaker_record: ArtifactRecord | None
    synthesis_request: SynthesisRequest
    include_timeline: bool
    audio_format: SynthesisAudioFormat = DEFAULT_SYNTHESIS_AUDIO_FORMAT


@dataclass(frozen=True)
class SynthesisExecutionOutcome:
    """Stored artifacts and metadata produced by a synthesis execution."""

    audio_record: ArtifactRecord
    audio_duration_seconds: float
    audio_format: SynthesisAudioFormat
    chunk_count: int
    timeline_artifact_id: str | None
    timeline_segments: tuple[dict[str, float | str], ...]


def prepare_synthesis_request(
    artifact_store: LocalArtifactStore,
    *,
    speaker_artifact_id: str,
    text: str,
    text_artifact_id: str,
    language_code: str,
    max_duration_seconds: float,
    include_timeline: bool,
    engine: SynthesisEngine,
    speaker_transcript_text: str | None,
    preset_speaker: str | None = None,
    audio_format: SynthesisAudioFormat | None = None,
    text_format: SynthesisTextFormat = SynthesisTextFormat.AUTO,
) -> PreparedSynthesisRequest:
    speaker = None
    if engine == SynthesisEngine.QWEN3:
        if not speaker_artifact_id:
            raise ValidationError(
                "dictator.grpc.voice.speaker_artifact_required",
                "speaker_artifact_id is required for qwen3 synthesis",
            )
        speaker = artifact_store.get_artifact(speaker_artifact_id)
    resolved_text = text
    if text_artifact_id:
        resolved_text = artifact_store.read_text(text_artifact_id)
    if not resolved_text.strip():
        raise ValidationError(
            "dictator.grpc.voice.missing_text",
            "text or text_artifact_id is required",
        )
    return PreparedSynthesisRequest(
        speaker_record=speaker,
        synthesis_request=SynthesisRequest(
            engine=engine,
            speaker_wav=speaker.path if speaker is not None else None,
            text=resolved_text,
            language_code=language_code or ("ru" if engine == SynthesisEngine.SILERO_RU else "en"),
            cap_seconds=max_duration_seconds or None,
            speaker_artifact_id=speaker_artifact_id or None,
            speaker_transcript_text=speaker_transcript_text,
            preset_speaker=preset_speaker,
            audio_format=audio_format,
            text_format=text_format,
        ),
        include_timeline=include_timeline,
        audio_format=audio_format
        or (SILERO_RU_SYNTHESIS_AUDIO_FORMAT if engine == SynthesisEngine.SILERO_RU else DEFAULT_SYNTHESIS_AUDIO_FORMAT),
    )


def execute_synthesis_request(
    *,
    artifact_store: LocalArtifactStore,
    execution_runtime,
    prepared: PreparedSynthesisRequest,
    progress_callback=None,
) -> SynthesisExecutionOutcome:
    from dictator.audio.ffmpeg_ops import concat_normalise
    from dictator.synthesis.service import cleanup_synthesis_result

    synthesis_service = execution_runtime.get_synthesis_service()
    result = None
    try:
        result = synthesis_service.synthesise_text(
            prepared.synthesis_request,
            progress_callback=progress_callback,
        )
        if hasattr(execution_runtime, "mark_synthesis_ready"):
            execution_runtime.mark_synthesis_ready()

        voice_label = (
            Path(prepared.speaker_record.filename).stem
            if prepared.speaker_record is not None
            else (prepared.synthesis_request.preset_speaker or prepared.synthesis_request.engine.value)
        )
        audio_reservation = artifact_store.reserve_artifact(
            f"{voice_label}_synth.wav",
            media_type="audio/wav",
            fallback_suffix=".wav",
        )
        try:
            concat_normalise(
                result.wav_paths,
                audio_reservation.path,
                prepared.synthesis_request.cap_seconds,
                target_sample_rate=prepared.audio_format.sample_rate_hz,
            )
            audio_duration_seconds = result.segments[-1].end_seconds if result.segments else 0.0
            audio_record = artifact_store.finalize_artifact(
                audio_reservation,
                audio_metadata=ArtifactAudioMetadata(
                    container=prepared.audio_format.container,
                    codec=prepared.audio_format.codec,
                    sample_rate_hz=prepared.audio_format.sample_rate_hz,
                    channel_count=prepared.audio_format.channel_count,
                    bit_depth=prepared.audio_format.bit_depth,
                    duration_seconds=audio_duration_seconds,
                ),
            )
        except Exception:
            artifact_store.discard_reservation(audio_reservation)
            raise

        timeline_segments = tuple(segment.to_timeline_dict() for segment in result.segments)
        timeline_artifact_id: str | None = None
        if prepared.include_timeline:
            voice_entry = {
                "id": prepared.synthesis_request.speaker_artifact_id or voice_label,
                "label": voice_label,
                "engine": prepared.synthesis_request.engine.value,
            }
            if prepared.speaker_record is not None:
                voice_entry["file"] = str(prepared.speaker_record.path)
            if prepared.synthesis_request.preset_speaker:
                voice_entry["speaker"] = prepared.synthesis_request.preset_speaker
            timeline_payload = {
                "textSegments": list(timeline_segments),
                "imageCues": [],
                "voices": [voice_entry],
            }
            timeline_record = artifact_store.write_artifact(
                [json.dumps(timeline_payload, ensure_ascii=False, indent=2).encode("utf-8")],
                filename=f"{Path(audio_record.filename).stem}.timeline.json",
                media_type="application/json",
                fallback_suffix=".json",
            )
            timeline_artifact_id = timeline_record.artifact_id
        return SynthesisExecutionOutcome(
            audio_record=audio_record,
            audio_duration_seconds=audio_duration_seconds,
            audio_format=prepared.audio_format,
            chunk_count=len(result.wav_paths),
            timeline_artifact_id=timeline_artifact_id,
            timeline_segments=timeline_segments,
        )
    finally:
        if result is not None:
            cleanup_synthesis_result(result)
