"""Local-disk artifact storage for gRPC transport adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import shutil
from typing import BinaryIO, Iterable, Iterator
import uuid


_FILENAME_SANITISER = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ArtifactAudioMetadata:
    container: str
    codec: str
    sample_rate_hz: int
    channel_count: int
    bit_depth: int
    duration_seconds: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "container": self.container,
            "codec": self.codec,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_count": self.channel_count,
            "bit_depth": self.bit_depth,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "ArtifactAudioMetadata":
        return cls(
            container=str(payload.get("container") or ""),
            codec=str(payload.get("codec") or ""),
            sample_rate_hz=int(payload.get("sample_rate_hz") or 0),
            channel_count=int(payload.get("channel_count") or 0),
            bit_depth=int(payload.get("bit_depth") or 0),
            duration_seconds=_optional_float(payload.get("duration_seconds")),
        )


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    path: Path
    metadata_path: Path
    audio_metadata: ArtifactAudioMetadata | None = None


@dataclass(frozen=True)
class ArtifactReservation:
    artifact_id: str
    filename: str
    media_type: str
    path: Path
    metadata_path: Path


class LocalArtifactStore:
    """Persist artifacts under a single root directory."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, filename: str | None, fallback_suffix: str = "") -> str:
        candidate = (filename or "artifact").strip() or "artifact"
        if fallback_suffix and not Path(candidate).suffix:
            candidate = f"{candidate}{fallback_suffix}"
        candidate = Path(candidate).name
        candidate = _FILENAME_SANITISER.sub("_", candidate)
        return candidate or f"artifact{fallback_suffix}"

    def _resolve_media_type(self, filename: str, media_type: str | None) -> str:
        if media_type:
            return media_type
        guessed, _ = mimetypes.guess_type(filename)
        return guessed or "application/octet-stream"

    def _artifact_paths(self, artifact_id: str, filename: str) -> tuple[Path, Path]:
        artifact_dir = self.root_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir / filename, artifact_dir / "metadata.json"

    def reserve_artifact(
        self,
        filename: str | None,
        media_type: str | None = None,
        fallback_suffix: str = "",
    ) -> ArtifactReservation:
        artifact_id = uuid.uuid4().hex
        safe_filename = self._safe_filename(filename, fallback_suffix=fallback_suffix)
        resolved_media_type = self._resolve_media_type(safe_filename, media_type)
        path, metadata_path = self._artifact_paths(artifact_id, safe_filename)
        return ArtifactReservation(
            artifact_id=artifact_id,
            filename=safe_filename,
            media_type=resolved_media_type,
            path=path,
            metadata_path=metadata_path,
        )

    def finalize_artifact(
        self,
        reservation: ArtifactReservation,
        *,
        audio_metadata: ArtifactAudioMetadata | None = None,
    ) -> ArtifactRecord:
        if not reservation.path.exists():
            raise FileNotFoundError(reservation.path)
        sha256 = hashlib.sha256(reservation.path.read_bytes()).hexdigest()
        size_bytes = reservation.path.stat().st_size
        resolved_audio_metadata = audio_metadata or self._probe_audio_metadata(reservation)
        payload = {
            "artifact_id": reservation.artifact_id,
            "filename": reservation.filename,
            "media_type": reservation.media_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        if resolved_audio_metadata is not None:
            payload["audio_metadata"] = resolved_audio_metadata.to_json_dict()
        reservation.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ArtifactRecord(
            artifact_id=reservation.artifact_id,
            filename=reservation.filename,
            media_type=reservation.media_type,
            size_bytes=size_bytes,
            sha256=sha256,
            path=reservation.path,
            metadata_path=reservation.metadata_path,
            audio_metadata=resolved_audio_metadata,
        )

    def discard_reservation(self, reservation: ArtifactReservation) -> None:
        shutil.rmtree(reservation.path.parent, ignore_errors=True)

    def write_artifact(
        self,
        chunks: Iterable[bytes],
        filename: str | None,
        media_type: str | None = None,
        fallback_suffix: str = "",
    ) -> ArtifactRecord:
        reservation = self.reserve_artifact(filename, media_type=media_type, fallback_suffix=fallback_suffix)
        with reservation.path.open("wb") as handle:
            for chunk in chunks:
                if chunk:
                    handle.write(chunk)
        return self.finalize_artifact(reservation)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        artifact_dir = self.root_dir / artifact_id
        metadata_path = artifact_dir / "metadata.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        audio_payload = payload.get("audio_metadata")
        audio_metadata = ArtifactAudioMetadata.from_json_dict(audio_payload) if isinstance(audio_payload, dict) else None
        path = artifact_dir / payload["filename"]
        return ArtifactRecord(
            artifact_id=payload["artifact_id"],
            filename=payload["filename"],
            media_type=payload["media_type"],
            size_bytes=int(payload["size_bytes"]),
            sha256=payload["sha256"],
            path=path,
            metadata_path=metadata_path,
            audio_metadata=audio_metadata,
        )

    def open_artifact(self, artifact_id: str) -> tuple[ArtifactRecord, BinaryIO]:
        record = self.get_artifact(artifact_id)
        return record, record.path.open("rb")

    def iter_artifact_chunks(
        self,
        artifact_id: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> Iterator[tuple[ArtifactRecord, int, bytes, bool]]:
        record, handle = self.open_artifact(artifact_id)
        offset = 0
        with handle:
            while True:
                payload = handle.read(chunk_size)
                if not payload:
                    if offset == 0:
                        yield record, offset, b"", True
                    break
                offset += len(payload)
                yield record, offset - len(payload), payload, offset >= record.size_bytes

    def read_text(self, artifact_id: str, encoding: str = "utf-8") -> str:
        record = self.get_artifact(artifact_id)
        return record.path.read_text(encoding=encoding)

    def _probe_audio_metadata(self, reservation: ArtifactReservation) -> ArtifactAudioMetadata | None:
        if not _looks_like_audio(reservation.filename, reservation.media_type):
            return None
        try:
            import ffmpeg

            probe = ffmpeg.probe(str(reservation.path))
        except Exception:
            return None

        streams = probe.get("streams") or ()
        audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
        if not isinstance(audio_stream, dict):
            return None
        format_payload = probe.get("format") if isinstance(probe.get("format"), dict) else {}
        return ArtifactAudioMetadata(
            container=_normalise_container(
                str(format_payload.get("format_name") or Path(reservation.filename).suffix.lstrip("."))
            ),
            codec=str(audio_stream.get("codec_name") or ""),
            sample_rate_hz=int(audio_stream.get("sample_rate") or 0),
            channel_count=int(audio_stream.get("channels") or 0),
            bit_depth=_resolve_bit_depth(audio_stream),
            duration_seconds=_optional_float(audio_stream.get("duration") or format_payload.get("duration")),
        )


def _looks_like_audio(filename: str, media_type: str) -> bool:
    if media_type.startswith("audio/"):
        return True
    return Path(filename).suffix.lower() in {
        ".aac",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
    }


def _normalise_container(format_name: str) -> str:
    first_name = format_name.split(",", 1)[0].strip().lower()
    if first_name == "wav":
        return "wav"
    if first_name in {"mp3", "mp4", "ogg", "webm", "flac", "aiff"}:
        return first_name
    return first_name


def _resolve_bit_depth(audio_stream: dict[str, object]) -> int:
    for key in ("bits_per_sample", "bits_per_raw_sample"):
        value = audio_stream.get(key)
        if value not in (None, "", 0, "0"):
            return int(value)
    codec = str(audio_stream.get("codec_name") or "")
    if codec.startswith("pcm_s16"):
        return 16
    if codec.startswith("pcm_s24"):
        return 24
    if codec.startswith("pcm_s32"):
        return 32
    return 0


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
