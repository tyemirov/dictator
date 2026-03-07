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
class ArtifactRecord:
    artifact_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    path: Path
    metadata_path: Path


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

    def finalize_artifact(self, reservation: ArtifactReservation) -> ArtifactRecord:
        if not reservation.path.exists():
            raise FileNotFoundError(reservation.path)
        sha256 = hashlib.sha256(reservation.path.read_bytes()).hexdigest()
        size_bytes = reservation.path.stat().st_size
        payload = {
            "artifact_id": reservation.artifact_id,
            "filename": reservation.filename,
            "media_type": reservation.media_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        reservation.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ArtifactRecord(
            artifact_id=reservation.artifact_id,
            filename=reservation.filename,
            media_type=reservation.media_type,
            size_bytes=size_bytes,
            sha256=sha256,
            path=reservation.path,
            metadata_path=reservation.metadata_path,
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
        path = artifact_dir / payload["filename"]
        return ArtifactRecord(
            artifact_id=payload["artifact_id"],
            filename=payload["filename"],
            media_type=payload["media_type"],
            size_bytes=int(payload["size_bytes"]),
            sha256=payload["sha256"],
            path=path,
            metadata_path=metadata_path,
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
