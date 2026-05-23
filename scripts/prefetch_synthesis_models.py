#!/usr/bin/env python3
"""Download synthesis model assets into a deterministic local directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download as hf_snapshot_download

DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_SILERO_RU_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
DEFAULT_SILERO_RU_MODEL_SHA256 = "50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437"


def _replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prefetch_qwen3(model_ref: str, destination: Path) -> None:
    _replace_dir(destination)
    hf_snapshot_download(
        repo_id=model_ref,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prefetch_silero_ru(model_url: str, model_sha256: str, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    torch.hub.download_url_to_file(model_url, str(destination))
    digest = _sha256_file(destination)
    expected_digest = model_sha256.strip().lower()
    if expected_digest and digest != expected_digest:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"silero_ru model digest mismatch for {destination}: expected {expected_digest}, got {digest}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qwen3-model-ref", default=DEFAULT_QWEN3_MODEL_ID)
    parser.add_argument("--silero-ru-model-url", default=DEFAULT_SILERO_RU_MODEL_URL)
    parser.add_argument("--silero-ru-model-sha256", default=DEFAULT_SILERO_RU_MODEL_SHA256)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prefetch_qwen3(args.qwen3_model_ref, output_root / "qwen3")
    prefetch_silero_ru(
        args.silero_ru_model_url,
        args.silero_ru_model_sha256,
        output_root / "silero" / "v5_5_ru.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
