#!/usr/bin/env python3
"""Download synthesis model assets into a deterministic local directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download as hf_snapshot_download

DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_SILERO_RU_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"


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


def prefetch_silero_ru(model_url: str, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    torch.hub.download_url_to_file(model_url, str(destination))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qwen3-model-ref", default=DEFAULT_QWEN3_MODEL_ID)
    parser.add_argument("--silero-ru-model-url", default=DEFAULT_SILERO_RU_MODEL_URL)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prefetch_qwen3(args.qwen3_model_ref, output_root / "qwen3")
    prefetch_silero_ru(args.silero_ru_model_url, output_root / "silero" / "v5_5_ru.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
