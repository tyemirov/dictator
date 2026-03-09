#!/usr/bin/env python3
"""Download synthesis model assets into deterministic local directories."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download as modelscope_snapshot_download
from TTS.api import TTS
from TTS.utils.manage import ModelManager

DEFAULT_XTTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_COSYVOICE3_MODEL_DIR = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_WETEXT_MODEL_ID = "pengzhendong/wetext"
COSYVOICE3_REQUIRED_FILES = [
    "cosyvoice3.yaml",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "speech_tokenizer_v3.batch.onnx",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt",
    "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
]


def _replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prefetch_xtts(model_ref: str, destination: Path) -> None:
    manager = ModelManager(models_file=TTS.get_models_file_path(), progress_bar=True)
    model_path, _, _ = manager.download_model(model_ref)
    model_path = Path(model_path)
    source_dir = model_path if model_path.is_dir() else model_path.parent
    if not source_dir.is_dir():
        raise RuntimeError(f"XTTS download did not produce a model directory: {source_dir}")
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source_dir, destination)


def prefetch_qwen3(model_ref: str, destination: Path) -> None:
    _replace_dir(destination)
    hf_snapshot_download(
        repo_id=model_ref,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )


def prefetch_cosyvoice3(model_ref: str, destination: Path) -> None:
    _replace_dir(destination)
    modelscope_snapshot_download(
        model_id=model_ref,
        local_dir=str(destination),
        allow_file_pattern=COSYVOICE3_REQUIRED_FILES,
    )


def prefetch_wetext(model_ref: str) -> None:
    modelscope_snapshot_download(model_id=model_ref)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--xtts-model-ref", default=DEFAULT_XTTS_MODEL_ID)
    parser.add_argument("--qwen3-model-ref", default=DEFAULT_QWEN3_MODEL_ID)
    parser.add_argument("--cosyvoice3-model-ref", default=DEFAULT_COSYVOICE3_MODEL_DIR)
    parser.add_argument("--wetext-model-ref", default=DEFAULT_WETEXT_MODEL_ID)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prefetch_xtts(args.xtts_model_ref, output_root / "xtts")
    prefetch_qwen3(args.qwen3_model_ref, output_root / "qwen3")
    prefetch_cosyvoice3(args.cosyvoice3_model_ref, output_root / "cosyvoice3")
    prefetch_wetext(args.wetext_model_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
