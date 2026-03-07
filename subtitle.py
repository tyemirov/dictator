#!/usr/bin/env python3
"""Upload audio to a Dictator gRPC server and render grouped SRT subtitles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpc

from dictator.client import SubtitleClient
from dictator.transport.grpc import ServerConfig


def _default_target_from_config(config: ServerConfig) -> str:
    host = config.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"{host}:{config.port}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--target", default=None, help="gRPC target host:port")
    parser.add_argument("--input", required=True, type=Path, help="input audio file")
    parser.add_argument("--model", default="base", help="transcription model size")
    parser.add_argument("--language", default="", help="language code")
    parser.add_argument("--autodetect-language", action="store_true")
    parser.add_argument("--auth-token", default=None, help="optional Dictator auth token")
    parser.add_argument("--granularity", choices=("words", "sentences"), default="words")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="optional output .srt path")
    parser.add_argument("--source-text", default=None, help="optional inline source text for forced alignment")
    parser.add_argument("--source-text-name", default="", help="logical source-text filename")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--source-text-file", type=Path, default=None, help="optional source transcript file")
    args = parser.parse_args()

    base = ServerConfig.from_sources(
        config_file=args.config,
        env_file=args.env_file,
    )
    target = args.target or _default_target_from_config(base)
    auth_token = args.auth_token if args.auth_token is not None else base.auth_token

    metadata = ()
    if auth_token:
        metadata = (("x-dictator-token", auth_token),)

    with grpc.insecure_channel(target) as channel:
        client = SubtitleClient(channel, metadata=metadata)
        result = client.render_file(
            args.input,
            model_size=args.model,
            language_code=args.language,
            autodetect_language=args.autodetect_language or not bool(args.language.strip()),
            granularity=args.granularity,
            group_size=args.group_size,
            source_text=args.source_text,
            source_text_file=args.source_text_file,
            source_text_name=args.source_text_name,
            include_srt_text=True,
        )
    if args.output is not None:
        args.output.write_text(result.srt_text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "languageCode": result.language_code,
                    "mode": result.mode,
                    "srtArtifactId": result.srt_artifact_id,
                    "output": str(args.output),
                },
                ensure_ascii=False,
            )
        )
        return
    print(result.srt_text, end="")


if __name__ == "__main__":  # pragma: no cover
    main()
