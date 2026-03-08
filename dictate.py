#!/usr/bin/env python3
"""Upload audio to a Dictator gRPC server and print llm-proxy style dictation output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpc

from dictator.client import DictationClient
from dictator.transport.grpc import ServerConfig


def _default_target_from_config(config: ServerConfig) -> str:
    host = config.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"{host}:{config.port}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--target", default=None, help="gRPC target host:port")
    parser.add_argument("--input", required=True, type=Path, help="input audio file")
    parser.add_argument("--model", default="base", help="transcription model size")
    parser.add_argument("--language", default="", help="language code")
    parser.add_argument("--autodetect-language", action="store_true")
    parser.add_argument("--auth-token", default=None, help="optional Dictator auth token")
    parser.add_argument("--include-words", action="store_true")
    args = parser.parse_args()

    base = ServerConfig.from_sources(config_file=args.config)
    target = args.target or _default_target_from_config(base)
    auth_token = args.auth_token if args.auth_token is not None else base.auth_token

    metadata = ()
    if auth_token:
        metadata = (("x-dictator-token", auth_token),)

    with grpc.insecure_channel(target) as channel:
        client = DictationClient(channel, metadata=metadata)
        result = client.dictate_file(
            args.input,
            model_size=args.model,
            language_code=args.language,
            autodetect_language=args.autodetect_language or not bool(args.language.strip()),
            include_word_segments=args.include_words,
        )
    if args.include_words:
        print(json.dumps({"text": result.text, "words": result.words}, ensure_ascii=False))
    else:
        print(json.dumps(result.to_http_payload(), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
