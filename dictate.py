#!/usr/bin/env python3
"""Upload audio to a Dictator gRPC server and print llm-proxy style dictation output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpc

from dictator.client import DictationClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:50051", help="gRPC target host:port")
    parser.add_argument("--input", required=True, type=Path, help="input audio file")
    parser.add_argument("--model", default="base", help="transcription model size")
    parser.add_argument("--language", default="", help="language code")
    parser.add_argument("--auth-token", default="", help="optional Dictator auth token")
    parser.add_argument("--include-words", action="store_true")
    args = parser.parse_args()

    metadata = ()
    if args.auth_token:
        metadata = (("x-dictator-token", args.auth_token),)

    with grpc.insecure_channel(args.target) as channel:
        client = DictationClient(channel, metadata=metadata)
        result = client.dictate_file(
            args.input,
            model_size=args.model,
            language_code=args.language,
            include_word_segments=args.include_words,
        )
    if args.include_words:
        print(json.dumps({"text": result.text, "words": result.words}, ensure_ascii=False))
    else:
        print(json.dumps(result.to_http_payload(), ensure_ascii=False))


if __name__ == "__main__":
    main()
