#!/usr/bin/env python3
"""Force-align audio to an existing transcript over Dictator gRPC and emit SRT."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import grpc

from dictator.client import AlignmentClient
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
    parser.add_argument("--input", required=True, type=Path, help="input audio or video")
    parser.add_argument("--text", required=True, type=Path, help="transcript text or SRT")
    parser.add_argument("--output", required=True, type=Path, help="destination SRT")
    parser.add_argument("--language", default="", help="language code; auto-detected when omitted")
    parser.add_argument("--device", default="auto", help="ignored by the gRPC client; alignment device is server-side")
    parser.add_argument("--auth-token", default=None, help="optional Dictator auth token")
    parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="strip punctuation before alignment",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output.exists() and not args.force:
        if input(f"{args.output} exists - overwrite? [y/N] ").lower() != "y":
            return

    base = ServerConfig.from_sources(config_file=args.config)
    target = args.target or _default_target_from_config(base)
    auth_token = args.auth_token if args.auth_token is not None else base.auth_token

    metadata = ()
    if auth_token:
        metadata = (("x-dictator-token", auth_token),)

    with grpc.insecure_channel(target) as channel:
        client = AlignmentClient(channel, metadata=metadata)
        result = client.align_file(
            args.input,
            transcript_file=args.text,
            language_code=args.language,
            remove_punctuation=args.remove_punctuation,
            include_srt_text=True,
            timeout_seconds=base.job_wait_timeout_seconds,
            poll_interval_seconds=base.job_poll_interval_seconds,
        )
    args.output.write_text(result.srt_text, encoding="utf-8")

    logging.info("aligned %d word(s) in language %s", len(result.words), result.language_code)
    logging.info("saved -> %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
