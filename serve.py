#!/usr/bin/env python3
"""Run the Dictator gRPC server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dictator.transport.grpc import ServerConfig, serve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-message-bytes", type=int, default=None)
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--download-chunk-bytes", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    base = ServerConfig.from_env()
    config = ServerConfig(
        host=args.host or base.host,
        port=args.port or base.port,
        max_workers=args.max_workers or base.max_workers,
        max_message_bytes=args.max_message_bytes or base.max_message_bytes,
        max_inflight=args.max_inflight or base.max_inflight,
        download_chunk_bytes=args.download_chunk_bytes or base.download_chunk_bytes,
        artifact_root=args.artifact_root or base.artifact_root,
        auth_token=args.auth_token or base.auth_token,
    )
    serve(config)


if __name__ == "__main__":
    main()
