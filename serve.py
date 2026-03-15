#!/usr/bin/env python3
"""Run the Dictator gRPC server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dictator.transport.grpc import ServerConfig, serve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info(
        "loading gRPC config from %s (exists=%s)",
        args.config,
        args.config.exists(),
    )

    config = ServerConfig.from_sources(
        config_file=args.config,
    )
    if not config.auth_token:
        raise ValueError("gRPC auth token must be configured")
    logging.info(
        "resolved gRPC settings host=%s port=%d artifact_root=%s auth_token=configured",
        config.host,
        config.port,
        config.artifact_root,
    )
    serve(config)


if __name__ == "__main__":  # pragma: no cover
    main()
