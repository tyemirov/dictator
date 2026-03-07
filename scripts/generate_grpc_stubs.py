#!/usr/bin/env python3
"""Generate Python protobuf and gRPC stubs for Dictator speech services."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = ROOT / "proto"
OUTPUT_ROOT = ROOT
PROTO_FILES = sorted(str(path.relative_to(PROTO_ROOT)) for path in PROTO_ROOT.rglob("*.proto"))


def ensure_package_markers() -> None:
    for package_dir in [
        ROOT / "dictator" / "speech",
        ROOT / "dictator" / "speech" / "v1",
    ]:
        package_dir.mkdir(parents=True, exist_ok=True)
        init_file = package_dir / "__init__.py"
        init_file.touch(exist_ok=True)


def main() -> int:
    ensure_package_markers()
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={PROTO_ROOT}",
        f"--python_out={OUTPUT_ROOT}",
        f"--grpc_python_out={OUTPUT_ROOT}",
        *PROTO_FILES,
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
