import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / ".mprlab/deploy/resources.yml"
CONSUMER_PATH = REPOSITORY_ROOT / "tests/fixtures/go_sdk_consumer/main.go"
EXPECTED_FIELDS = {
    "preset_speaker": "baya",
    "text_format": "SYNTHESIS_TEXT_FORMAT_SSML",
}


def run(command: list[str], directory: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command, cwd=directory, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"SDK consumer command {command!r} failed in {directory}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed.stdout


def verify(source: str) -> dict[str, object]:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    resources = [
        resource for resource in manifest["mprlab_resources"]["resources"]
        if resource["kind"] == "go_module"
    ]
    if len(resources) != 1:
        raise ValueError(f"SDK verification requires one go_module resource, found {len(resources)}")
    resource = resources[0]
    module = resource["module"]
    version = resource["version"]
    environment = {**os.environ, "GOWORK": "off", "GOFLAGS": ""}
    with tempfile.TemporaryDirectory(prefix="dictator-sdk-consumer-") as temporary:
        directory = Path(temporary)
        run(["go", "mod", "init", "example.com/dictator-sdk-consumer"], directory, environment)
        run(["go", "mod", "edit", f"-require={module}@{version}"], directory, environment)
        if source == "local":
            module_root = REPOSITORY_ROOT / resource["source"]
            local_module = json.loads(run(["go", "list", "-m", "-json"], module_root, environment))
            if local_module["Path"] != module:
                raise ValueError(f"SDK source declares another module: {local_module['Path']!r}")
            run(["go", "mod", "edit", f"-replace={module}={module_root}"], directory, environment)
        else:
            # A fresh cache requires the package manager to retrieve the published SDK.
            environment["GOMODCACHE"] = str(directory / "module-cache")
        try:
            download = None
            if source == "released":
                download = json.loads(run(
                    ["go", "mod", "download", "-json", f"{module}@{version}"], directory, environment,
                ))
            shutil.copyfile(CONSUMER_PATH, directory / "main.go")
            fields = json.loads(run(["go", "run", "-mod=mod", "."], directory, environment))
            if fields != EXPECTED_FIELDS:
                raise ValueError(f"SDK consumer fields differ: {fields!r}")
            selected = json.loads(run(["go", "list", "-m", "-json", module], directory, environment))
            if selected["Version"] != version:
                raise ValueError(f"SDK consumer selected another version: {selected!r}")
            return {
                "source": source, "module": module, "version": version,
                "fields": fields, "download": download,
            }
        finally:
            if source == "released":
                run(["go", "clean", "-modcache"], directory, environment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the declared Go SDK through a separate consumer.")
    parser.add_argument("--source", choices=("local", "released"), required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.source), indent=2))


if __name__ == "__main__":
    main()
