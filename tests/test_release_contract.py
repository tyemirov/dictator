import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


class ReleaseTagWorkflowTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    sdk_prefix = "sdk/go/dictatorspeechv1/"

    def setUp(self) -> None:
        workflow = yaml.load(
            (self.repository_root / ".github/workflows/release-gpu.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        self.assertEqual(workflow["on"]["push"]["tags"], ["**"])
        validation_job = workflow["jobs"]["validate-tag"]
        self.assertNotIn("if", validation_job)
        self.validation_step, = [step for step in validation_job["steps"] if "run" in step]
        self.assertEqual(self.validation_step["shell"], "bash")
        self.assertNotIn("if", self.validation_step)
        tests_job = workflow["jobs"]["tests"]
        self.assertEqual(tests_job["needs"], "validate-tag")
        self.assertEqual(tests_job["uses"], "./.github/workflows/test.yml")
        self.assertNotIn("if", tests_job)

        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        workspace = Path(temporary_directory.name)
        self.origin = workspace / "origin"
        self.checkout = workspace / "checkout"
        self.origin.mkdir()
        self.git(self.origin, "init", "--initial-branch=master")
        self.commit(self.origin)
        self.git(workspace, "clone", str(self.origin), str(self.checkout))

    def git(self, directory: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=directory, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    def commit(self, directory: Path) -> None:
        self.git(
            directory, "-c", "user.name=Release test", "-c", "user.email=release@example.com",
            "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Test commit",
        )

    def validate_tag(self, tag: str) -> subprocess.CompletedProcess[str]:
        self.git(self.checkout, "-c", "tag.gpgsign=false", "tag", tag)
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", self.validation_step["run"]],
            cwd=self.checkout, env={**os.environ, "GITHUB_REF_NAME": tag},
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_application_tags_on_master_pass(self) -> None:
        for tag in ("1.2.3", "v1.2.3", "v1.2.3-rc.1"):
            with self.subTest(tag=tag):
                result = self.validate_tag(tag)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(f"Validated tag '{tag}'", result.stdout)

    def test_sdk_tags_on_master_pass(self) -> None:
        for version in ("v1.11.0", "v0.1.0", "v1.12.3"):
            with self.subTest(version=version):
                result = self.validate_tag(self.sdk_prefix + version)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("on origin/master", result.stdout)

    def test_malformed_sdk_tags_fail(self) -> None:
        for version in ("1.11.0", "v1.11", "v01.11.0", "v1.011.0", "v1.11.00", "v1.11.0-rc.1", "v1.11.0+build"):
            with self.subTest(version=version):
                result = self.validate_tag(self.sdk_prefix + version)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Error:", result.stdout)

    def test_unknown_and_malformed_application_tags_fail(self) -> None:
        for tag in ("sdk/go/unknown/v1.11.0", "sdk/go/dictatorspeechv1-extra/v1.11.0", "release", "v1.2", "v1.2.3+build"):
            with self.subTest(tag=tag):
                result = self.validate_tag(tag)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Error:", result.stdout)

    def test_application_and_sdk_tags_outside_master_fail(self) -> None:
        self.git(self.checkout, "checkout", "-b", "unmerged")
        self.commit(self.checkout)
        for tag in ("v1.2.3", self.sdk_prefix + "v1.11.0"):
            with self.subTest(tag=tag):
                result = self.validate_tag(tag)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("not contained in origin/master", result.stdout)


class ReleaseContractTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    obsolete_lifecycle_paths = (
        "scripts/release.sh",
        "scripts/publish-release.sh",
        "scripts/deploy.sh",
        "scripts/release",
    )

    def test_lifecycle_targets_invoke_selected_installed_gateway(self):
        with tempfile.TemporaryDirectory(prefix="dictator lifecycle ") as temporary_root:
            workspace_root = Path(temporary_root)
            application_root = workspace_root / "application checkout"
            application_root.mkdir()
            subprocess.run(["git", "init", "--quiet"], cwd=application_root, check=True)
            probe_source = workspace_root / "probe.go"
            probe_executable = workspace_root / "Gateway probe"
            probe_source.write_text(
                'package main\n'
                'import ("encoding/json"; "os"; "strconv")\n'
                'func main() {\n'
                '  json.NewEncoder(os.Stdout).Encode(os.Args[1:])\n'
                '  status, err := strconv.Atoi(os.Getenv("PROBE_STATUS"))\n'
                '  if err != nil { panic(err) }\n'
                '  os.Exit(status)\n'
                '}\n',
                encoding="utf-8",
            )
            subprocess.run(
                ["go", "build", "-o", str(probe_executable), str(probe_source)],
                cwd=workspace_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            for target in ("release", "publish", "deploy"):
                for status in (0, 17):
                    with self.subTest(target=target, status=status):
                        completed = subprocess.run(
                            ["make", "--no-print-directory", "--file",
                             str(self.repository_root / "Makefile"), target,
                             f"MPRLAB_GATEWAY_EXECUTABLE={probe_executable}"],
                            cwd=application_root,
                            env={**os.environ, "PROBE_STATUS": str(status)},
                            stdin=subprocess.DEVNULL, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        )
                        self.assertEqual(completed.returncode == 0, status == 0, completed.stdout)
                        self.assertEqual(
                            json.loads(completed.stdout.splitlines()[0]),
                            [f"app-{target}", "--app-root", str(application_root.resolve())],
                        )
                        if status:
                            self.assertIn("Error 17", completed.stdout)

    def test_obsolete_application_lifecycle_is_absent(self):
        for relative_path in self.obsolete_lifecycle_paths:
            self.assertFalse((self.repository_root / relative_path).exists(), relative_path)


if __name__ == "__main__":
    unittest.main()
