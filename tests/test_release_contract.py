import os
from pathlib import Path
import subprocess
import unittest


class ReleaseContractTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    release_tool_directory = repository_root / "scripts" / "release"
    release_tools = (
        "prepare_release.sh",
        "publish_release.sh",
        "release_helper.py",
        "prepare_container_artifact.sh",
        "publish_container_artifacts.sh",
    )

    def test_release_tooling_is_repository_owned(self):
        makefile_source = (self.repository_root / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "RELEASE_TOOL_DIR := $(abspath $(CURDIR)/scripts/release)",
            makefile_source,
        )
        self.assertNotIn("agentSkills/gitrelease", makefile_source)

        for release_tool in self.release_tools:
            release_tool_path = self.release_tool_directory / release_tool
            self.assertTrue(release_tool_path.is_file(), release_tool)
            self.assertTrue(os.access(release_tool_path, os.X_OK), release_tool)

        for wrapper_name, owned_pipeline in (
            ("release.sh", "scripts/release/prepare_release.sh"),
            ("publish-release.sh", "scripts/release/publish_release.sh"),
        ):
            wrapper_source = (self.repository_root / "scripts" / wrapper_name).read_text(encoding="utf-8")
            self.assertIn(owned_pipeline, wrapper_source)
            self.assertNotIn("agentSkills/gitrelease", wrapper_source)
            self.assertNotIn("RELEASE_PIPELINE ]]", wrapper_source)
            self.assertNotIn("PUBLISH_RELEASE_PIPELINE ]]", wrapper_source)

    def test_release_targets_resolve_only_repository_owned_tools(self):
        completed = subprocess.run(
            ("make", "--dry-run", "release", "container-artifacts", "publish"),
            cwd=self.repository_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("scripts/release/prepare_container_artifact.sh", completed.stdout)
        self.assertIn("scripts/release/publish_container_artifacts.sh", completed.stdout)
        self.assertNotIn("agentSkills/gitrelease", completed.stdout)

    def test_release_pipeline_override_is_rejected_by_owned_entrypoint(self):
        environment = os.environ.copy()
        environment["RELEASE_PIPELINE"] = "/tmp/forbidden-release-pipeline"
        completed = subprocess.run(
            ("bash", "scripts/release.sh", "--help"),
            cwd=self.repository_root,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("Usage:", completed.stdout)
        self.assertIn("prepare_release.sh", completed.stdout)
        self.assertNotIn("forbidden-release-pipeline", completed.stdout)

        container_help = subprocess.run(
            (str(self.release_tool_directory / "prepare_container_artifact.sh"), "--help"),
            cwd=self.repository_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(container_help.returncode, 0, container_help.stdout)
        self.assertIn("Usage:", container_help.stdout)


if __name__ == "__main__":
    unittest.main()
