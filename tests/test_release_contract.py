from pathlib import Path
import subprocess
import tempfile
import unittest


class ReleaseContractTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    obsolete_lifecycle_paths = (
        "scripts/release.sh",
        "scripts/publish-release.sh",
        "scripts/deploy.sh",
        "scripts/release",
    )

    def test_lifecycle_targets_delegate_to_physical_sibling_gateway(self):
        makefile_source = (self.repository_root / "Makefile").read_text(encoding="utf-8")
        self.assertIn('gateway_root="$$(dirname "$${application_root}")/mprlab-gateway"', makefile_source)
        self.assertIn('"app-$@"', makefile_source)
        self.assertIn('MPRLAB_APP_ROOT="$${application_root}"', makefile_source)
        self.assertNotIn("GATEWAY_DIR", makefile_source)

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            application_root = workspace_root / "dictator"
            gateway_root = workspace_root / "mprlab-gateway"
            application_root.mkdir()
            gateway_root.mkdir()
            subprocess.run(
                ("git", "init", "--quiet"),
                cwd=application_root,
                check=True,
            )
            (gateway_root / "Makefile").write_text(
                ".PHONY: app-release app-publish app-deploy\n"
                "app-release app-publish app-deploy:\n"
                "\t@:\n",
                encoding="utf-8",
            )

            for target in ("release", "publish", "deploy"):
                completed = subprocess.run(
                    (
                        "make",
                        "--dry-run",
                        "--file",
                        str(self.repository_root / "Makefile"),
                        target,
                    ),
                    cwd=application_root,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)
                self.assertIn(f'"app-{target}"', completed.stdout)
                self.assertIn('MPRLAB_APP_ROOT="${application_root}"', completed.stdout)

    def test_obsolete_application_lifecycle_is_absent(self):
        for relative_path in self.obsolete_lifecycle_paths:
            self.assertFalse((self.repository_root / relative_path).exists(), relative_path)


if __name__ == "__main__":
    unittest.main()
