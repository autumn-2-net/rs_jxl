"""Version overrides must not drift between Cargo, Python and release artifacts."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tomlkit

import prepare_ci_version as ci_version


class VersionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        files = {
            "Cargo.toml": '[workspace]\nmembers = ["crates/core", "crates/cli"]\n'
                          '[workspace.package]\nversion = "0.1.0"\n',
            "pyproject.toml": '[project]\nname = "rs-jxl"\nversion = "0.1.0"\n',
            "python/rs_jxl/__init__.py": '"""Package."""\n__version__ = "0.1.0"\n',
            "crates/core/Cargo.toml": '[package]\nname = "rs-jxl"\nversion.workspace = true\n',
            "crates/cli/Cargo.toml": '[package]\nname = "rs-jxl-cli"\nversion.workspace = true\n',
            "Cargo.lock": 'version = 4\n\n'
                          '[[package]]\nname = "rs-jxl"\nversion = "0.1.0"\n\n'
                          '[[package]]\nname = "rs-jxl-cli"\nversion = "0.1.0"\n'
                          'dependencies = ["rs-jxl 0.1.0", "serde"]\n\n'
                          '[[package]]\nname = "serde"\nversion = "1.0.0"\n'
                          'source = "registry+https://example.invalid/index"\nchecksum = "unchanged"\n',
        }
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_empty_input_keeps_all_source_files_untouched(self):
        before = self.snapshot()
        self.assertEqual(ci_version.prepare(self.root, ""), "0.1.0")
        self.assertEqual(self.snapshot(), before)

    def test_override_changes_all_local_versions_but_not_dependencies(self):
        self.assertEqual(ci_version.prepare(self.root, " v1.2.3 "), "1.2.3")
        cargo = tomlkit.parse((self.root / "Cargo.toml").read_text())
        project = tomlkit.parse((self.root / "pyproject.toml").read_text())
        lock = tomlkit.parse((self.root / "Cargo.lock").read_text())
        self.assertEqual(cargo["workspace"]["package"]["version"], "1.2.3")
        self.assertEqual(project["project"]["version"], "1.2.3")
        self.assertEqual([item["version"] for item in lock["package"]], ["1.2.3", "1.2.3", "1.0.0"])
        self.assertEqual(lock["package"][1]["dependencies"], ["rs-jxl 1.2.3", "serde"])
        self.assertEqual(lock["package"][2]["checksum"], "unchanged")
        tree = ast.parse((self.root / "python/rs_jxl/__init__.py").read_text())
        self.assertEqual(tree.body[1].value.value, "1.2.3")

    def test_invalid_input_cannot_write_or_inject_github_environment(self):
        before = self.snapshot()
        for value in ("1.2", "1.2.3rc1", "01.2.3", "1.2.3\nINJECT=1", "$(whoami)", "../1.2.3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ci_version.prepare(self.root, value)
            self.assertEqual(self.snapshot(), before)

    def test_inconsistent_source_versions_are_rejected_without_writes(self):
        path = self.root / "python/rs_jxl/__init__.py"
        path.write_text('__version__ = "9.9.9"\n')
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "disagree"):
            ci_version.prepare(self.root, "1.2.3")
        self.assertEqual(self.snapshot(), before)

    def test_missing_workspace_lock_entry_is_rejected(self):
        path = self.root / "Cargo.lock"
        lock = tomlkit.parse(path.read_text())
        del lock["package"][0]
        path.write_text(tomlkit.dumps(lock))
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "exactly the workspace"):
            ci_version.prepare(self.root, "1.2.3")
        self.assertEqual(self.snapshot(), before)

    def test_github_outputs_use_only_normalized_version(self):
        output = self.root / "github-output"
        environment = self.root / "github-env"
        with patch.object(ci_version, "ROOT", self.root), patch.dict("os.environ", {
            "RS_JXL_BUILD_VERSION": "v1.2.3", "GITHUB_OUTPUT": str(output),
            "GITHUB_ENV": str(environment),
        }):
            ci_version.main()
        self.assertEqual(output.read_text(), "version=1.2.3\n")
        self.assertEqual(environment.read_text(), "RS_JXL_RELEASE_VERSION=1.2.3\n")


if __name__ == "__main__":
    unittest.main()
