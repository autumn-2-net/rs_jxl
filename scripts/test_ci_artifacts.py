"""Small, dependency-free checks for CI artifact layout and required notices."""

import hashlib
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import package_ci


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cli = self.root / "binary"
        self.cli.write_bytes(b"test executable")
        for name in package_ci.NOTICES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        self.dist = self.root / "dist"

    def test_windows_archive_includes_only_binary_and_notices(self):
        path = package_ci.archive_cli(self.root, self.cli, self.dist, "windows-x86_64")
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(set(archive.namelist()), {
                "rs-jxl-windows-x86_64/" + name
                for name in ("rs-jxl.exe", *package_ci.NOTICES)
            })
            self.assertEqual(archive.read("rs-jxl-windows-x86_64/rs-jxl.exe"), self.cli.read_bytes())

    def test_linux_archive_has_executable_mode_and_anonymous_ownership(self):
        path = package_ci.archive_cli(self.root, self.cli, self.dist, "linux-x86_64")
        with tarfile.open(path) as archive:
            self.assertEqual(len(archive.getmembers()), 1 + len(package_ci.NOTICES))
            self.assertEqual(archive.getmember("rs-jxl-linux-x86_64/rs-jxl").mode, 0o755)
            for member in archive.getmembers():
                self.assertTrue(member.isfile())
                self.assertEqual((member.uid, member.gid, member.uname, member.gname), (0, 0, "", ""))

    def test_missing_notice_fails_before_packaging(self):
        (self.root / package_ci.NOTICES[-1]).unlink()
        with self.assertRaises(FileNotFoundError):
            package_ci.archive_cli(self.root, self.cli, self.dist, "windows-x86_64")
        self.assertFalse(self.dist.exists())

    def test_checksums_cover_exact_files(self):
        self.dist.mkdir()
        wheel = self.dist / "test.whl"
        wheel.write_bytes(b"wheel")
        result = package_ci.write_checksums(self.dist, [wheel])
        self.assertEqual(result.read_text(), f"{hashlib.sha256(b'wheel').hexdigest()}  test.whl\n")

    def test_other_platform_is_rejected(self):
        with self.assertRaises(ValueError):
            package_ci.archive_cli(self.root, self.cli, self.dist, "macos")

    def test_versioned_archive_uses_normalized_version(self):
        path = package_ci.archive_cli(self.root, self.cli, self.dist, "windows-x86_64", "v1.2.3")
        self.assertEqual(path.name, "rs-jxl-1.2.3-windows-x86_64.zip")
        with zipfile.ZipFile(path) as archive:
            self.assertIn("rs-jxl-1.2.3-windows-x86_64/rs-jxl.exe", archive.namelist())

    def test_version_cannot_escape_output_directory(self):
        with self.assertRaises(ValueError):
            package_ci.archive_cli(self.root, self.cli, self.dist, "windows-x86_64", "../1.2.3")
        self.assertFalse(self.dist.exists())


if __name__ == "__main__":
    unittest.main()
