"""Small fixtures for release layout, checksum validation and path safety."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from package_ci import write_checksums
import prepare_ci_release as ci_release


VERSION = "1.2.3"


class ReleaseTests(unittest.TestCase):
    def fixture(self, linux_tags="manylinux_2_28_x86_64"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        artifacts = root / "input"
        for platform, extension in ci_release.PLATFORMS.items():
            directory = artifacts / f"rs-jxl-{VERSION}-{platform}"
            directory.mkdir(parents=True)
            tags = "win_amd64" if platform == "windows-x86_64" else linux_tags
            wheel = directory / f"rs_jxl-{VERSION}-py3-none-{tags}.whl"
            archive = directory / f"{directory.name}{extension}"
            wheel.write_bytes(b"wheel fixture: " + platform.encode("ascii"))
            archive.write_bytes(b"archive fixture: " + platform.encode("ascii"))
            write_checksums(directory, [wheel, archive])
        return artifacts, root / "output"

    def platform_dir(self, artifacts, platform="linux-x86_64"):
        return artifacts / f"rs-jxl-{VERSION}-{platform}"

    def assert_rejected(self, artifacts, output, version=VERSION):
        with self.assertRaises((ValueError, OSError)):
            ci_release.prepare(artifacts, output, version)
        self.assertFalse(output.exists())

    def symlink(self, path, target):
        try:
            path.symlink_to(target, target_is_directory=target.is_dir())
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlinks unavailable: {error}")

    def test_flat_output_and_combined_checksums(self):
        artifacts, output = self.fixture()
        before = {path: path.read_bytes() for path in artifacts.glob("*/*")}
        result = ci_release.prepare(artifacts, output, " v1.2.3 ")
        expected = {path.name: data for path, data in before.items()
                    if path.name != "SHA256SUMS"}
        self.assertEqual(set(output.iterdir()), set(result))
        self.assertEqual({path.name for path in result}, set(expected) | {"SHA256SUMS"})
        for name, data in expected.items():
            self.assertEqual((output / name).read_bytes(), data)
        checksums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                            for name, data in sorted(expected.items()))
        self.assertEqual((output / "SHA256SUMS").read_text(), checksums)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_auditwheel_multi_tags_in_either_order(self):
        for tags in ("manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64",
                     "manylinux_2_28_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64"):
            with self.subTest(tags=tags):
                artifacts, output = self.fixture(tags)
                ci_release.prepare(artifacts, output, VERSION)
                self.assertTrue((output / f"rs_jxl-{VERSION}-py3-none-{tags}.whl").is_file())

    def test_wrong_linux_platform_tags(self):
        for tags in ("linux_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_aarch64",
                     "manylinux_2_28_x86_64.win_amd64", "manylinux_2_28_x86_64junk"):
            with self.subTest(tags=tags):
                self.assert_rejected(*self.fixture(tags))

    def test_two_matching_linux_wheels_are_rejected(self):
        artifacts, output = self.fixture()
        directory = self.platform_dir(artifacts)
        extra = directory / f"rs_jxl-{VERSION}-py3-none-manylinux_2_28_x86_64.manylinux2014_x86_64.whl"
        extra.write_bytes(b"second wheel")
        write_checksums(directory, [path for path in directory.iterdir() if path.name != "SHA256SUMS"])
        self.assert_rejected(artifacts, output)

    def test_crlf_checksums_and_uppercase_digests(self):
        artifacts, output = self.fixture()
        for manifest in artifacts.glob("*/SHA256SUMS"):
            lines = manifest.read_text().splitlines()
            manifest.write_bytes("".join(line[:64].upper() + line[64:] + "\r\n"
                                         for line in lines).encode("ascii"))
        self.assertEqual(len(ci_release.prepare(artifacts, output, VERSION)), 5)

    def test_missing_or_unexpected_artifact_files(self):
        for change in ("missing-wheel", "missing-archive", "missing-checksums", "extra-file",
                       "extra-wheel", "wrong-wheel-version", "wrong-windows-tag",
                       "wrong-archive-version", "directory-as-file"):
            with self.subTest(change=change):
                artifacts, output = self.fixture()
                directory = self.platform_dir(artifacts, "windows-x86_64")
                wheel = next(directory.glob("*.whl"))
                archive = next(directory.glob("*.zip"))
                if change.startswith("missing-"):
                    {"missing-wheel": wheel, "missing-archive": archive,
                     "missing-checksums": directory / "SHA256SUMS"}[change].unlink()
                elif change == "extra-file":
                    (directory / "unexpected.txt").write_bytes(b"extra")
                elif change == "extra-wheel":
                    (directory / wheel.name.replace(VERSION, "9.9.9")).write_bytes(b"extra")
                elif change == "wrong-wheel-version":
                    wheel.rename(wheel.with_name(wheel.name.replace(VERSION, "9.9.9")))
                elif change == "wrong-windows-tag":
                    wheel.rename(wheel.with_name(wheel.name.replace("win_amd64", "win32")))
                elif change == "wrong-archive-version":
                    archive.rename(archive.with_name(archive.name.replace(VERSION, "9.9.9")))
                else:
                    wheel.unlink()
                    wheel.mkdir()
                self.assert_rejected(artifacts, output)

    def test_exactly_two_platform_directories(self):
        for change in ("missing", "unexpected", "wrong-version"):
            with self.subTest(change=change):
                artifacts, output = self.fixture()
                directory = self.platform_dir(artifacts)
                if change == "missing":
                    directory.rename(artifacts.parent / "removed")
                elif change == "unexpected":
                    (artifacts / "extra").mkdir()
                else:
                    directory.rename(directory.with_name(directory.name.replace(VERSION, "9.9.9")))
                self.assert_rejected(artifacts, output)

    def test_invalid_checksum_records(self):
        for change in ("empty", "missing", "duplicate", "malformed", "bad-hex", "wrong-hash",
                       "extra", "traversal", "backslash", "absolute", "blank", "non-ascii"):
            with self.subTest(change=change):
                artifacts, output = self.fixture()
                manifest = self.platform_dir(artifacts) / "SHA256SUMS"
                lines = manifest.read_text().splitlines(keepends=True)
                digest = lines[0][:64]
                records = {
                    "empty": "", "missing": lines[0], "duplicate": "".join(lines + [lines[0]]),
                    "malformed": lines[0].replace("  ", " ") + lines[1],
                    "bad-hex": "g" * 64 + lines[0][64:] + lines[1],
                    "wrong-hash": "0" * 64 + lines[0][64:] + lines[1],
                    "extra": "".join(lines) + f"{digest}  unexpected.whl\n",
                    "traversal": f"{digest}  ../outside\n" + lines[1],
                    "backslash": f"{digest}  ..\\outside\n" + lines[1],
                    "absolute": f"{digest}  /outside\n" + lines[1],
                    "blank": "".join(lines) + "\n", "non-ascii": "\xff\n",
                }
                manifest.write_bytes(records[change].encode("latin-1"))
                self.assert_rejected(artifacts, output)

    def test_corruption_on_second_platform_does_not_create_output(self):
        artifacts, output = self.fixture()
        next(self.platform_dir(artifacts).glob("*.whl")).write_bytes(b"changed")
        self.assert_rejected(artifacts, output)

    def test_existing_outputs_are_untouched(self):
        for kind in ("file", "directory", "empty-directory"):
            with self.subTest(kind=kind):
                artifacts, output = self.fixture()
                sentinel = output
                if kind != "file":
                    output.mkdir()
                    sentinel = output / "SHA256SUMS"
                if kind != "empty-directory":
                    sentinel.write_bytes(b"keep")
                with self.assertRaises(FileExistsError):
                    ci_release.prepare(artifacts, output, VERSION)
                if kind != "empty-directory":
                    self.assertEqual(sentinel.read_bytes(), b"keep")
                if kind == "directory":
                    self.assertEqual(list(output.iterdir()), [sentinel])
                elif kind == "empty-directory":
                    self.assertEqual(list(output.iterdir()), [])

    def test_invalid_version_and_path_traversal(self):
        artifacts, output = self.fixture()
        for version in ("", "../1.2.3", "1.2.3\nOTHER=1"):
            with self.subTest(version=version):
                self.assert_rejected(artifacts, output, version)
        self.assert_rejected(artifacts / ".." / "input", output)
        self.assert_rejected(artifacts, artifacts / ".." / "output")
        self.assert_rejected(artifacts, artifacts / "nested-output")

    def test_symlinked_inputs_and_outputs(self):
        for kind in ("root", "platform", "wheel", "checksums", "output", "output-parent",
                     "dangling-output"):
            with self.subTest(kind=kind):
                artifacts, output = self.fixture()
                directory = self.platform_dir(artifacts)
                if kind in ("root", "platform", "wheel", "checksums"):
                    path = {"root": artifacts, "platform": directory,
                            "wheel": next(directory.glob("*.whl")),
                            "checksums": directory / "SHA256SUMS"}[kind]
                    target = artifacts.parent / "moved"
                    path.rename(target)
                    self.symlink(path, target)
                else:
                    target = artifacts.parent / "target"
                    if kind != "dangling-output":
                        target.mkdir()
                        (target / "keep").write_bytes(b"keep")
                    self.symlink(output, target)
                    if kind == "output-parent":
                        output = output / "child"
                with self.assertRaises(ValueError):
                    ci_release.prepare(artifacts, output, VERSION)
                if kind in ("output", "output-parent"):
                    self.assertEqual(list(target.iterdir()), [target / "keep"])
                    self.assertEqual((target / "keep").read_bytes(), b"keep")

    def test_cli_arguments(self):
        artifacts, output = self.fixture()
        with patch("sys.argv", ["prepare_ci_release.py", "--artifacts", str(artifacts),
                                "--output", str(output), "--version", "v1.2.3"]), patch("builtins.print"):
            ci_release.main()
        self.assertEqual(len(list(output.iterdir())), 5)


if __name__ == "__main__":
    unittest.main()
