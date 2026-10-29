"""Packaging-only tests: synthetic headers, temporary projects, no Cargo builds."""

import email
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from packaging.version import Version
import pytest
import tomlkit


ROOT = Path(__file__).resolve().parents[2]
EMBEDDED_ASSETS = (
    "vendor/jxl-rs/jxl/src/util/dither_32x32.bin",
    "crates/rs-jxl/tests/srgb_profile.hex",
)


@pytest.fixture
def packaging(monkeypatch):
    # Build helpers stay in the source tree even when rs_jxl comes from a wheel.
    monkeypatch.syspath_prepend(str(ROOT))
    import build_support.commands
    import build_support.native
    import build_support.sources

    for name in ("RS_JXL_NATIVE_LIB", "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET",
                 "_PYTHON_HOST_PLATFORM", "ARCHFLAGS"):
        monkeypatch.delenv(name, raising=False)
    return build_support


def native_header(system, arch="x86_64"):
    """Not loadable code: only enough header data for format/architecture checks."""
    bits = 32 if arch == "i686" else 64
    if system == "win32":
        data = bytearray(512)
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 0x3C, 128)
        data[128:132] = b"PE\0\0"
        struct.pack_into("<H", data, 132, {"x86_64": 0x8664, "i686": 0x14C}[arch])
        struct.pack_into("<HHH", data, 148, 240 if bits == 64 else 224, 0x2002,
                         0x20B if bits == 64 else 0x10B)
    elif system == "linux":
        data = bytearray(64)
        data[:7] = b"\x7fELF" + bytes([2 if bits == 64 else 1, 1, 1])
        struct.pack_into("<HHI", data, 16, 3, 62 if bits == 64 else 3, 1)
    elif system == "darwin":
        data = bytearray(64)
        data[:4] = b"\xcf\xfa\xed\xfe" if bits == 64 else b"\xce\xfa\xed\xfe"
        struct.pack_into("<III", data, 4, 0x1000007 if bits == 64 else 7, 3, 6)
    else:
        raise AssertionError(system)
    return bytes(data)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def forbid_subprocess(*args, **kwargs):
    raise AssertionError("Packaging must not execute a subprocess in this test")


@pytest.mark.parametrize("system", ["win32", "linux", "darwin"])
def test_prebuilt_validates_native_header_without_execution(packaging, monkeypatch, tmp_path, system):
    monkeypatch.setattr(packaging.native.sys, "platform", system)
    monkeypatch.setattr(packaging.native, "_host_arch", lambda: ("x86_64", 64))
    candidate = write(tmp_path / "prebuilt" / packaging.native._native_name(), native_header(system))
    monkeypatch.setenv("RS_JXL_NATIVE_LIB", str(candidate))
    monkeypatch.setattr(packaging.native.subprocess, "run", forbid_subprocess)
    assert packaging.native._native_library(tmp_path) == candidate.resolve()


@pytest.mark.parametrize("system", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("damage", ["architecture", "foreign-format", "truncated", "executable"])
def test_prebuilt_rejects_incompatible_headers(packaging, monkeypatch, tmp_path, system, damage):
    monkeypatch.setattr(packaging.native.sys, "platform", system)
    monkeypatch.setattr(packaging.native, "_host_arch", lambda: ("x86_64", 64))
    data = bytearray(native_header(system, "i686" if damage == "architecture" else "x86_64"))
    if damage == "foreign-format":
        data = native_header("linux" if system != "linux" else "win32")
    elif damage == "truncated":
        data = data[:20]
    elif damage == "executable":
        if system == "win32":
            struct.pack_into("<H", data, 150, 2)
        elif system == "linux":
            struct.pack_into("<H", data, 16, 2)
        else:
            struct.pack_into("<I", data, 12, 2)
    candidate = write(tmp_path / packaging.native._native_name(), bytes(data))
    monkeypatch.setenv("RS_JXL_NATIVE_LIB", str(candidate))
    monkeypatch.setattr(packaging.native.subprocess, "run", forbid_subprocess)
    with pytest.raises(RuntimeError, match="Invalid native format/architecture"):
        packaging.native._native_library(tmp_path)


@pytest.mark.parametrize("value, message", [("", "must name"), ("wrong.dll", "must be named"),
                                            (None, "missing")])
def test_prebuilt_never_falls_back_to_cargo(packaging, monkeypatch, tmp_path, value, message):
    monkeypatch.setenv("RS_JXL_NATIVE_LIB", value if value is not None else packaging.native._native_name())
    monkeypatch.setattr(packaging.native.subprocess, "run", forbid_subprocess)
    with pytest.raises(RuntimeError, match=message):
        packaging.native._native_library(tmp_path)


def test_prebuilt_relative_to_repo_not_caller(packaging, monkeypatch, tmp_path):
    candidate = write(tmp_path / "prebuilt" / packaging.native._native_name(), native_header(sys.platform))
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("RS_JXL_NATIVE_LIB", "prebuilt/" + candidate.name)
    monkeypatch.setattr(packaging.native.subprocess, "run", forbid_subprocess)
    assert packaging.native._native_library(tmp_path) == candidate.resolve()


@pytest.mark.parametrize("system,target", [("win32", "x86_64-pc-windows-msvc"),
                                           ("linux", "x86_64-unknown-linux-gnu"),
                                           ("darwin", "x86_64-apple-darwin")])
def test_native_target_accepted(packaging, monkeypatch, system, target):
    monkeypatch.setattr(packaging.native.sys, "platform", system)
    monkeypatch.setattr(packaging.native, "_host_arch", lambda: ("x86_64", 64))
    monkeypatch.setattr(packaging.native.platform, "libc_ver", lambda: ("glibc", "2.31"))
    monkeypatch.setenv("CARGO_BUILD_TARGET", target)
    assert packaging.native._build_target() == target


@pytest.mark.parametrize("target", ["i686-pc-windows-msvc", "x86_64-unknown-linux-gnu",
                                     "x86_64-apple-darwin", "custom.json"])
def test_cross_target_rejected_even_for_prebuilt(packaging, monkeypatch, tmp_path, target):
    monkeypatch.setattr(packaging.native.sys, "platform", "win32")
    monkeypatch.setattr(packaging.native, "_host_arch", lambda: ("x86_64", 64))
    monkeypatch.setenv("CARGO_BUILD_TARGET", target)
    monkeypatch.setenv("RS_JXL_NATIVE_LIB", "rs_jxl_ffi.dll")
    monkeypatch.setattr(packaging.native.subprocess, "run", forbid_subprocess)
    with pytest.raises(RuntimeError, match="Unsupported cross-build"):
        packaging.native._native_library(tmp_path)


@pytest.mark.parametrize("name", ["ARCHFLAGS", "_PYTHON_HOST_PLATFORM"])
def test_cross_tag_environment_rejected(packaging, monkeypatch, name):
    monkeypatch.setenv(name, "foreign-platform")
    with pytest.raises(RuntimeError, match="Cross-build override"):
        packaging.native._build_target()


def fixture_project(tmp_path, packaging, version=None):
    root = tmp_path / "project with spaces"
    root.mkdir()
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in"):
        shutil.copy2(ROOT / name, root / name)
    for name in packaging.sources.BUILD_SUPPORT_FILES:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    if version is not None:
        manifest = root / "pyproject.toml"
        project = tomlkit.parse(manifest.read_text(encoding="utf-8"))
        project["project"]["version"] = version
        manifest.write_text(tomlkit.dumps(project), encoding="utf-8")
    for name in packaging.sources.REQUIRED_SOURCES:
        if not (root / name).exists():
            write(root / name, "# Packaging fixture\n" if not name.endswith(".rs") else "// Fixture\n")
    # Real, distinct BSD/Apache texts and module license references catch filename
    # collisions that identical placeholder notices would silently conceal.
    for name in packaging.sources.LICENSE_FILES + EMBEDDED_ASSETS:
        shutil.copy2(ROOT / name, root / name)
    write(root / "python/rs_jxl/__init__.py", "fixture_value = 1\n")
    write(root / "python/rs_jxl/py.typed", "")
    write(root / "python/tests/test_fixture.py", "def test_fixture():\n    assert True\n")
    write(root / "docs/PYTHON_INTERFACE.md", "Install torch manually for tensor support.\n")
    return root


@pytest.mark.parametrize("target_dir", [None, "custom output", "absolute"])
@pytest.mark.parametrize("explicit_target", [False, True])
def test_cargo_command_and_deterministic_artifact(packaging, monkeypatch, tmp_path,
                                                target_dir, explicit_target):
    root = fixture_project(tmp_path, packaging)
    monkeypatch.setattr(packaging.native.sys, "platform", "win32")
    monkeypatch.setattr(packaging.native, "_host_arch", lambda: ("x86_64", 64))
    target = "x86_64-pc-windows-msvc" if explicit_target else None
    if target:
        monkeypatch.setenv("CARGO_BUILD_TARGET", target)
    output = root / "target"
    if target_dir:
        configured = tmp_path / "absolute output" if target_dir == "absolute" else Path(target_dir)
        monkeypatch.setenv("CARGO_TARGET_DIR", str(configured))
        output = configured if configured.is_absolute() else root / configured
    candidate = output / target / "release" if target else output / "release"
    candidate = write(candidate / "rs_jxl_ffi.dll", native_header("win32"))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps({
            "reason": "compiler-artifact", "target": {"name": "rs_jxl_ffi", "crate_types": ["cdylib"]},
            "filenames": [str(candidate)],
        }) + "\n")

    monkeypatch.setattr(packaging.native.subprocess, "run", run)
    monkeypatch.chdir(tmp_path.parent)
    assert packaging.native._native_library(root) == candidate.resolve()
    assert len(calls) == 1
    command, options = calls[0]
    assert command[:8] == ["cargo", "build", "--locked", "--offline", "--release", "-p",
                           "rs-jxl-ffi", "--target-dir"]
    assert command[8] == str(output.resolve())
    assert "--message-format=json-render-diagnostics" in command
    if target:
        assert command[-2:] == ["--target", target]
    else:
        assert "--target" not in command
    assert options["check"] is True
    assert options["cwd"] == root
    assert not options.get("shell", False)


def test_cargo_cannot_silently_reuse_stale_artifact(packaging, monkeypatch, tmp_path):
    root = fixture_project(tmp_path, packaging)
    write(root / "target/release" / packaging.native._native_name(), native_header(sys.platform))
    monkeypatch.setattr(packaging.native.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="{}\n"))
    with pytest.raises(RuntimeError, match="No stale library"):
        packaging.native._native_library(root)


@pytest.mark.parametrize("failure", [FileNotFoundError("cargo"), subprocess.CalledProcessError(1, ["cargo"])])
def test_cargo_failure_explains_offline_requirement(packaging, monkeypatch, tmp_path, failure):
    root = fixture_project(tmp_path, packaging)

    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(packaging.native.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Locked offline Cargo build failed"):
        packaging.native._native_library(root)


def test_incomplete_sources_are_fatal(packaging, monkeypatch, tmp_path):
    monkeypatch.setattr(packaging.native.subprocess, "run", forbid_subprocess)
    with pytest.raises(RuntimeError, match="Incomplete source packaging") as error:
        packaging.native._native_library(tmp_path)
    assert "crates/rs-jxl-ffi/Cargo.toml" in str(error.value)
    assert "vendor/jxl-rs/jxl/src/util/dither_32x32.bin" in str(error.value)


def test_build_py_overwrites_old_binary_and_editable_does_not_copy(packaging, monkeypatch, tmp_path):
    root = fixture_project(tmp_path, packaging)
    monkeypatch.setattr(packaging.commands, "ROOT", root)
    native = write(root / "target/release" / packaging.native._native_name(), native_header(sys.platform))
    monkeypatch.setattr(packaging.native, "_native_library", lambda root: native)
    monkeypatch.setattr(packaging.commands._build_py, "run", lambda self: None)
    distribution = packaging.commands.BinaryDistribution({"packages": ["rs_jxl"]})
    command = packaging.commands.build_py(distribution)
    command.ensure_finalized()
    command.build_lib = str(root / "build/lib")
    bundled = write(Path(command.build_lib) / "rs_jxl" / native.name, b"stale")
    os.utime(native, (1, 1))
    command.run()
    assert bundled.read_bytes() == native.read_bytes()
    command.build_lib = str(root / "editable-output")
    command.editable_mode = True
    command.run()
    assert not Path(command.build_lib).exists()
    assert not (root / "python/rs_jxl" / native.name).exists()


def test_wheel_tag_and_cross_override(packaging):
    distribution = packaging.commands.BinaryDistribution({"name": "rs-jxl", "version": "0.1.0"})
    command = packaging.commands.bdist_wheel(distribution)
    command.ensure_finalized()
    assert command.root_is_pure is False
    python_tag, abi_tag, platform_tag = command.get_tag()
    assert (python_tag, abi_tag) == ("py3", "none")
    assert platform_tag != "any"
    command = packaging.commands.bdist_wheel(packaging.commands.BinaryDistribution())
    command.plat_name = "unsupported_platform"
    with pytest.raises(RuntimeError, match="Cross-build --plat-name"):
        command.ensure_finalized()


def run_backend(root, operation, native=None, error=None):
    env = os.environ.copy()
    for key in ("RS_JXL_NATIVE_LIB", "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET",
                "_PYTHON_HOST_PLATFORM", "ARCHFLAGS"):
        env.pop(key, None)
    env["SOURCE_DATE_EPOCH"] = "1700000000"
    env["PYTHONHASHSEED"] = "0"
    if native:
        env["RS_JXL_NATIVE_LIB"] = str(native)
    # Even an accidental Cargo invocation fails before executing anything.
    code = (
        "from setuptools import build_meta\n"
        "from pathlib import Path\n"
        "import sys\n"
        "sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != Path.cwd()]\n"
        "import subprocess\n"
        "def forbidden(*a, **kw):\n"
        "    raise AssertionError('Packaging fixture must never run Cargo')\n"
        "subprocess.run = forbidden\n"
        f"build_meta.{operation}('dist')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if error:
        assert result.returncode != 0
        assert error in result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    suffix = "*.tar.gz" if operation == "build_sdist" else "*.whl"
    artifacts = list((root / "dist").glob(suffix))
    assert len(artifacts) == 1
    return artifacts[0]


def inspect_notices(archive, root, packaging):
    names = archive.namelist()
    metadata_path = next(name for name in names if name.endswith("/METADATA"))
    metadata = email.message_from_bytes(archive.read(metadata_path))
    info = metadata_path.rsplit("/", 1)[0]
    assert {Path(name).as_posix() for name in metadata.get_all("License-File", [])} == set(packaging.sources.LICENSE_FILES)
    assert metadata.get("License") is None
    assert metadata.get("License-Expression") is None
    assert not any(value.startswith("License ::") for value in metadata.get_all("Classifier", []))
    for name in packaging.sources.LICENSE_FILES:
        assert archive.read(f"{info}/licenses/{name}") == (root / name).read_bytes()
        assert f"{info}/{Path(name).name}" not in names
    assert b"Apache License" in archive.read(f"{info}/licenses/vendor/libsais-rs/LICENSE")
    assert b"Redistribution and use" in archive.read(f"{info}/licenses/vendor/jxl-rs/LICENSE")
    assert b"Retained Upstream License" in archive.read(f"{info}/licenses/docs/PORTING_KERNELS.md")


def inspect_wheel(path, native, root, packaging, expected_version):
    assert "-py3-none-" in path.name
    assert not path.name.endswith("-any.whl")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert archive.read("rs_jxl/" + native.name) == native.read_bytes()
        assert "rs_jxl/__init__.py" in names
        assert "rs_jxl/py.typed" in names
        assert not any("scripts/" in name or name.startswith(
            ("crates/", "vendor/", "tests/", "build_support/")) for name in names)
        wheel = archive.read(next(name for name in names if name.endswith("/WHEEL"))).decode()
        assert "Root-Is-Purelib: false" in wheel
        assert "Tag: py3-none-" in wheel
        metadata = email.message_from_bytes(archive.read(next(name for name in names if name.endswith("/METADATA"))))
        assert metadata["Name"] == "rs-jxl"
        assert metadata["Version"] == expected_version, "Wheel metadata version differs from the build input"
        assert parse_wheel_filename(path.name)[1] == Version(expected_version)
        assert metadata["Requires-Python"] == ">=3.10"
        assert {str(Requirement(value)) for value in metadata.get_all("Requires-Dist")} == {"cffi>=1.15", "numpy>=1.22"}
        assert metadata.get_all("Provides-Extra", []) == []
        inspect_notices(archive, root, packaging)


@pytest.mark.parametrize("version", [None, "1.0.0", "2.7.3"],
                         ids=["project-version", "release-1.0.0", "release-2.7.3"])
def test_real_wheel_and_sdist_roundtrip_without_cargo(packaging, tmp_path, version):
    root = fixture_project(tmp_path, packaging, version)
    expected_version = tomlkit.parse((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    native = write(tmp_path / "prebuilt" / packaging.native._native_name(), native_header(sys.platform))
    required_extra = ["crates/rs-jxl/src/nested/module.rs", "vendor/libsais-rs/src/libsais64.rs",
                      "vendor/jxl-rs/jxl/src/nested/module.rs", "python/tests/test_fixture.py",
                      "docs/NEW_GUIDE.md", "docs/nested/INDEX.md"]
    for name in required_extra:
        if not (root / name).exists():
            write(root / name, "// fixture\n")
    excluded = [
        "target/release/leak.rs", "out/leak.py", "private/leak.toml", "data/leak.rs",
        "crates/rs-jxl/private/leak.rs", "crates/rs-jxl/src/data/leak.rs",
        "vendor/jxl-rs/target/release/build/leak.rs", "vendor/libsais-rs/target/debug/leak.rs",
        "vendor/jxl-rs/jxl/target/debug/build/leak.rs", "vendor/jxl-rs/.git/objects/leak.rs",
        "vendor/jxl-rs/jxl/.cache/nested/leak.py", "vendor/libsais-rs/.pytest_cache/leak.toml",
        "vendor/libsais-rs/private/leak.rs", "python/rs_jxl/__pycache__/leak.py",
        "python/rs_jxl/private/leak.py", "python/rs_jxl/.env", "vendor/jxl-rs/.env.secret",
        "python/rs_jxl/old.dll", "python/rs_jxl/leak.key", "vendor/jxl-rs/jxl/resources/leak.rs",
        "docs/private/leak.md", "docs/data/leak.md", "docs/.cache/leak.md",
        "build_support/__pycache__/leak.py", "build_support/private/leak.py",
    ]
    for name in excluded:
        write(root / name, "not for distribution\n")
    source = run_backend(root, "build_sdist")
    extracted = tmp_path / "extracted"
    with tarfile.open(source) as archive:
        names = {"/".join(Path(member.name).parts[1:]) for member in archive.getmembers() if member.isfile()}
        assert set(packaging.sources.REQUIRED_SOURCES).issubset(names)
        assert set(required_extra).issubset(names)
        assert not set(excluded).intersection(names)
        prefix = archive.getmembers()[0].name.split("/")[0]
        metadata = email.message_from_bytes(archive.extractfile(f"{prefix}/PKG-INFO").read())
        assert metadata["Version"] == expected_version
        for name in packaging.sources.LICENSE_FILES + EMBEDDED_ASSETS + packaging.sources.BUILD_SUPPORT_FILES:
            assert archive.extractfile(f"{prefix}/{name}").read() == (root / name).read_bytes()
        for name in packaging.sources.BUILD_SUPPORT_FILES:
            assert (root / name).read_bytes() == (ROOT / name).read_bytes()
        # This archive was just produced from the test's own temporary fixture.
        archive.extractall(extracted)
    wheel = run_backend(root, "build_wheel", native)
    inspect_wheel(wheel, native, root, packaging, expected_version)
    # The check must still reject a stale or otherwise mismatched wheel version.
    with pytest.raises(AssertionError, match="Wheel metadata version"):
        inspect_wheel(wheel, native, root, packaging, expected_version + ".post1")
    unpacked = next(extracted.iterdir())
    rebuilt = run_backend(unpacked, "build_wheel", native)
    inspect_wheel(rebuilt, native, unpacked, packaging, expected_version)
    assert hashlib.sha256(wheel.read_bytes()).digest() == hashlib.sha256(rebuilt.read_bytes()).digest()


def test_sdist_missing_vendor_attribution_is_fatal(packaging, monkeypatch, tmp_path):
    root = fixture_project(tmp_path, packaging)
    (root / "vendor/libsais-rs/LICENSE").unlink()
    monkeypatch.setattr(packaging.commands, "ROOT", root)
    command = packaging.commands.sdist(packaging.commands.BinaryDistribution())
    with pytest.raises(RuntimeError, match="vendor/libsais-rs/LICENSE"):
        command.run()


def test_sdist_manifest_omission_is_fatal(packaging):
    command = packaging.commands.sdist(packaging.commands.BinaryDistribution())
    command.filelist = SimpleNamespace(
        files=[name for name in packaging.sources.REQUIRED_SOURCES if name != "Cargo.lock"])
    with pytest.raises(RuntimeError, match="Incomplete sdist manifest; missing:\n  Cargo.lock"):
        command.make_distribution()


def test_pep660_editable_leaves_native_library_for_source_loader(packaging, tmp_path):
    root = fixture_project(tmp_path, packaging)
    native = write(root / "target/release" / packaging.native._native_name(), native_header(sys.platform))
    wheel = run_backend(root, "build_editable", native)
    with zipfile.ZipFile(wheel) as archive:
        assert any(name.endswith(".pth") for name in archive.namelist())
        assert not any(name.endswith(native.name) for name in archive.namelist())
        assert not any(name.startswith("build_support/") for name in archive.namelist())
        inspect_notices(archive, root, packaging)
    assert native.is_file()
    assert not (root / "python/rs_jxl" / native.name).exists()


def test_windows_architecture_does_not_launch_platform_probe(packaging, monkeypatch):
    monkeypatch.setattr(packaging.native.sys, "platform", "win32")
    monkeypatch.setattr(packaging.native.sysconfig, "get_platform", lambda: "win-amd64")
    monkeypatch.setattr(packaging.native.struct, "calcsize", lambda fmt: 8)
    monkeypatch.setattr(packaging.native.platform, "machine", forbid_subprocess)
    assert packaging.native._host_arch() == ("x86_64", 64)


def test_pe_offset_and_optional_header_are_bounded(packaging, monkeypatch, tmp_path):
    monkeypatch.setattr(packaging.native.sys, "platform", "win32")
    monkeypatch.setattr(packaging.native, "_host_arch", lambda: ("x86_64", 64))
    data = bytearray(native_header("win32"))
    struct.pack_into("<I", data, 0x3C, 0xFFFFFFFF)
    path = write(tmp_path / "rs_jxl_ffi.dll", bytes(data))
    with pytest.raises(RuntimeError, match="Invalid native format/architecture"):
        packaging.native._validate_native(path)
    data = bytearray(native_header("win32"))
    struct.pack_into("<H", data, 148, 0xFFFF)
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="Invalid native format/architecture"):
        packaging.native._validate_native(path)


def test_cross_libc_target_rejected(packaging, monkeypatch):
    monkeypatch.setattr(packaging.native.sys, "platform", "linux")
    monkeypatch.setattr(packaging.native, "_host_arch", lambda: ("x86_64", 64))
    monkeypatch.setattr(packaging.native.platform, "libc_ver", lambda: ("glibc", "2.31"))
    monkeypatch.setenv("CARGO_BUILD_TARGET", "x86_64-unknown-linux-musl")
    with pytest.raises(RuntimeError, match="Unsupported cross-build"):
        packaging.native._build_target()


def test_skip_build_cannot_bypass_native_validation(packaging):
    command = packaging.commands.bdist_wheel(packaging.commands.BinaryDistribution())
    command.skip_build = True
    with pytest.raises(RuntimeError, match="native validation must not be skipped"):
        command.run()


def test_metadata_only_does_not_require_native_build(packaging, tmp_path):
    root = fixture_project(tmp_path, packaging)
    # Metadata preparation must work even before the native crate has landed.
    (root / "crates/rs-jxl-ffi/Cargo.toml").unlink()
    code = (
        "from setuptools import build_meta\n"
        "import subprocess\n"
        "def forbidden(*a, **kw):\n"
        "    raise AssertionError('metadata must not invoke Cargo')\n"
        "subprocess.run = forbidden\n"
        "build_meta.prepare_metadata_for_build_wheel('.')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(root.glob("*.dist-info/METADATA"))) == 1


@pytest.mark.parametrize("operation", ["build_wheel", "build_editable"])
def test_backend_cannot_suppress_native_failure(packaging, tmp_path, operation):
    root = fixture_project(tmp_path, packaging)
    native = write(tmp_path / "prebuilt" / packaging.native._native_name(), b"not a native library")
    run_backend(root, operation, native, error="Invalid native format/architecture")
    assert not list((root / "dist").glob("*.whl"))


@pytest.mark.parametrize("notice", ["vendor/jxl-rs/LICENSE", "vendor/jxl-rs/PATENTS",
                                     "vendor/jxl-rs/AUTHORS", "vendor/jxl-rs/jxl/LICENSE",
                                     "vendor/jxl-rs/jxl_macros/LICENSE", "vendor/jxl-rs/jxl_simd/LICENSE",
                                     "vendor/jxl-rs/jxl_transforms/LICENSE", "vendor/libsais-rs/LICENSE",
                                     "docs/PORTING_KERNELS.md"])
@pytest.mark.parametrize("omission", ["file", "configuration"])
def test_every_wheel_notice_is_mandatory(packaging, monkeypatch, tmp_path, notice, omission):
    root = fixture_project(tmp_path, packaging)
    monkeypatch.setattr(packaging.commands, "ROOT", root)
    command = packaging.commands.bdist_wheel(packaging.commands.BinaryDistribution())
    command.distribution.metadata.license_files = list(packaging.sources.LICENSE_FILES)
    if omission == "file":
        (root / notice).unlink()
    else:
        command.distribution.metadata.license_files.remove(notice)
    with pytest.raises(RuntimeError, match="Incomplete wheel notices") as error:
        command.egg2dist(str(root / "unused.egg-info"), str(root / "unused.dist-info"))
    assert notice in str(error.value)


@pytest.mark.parametrize("operation", ["build_wheel", "build_editable"])
def test_prebuilt_wheel_cannot_omit_notices(packaging, tmp_path, operation):
    root = fixture_project(tmp_path, packaging)
    native = write(tmp_path / "prebuilt" / packaging.native._native_name(), native_header(sys.platform))
    (root / "docs/PORTING_KERNELS.md").unlink()
    run_backend(root, operation, native, error="Incomplete wheel notices")
    assert not list((root / "dist").glob("*.whl"))


@pytest.mark.parametrize("asset", EMBEDDED_ASSETS)
def test_sdist_cannot_omit_embedded_asset(packaging, monkeypatch, tmp_path, asset):
    root = fixture_project(tmp_path, packaging)
    (root / asset).unlink()
    monkeypatch.setattr(packaging.commands, "ROOT", root)
    command = packaging.commands.sdist(packaging.commands.BinaryDistribution())
    with pytest.raises(RuntimeError, match="Incomplete source packaging") as error:
        command.run()
    assert asset in str(error.value)


@pytest.mark.parametrize("support", ["__init__.py", "native.py", "sources.py", "commands.py"])
def test_missing_build_support_prevents_cargo(packaging, monkeypatch, tmp_path, support):
    root = fixture_project(tmp_path, packaging)
    missing = "build_support/" + support
    (root / missing).unlink()
    monkeypatch.setattr(packaging.native.subprocess, "run", forbid_subprocess)
    with pytest.raises(RuntimeError, match="Incomplete source packaging") as error:
        packaging.native._native_library(root)
    assert missing in str(error.value)


@pytest.mark.parametrize("support", ["__init__.py", "native.py", "sources.py", "commands.py"])
@pytest.mark.parametrize("omission", ["file", "manifest"])
def test_sdist_cannot_omit_build_support(packaging, monkeypatch, tmp_path, support, omission):
    root = fixture_project(tmp_path, packaging)
    missing = "build_support/" + support
    monkeypatch.setattr(packaging.commands, "ROOT", root)
    command = packaging.commands.sdist(packaging.commands.BinaryDistribution())
    if omission == "file":
        (root / missing).unlink()
        with pytest.raises(RuntimeError, match="Incomplete source packaging") as error:
            command.run()
    else:
        command.filelist = SimpleNamespace(
            files=[name for name in packaging.sources.REQUIRED_SOURCES if name != missing])
        with pytest.raises(RuntimeError, match="Incomplete sdist manifest") as error:
            command.make_distribution()
    assert missing in str(error.value)


def test_backend_rejects_sdist_without_build_support(packaging, tmp_path):
    root = fixture_project(tmp_path, packaging)
    manifest = root / "MANIFEST.in"
    write(manifest, manifest.read_text(encoding="utf-8") + "\nrecursive-exclude build_support *.py\n")
    run_backend(root, "build_sdist", error="Incomplete sdist manifest; missing:")
    assert not list((root / "dist").glob("*.tar.gz"))


def test_setup_import_from_unrelated_directory(packaging, tmp_path):
    root = fixture_project(tmp_path, packaging)
    caller = tmp_path / "unrelated caller"
    caller.mkdir()
    code = (
        "from setuptools import setup\n"
        "import importlib.util\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n"
        "def forbidden(*a, **kw):\n"
        "    raise AssertionError('loading setup must not invoke Cargo')\n"
        "subprocess.run = forbidden\n"
        "root = Path(sys.argv[1]).resolve()\n"
        "spec = importlib.util.spec_from_file_location('rs_jxl_packaging', root / 'setup.py')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "from build_support import commands\n"
        "assert commands.ROOT == root\n"
        "assert module.build_py is commands.build_py\n"
        "assert module.bdist_wheel is commands.bdist_wheel\n"
        "assert module.editable_wheel is commands.editable_wheel\n"
        "assert module.sdist is commands.sdist\n"
        "assert module.BinaryDistribution is commands.BinaryDistribution\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code, str(root)], cwd=caller,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
