"""Build a platform wheel containing the Rust C ABI, without a Python ABI tag."""

import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
from setuptools.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


ROOT = Path(__file__).resolve().parent
NATIVE_NAMES = {
    "win32": "rs_jxl_ffi.dll",
    "linux": "librs_jxl_ffi.so",
    "darwin": "librs_jxl_ffi.dylib",
}
LICENSE_FILES = (
    "vendor/jxl-rs/LICENSE", "vendor/jxl-rs/PATENTS", "vendor/jxl-rs/AUTHORS",
    "vendor/libsais-rs/LICENSE", "docs/PORTING_KERNELS.md",
) + tuple(
    f"vendor/jxl-rs/{crate}/LICENSE"
    for crate in ("jxl", "jxl_macros", "jxl_simd", "jxl_transforms")
)
REQUIRED_SOURCES = (
    "pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "AGENTS.md",
    "docs/PYTHON_API.md", "docs/PYTHON_INTERFACE.md", "docs/ARCHITECTURE.md", "docs/README.md",
    "Cargo.toml", "Cargo.lock", "python/rs_jxl/__init__.py", "python/rs_jxl/_ffi.py",
    "crates/rs-jxl/Cargo.toml", "crates/rs-jxl/src/lib.rs", "crates/rs-jxl/tests/srgb_profile.hex",
    "crates/rs-jxl-cli/Cargo.toml", "crates/rs-jxl-cli/src/main.rs",
    "crates/rs-jxl-image/Cargo.toml", "crates/rs-jxl-image/src/lib.rs",
    "crates/rs-jxl-ffi/Cargo.toml", "crates/rs-jxl-ffi/src/lib.rs",
    "crates/rs-jxl-ffi/include/rs_jxl.h",
    "vendor/jxl-rs/Cargo.toml", "vendor/jxl-rs/README.md",
    "vendor/jxl-rs/PROVENANCE.md",
    "vendor/jxl-rs/jxl/src/util/dither_32x32.bin",
    "vendor/libsais-rs/Cargo.toml", "vendor/libsais-rs/src/lib.rs",
    "vendor/libsais-rs/README.md",
    "vendor/libsais-rs/PROVENANCE.md",
) + tuple(
    f"vendor/jxl-rs/{crate}/{name}"
    for crate in ("jxl", "jxl_macros", "jxl_simd", "jxl_transforms")
    for name in ("Cargo.toml", "src/lib.rs")
) + LICENSE_FILES


def _native_name():
    try:
        return NATIVE_NAMES[sys.platform]
    except KeyError:
        raise RuntimeError(f"Unsupported native platform: {sys.platform}") from None


def _host_arch():
    if sys.platform == "win32":
        machine = {"win-amd64": "x86_64", "win32": "i686",
                   "win-arm64": "aarch64"}.get(sysconfig.get_platform(), "unsupported")
    else:
        machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64",
               "x86": "i686", "i386": "i686", "i586": "i686"}.get(machine, machine)
    if machine == "x86_64" and struct.calcsize("P") == 4:
        machine = "i686"
    bits = {"x86_64": 64, "i686": 32, "aarch64": 64}.get(machine)
    if bits is None or bits != struct.calcsize("P") * 8:
        raise RuntimeError(f"Unsupported Python architecture: {machine}")
    return machine, bits


def _build_target():
    # Cross tags and universal binaries require a separate, audited build flow.
    for name in ("_PYTHON_HOST_PLATFORM", "ARCHFLAGS"):
        if os.environ.get(name):
            raise RuntimeError(f"Cross-build override {name} is not supported")
    _native_name()
    arch, _ = _host_arch()
    target = os.environ.get("CARGO_BUILD_TARGET")
    if not target:
        return None
    if sys.platform == "win32":
        suffixes = ("pc-windows-msvc", "pc-windows-gnu")
    elif sys.platform == "darwin":
        suffixes = ("apple-darwin",)
    else:
        libc = platform.libc_ver()[0]
        suffixes = {"glibc": ("unknown-linux-gnu",),
                    "musl": ("unknown-linux-musl",)}.get(libc, ())
    if target not in {f"{arch}-{suffix}" for suffix in suffixes}:
        raise RuntimeError(
            f"Unsupported cross-build CARGO_BUILD_TARGET={target!r}; "
            "build with a Python interpreter native to the target platform"
        )
    return target


def _validate_native(path):
    """Inspect headers only. Never load or execute an untrusted prebuilt library."""
    expected = _native_name()
    if path.name != expected:
        raise RuntimeError(f"Native library must be named {expected}, got {path.name}")
    if not path.is_file():
        raise RuntimeError(f"Native library is missing: {path}")
    arch, bits = _host_arch()
    valid = False
    with path.open("rb") as stream:
        header = stream.read(64)
        if sys.platform == "win32" and len(header) == 64 and header[:2] == b"MZ":
            offset = struct.unpack_from("<I", header, 0x3C)[0]
            if 64 <= offset <= path.stat().st_size - 26:
                stream.seek(offset)
                pe = stream.read(26)
                machine = {"i686": 0x14C, "x86_64": 0x8664, "aarch64": 0xAA64}[arch]
                optional_size, flags = struct.unpack_from("<HH", pe, 20)
                valid = (
                    pe[:4] == b"PE\0\0"
                    and struct.unpack_from("<H", pe, 4)[0] == machine
                    and flags & 0x2000 != 0  # IMAGE_FILE_DLL
                    and optional_size >= (112 if bits == 64 else 96)
                    and offset + 24 + optional_size <= path.stat().st_size
                    and struct.unpack_from("<H", pe, 24)[0] == (0x20B if bits == 64 else 0x10B)
                )
        elif sys.platform == "linux" and len(header) >= (64 if bits == 64 else 52):
            endian = "<" if sys.byteorder == "little" else ">"
            machine = {"i686": 3, "x86_64": 62, "aarch64": 183}[arch]
            valid = (
                header[:4] == b"\x7fELF"
                and header[4] == (2 if bits == 64 else 1)
                and header[5] == (1 if sys.byteorder == "little" else 2)
                and header[6] == 1
                and header[7] in (0, 3)  # System V or GNU/Linux
                and struct.unpack_from(endian + "HHI", header, 16) == (3, machine, 1)
            )
        elif sys.platform == "darwin" and len(header) >= 32:
            machine = {"i686": 7, "x86_64": 0x1000007, "aarch64": 0x100000C}[arch]
            valid = (
                header[:4] == (b"\xcf\xfa\xed\xfe" if bits == 64 else b"\xce\xfa\xed\xfe")
                and struct.unpack_from("<I", header, 4)[0] == machine
                and struct.unpack_from("<I", header, 12)[0] == 6  # MH_DYLIB, not a bundle/executable
            )
    if not valid:
        raise RuntimeError(
            f"Invalid native format/architecture for {sys.platform}/{arch}: {path}; "
            "expected a matching PE DLL, ELF shared library, or thin Mach-O dylib"
        )
    return path


def _require_sources(root):
    missing = [name for name in REQUIRED_SOURCES if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Incomplete source packaging; missing:\n  " + "\n  ".join(missing))


def _native_library(root):
    target = _build_target()
    explicit = os.environ.get("RS_JXL_NATIVE_LIB")
    if explicit is not None:
        if not explicit.strip():
            raise RuntimeError("RS_JXL_NATIVE_LIB must name a prebuilt native library")
        candidate = Path(explicit).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return _validate_native(candidate.resolve())

    _require_sources(root)
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR") or "target").expanduser()
    if not target_dir.is_absolute():
        target_dir = root / target_dir
    target_dir = target_dir.resolve()
    output_dir = target_dir / target if target else target_dir
    candidate = output_dir / "release" / _native_name()
    command = ["cargo", "build", "--locked", "--offline", "--release", "-p", "rs-jxl-ffi",
               "--target-dir", str(target_dir), "--message-format=json-render-diagnostics"]
    if target:
        command.extend(["--target", target])
    try:
        result = subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE,
                                text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Locked offline Cargo build failed. Install the Rust toolchain and cache "
            "the locked dependencies, or set RS_JXL_NATIVE_LIB to a validated prebuilt library."
        ) from error
    # Cargo's artifact report prevents a stale host library from hiding a config-file
    # target override. Only the deterministic release path above is ever accepted.
    reported = set()
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(message, dict) and message.get("reason") == "compiler-artifact"
                and message.get("target", {}).get("name") == "rs_jxl_ffi"
                and "cdylib" in message.get("target", {}).get("crate_types", [])):
            reported.update(Path(name).resolve() for name in message.get("filenames", []))
    if candidate.resolve() not in reported:
        raise RuntimeError(
            f"Cargo did not report the required native artifact: {candidate}. "
            "Cargo config cross-builds are unsupported; set a native CARGO_BUILD_TARGET "
            "and CARGO_TARGET_DIR explicitly. No stale library will be bundled."
        )
    return _validate_native(candidate)


class build_py(_build_py):
    def run(self):
        if not (ROOT / "python/rs_jxl/__init__.py").is_file():
            raise RuntimeError("Incomplete Python package: python/rs_jxl/__init__.py is missing")
        candidate = _native_library(ROOT)
        super().run()
        if self.editable_mode:
            # The source loader uses target[/<triple>]/release or RS_JXL_NATIVE_LIB.
            return
        package_dir = Path(self.build_lib) / "rs_jxl"
        package_dir.mkdir(parents=True, exist_ok=True)
        for name in NATIVE_NAMES.values():
            if name != candidate.name and (package_dir / name).exists():
                raise RuntimeError(f"Stale foreign native library in build directory: {package_dir / name}")
        # Always replace the build output, even if the prebuilt file has an older mtime.
        shutil.copy2(candidate, package_dir / candidate.name)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        if not self.editable_mode:
            outputs.append(str(Path(self.build_lib) / "rs_jxl" / _native_name()))
        return outputs


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class editable_wheel(_editable_wheel):
    def _safely_run(self, cmd_name):
        # Setuptools 68 otherwise downgrades custom build_py errors to warnings.
        return self.run_command(cmd_name)


class bdist_wheel(_bdist_wheel):
    def egg2dist(self, egginfo_path, distinfo_path):
        notices = tuple(Path(name) for name in self.distribution.metadata.license_files or ())
        configured = {name.as_posix() for name in notices}
        missing = [name for name in LICENSE_FILES
                   if name not in configured or not (ROOT / name).is_file()]
        if missing:
            raise RuntimeError("Incomplete wheel notices; missing or unconfigured:\n  " + "\n  ".join(missing))
        if any(name.is_absolute() or ".." in name.parts for name in notices):
            raise RuntimeError("Wheel license-files must be paths within the source tree")
        super().egg2dist(egginfo_path, distinfo_path)
        # Older wheel releases flatten LICENSE basenames and overwrite distinct
        # notices. Preserve their paths (including module ../LICENSE references).
        info = Path(distinfo_path)
        for name in notices:
            destination = info / "licenses" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            flattened = info / name.name
            if flattened.is_file():
                flattened.unlink()

    def finalize_options(self):
        _build_target()
        requested = self.plat_name
        host_tag = sysconfig.get_platform().lower().replace("-", "_").replace(".", "_")
        if requested and requested.lower().replace("-", "_").replace(".", "_") != host_tag:
            raise RuntimeError("Cross-build --plat-name overrides are not supported")
        super().finalize_options()
        self.root_is_pure = False

    def run(self):
        if self.skip_build:
            raise RuntimeError("--skip-build is unsupported: native validation must not be skipped")
        super().run()

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        arch, _ = _host_arch()
        if sys.platform == "win32":
            suffix = {"x86_64": "amd64", "i686": "win32", "aarch64": "arm64"}[arch]
        elif sys.platform == "darwin":
            suffix = {"aarch64": "arm64"}.get(arch, arch)
        else:
            suffix = arch
        if not platform_tag.endswith(suffix):
            raise RuntimeError(f"Unsupported cross/universal wheel platform: {platform_tag}")
        return "py3", "none", platform_tag


class sdist(_sdist):
    def run(self):
        _require_sources(ROOT)
        super().run()

    def make_distribution(self):
        included = {Path(name).as_posix() for name in self.filelist.files}
        missing = sorted(set(REQUIRED_SOURCES) - included)
        if missing:
            raise RuntimeError("Incomplete sdist manifest; missing:\n  " + "\n  ".join(missing))
        super().make_distribution()


if __name__ == "__main__":
    setup(cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel,
                   "editable_wheel": editable_wheel, "sdist": sdist},
          distclass=BinaryDistribution)
