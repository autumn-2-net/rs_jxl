"""Setuptools hooks for native wheels, editable installs and complete sdists."""

from pathlib import Path
import shutil
import sys
import sysconfig

from setuptools import Distribution
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
from setuptools.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from . import native, sources


ROOT = Path(__file__).resolve().parents[1]


class build_py(_build_py):
    def run(self):
        if not (ROOT / "python/rs_jxl/__init__.py").is_file():
            raise RuntimeError("Incomplete Python package: python/rs_jxl/__init__.py is missing")
        candidate = native._native_library(ROOT)
        super().run()
        if self.editable_mode:
            # The source loader uses target[/<triple>]/release or RS_JXL_NATIVE_LIB.
            return
        package_dir = Path(self.build_lib) / "rs_jxl"
        package_dir.mkdir(parents=True, exist_ok=True)
        for name in native.NATIVE_NAMES.values():
            if name != candidate.name and (package_dir / name).exists():
                raise RuntimeError(f"Stale foreign native library in build directory: {package_dir / name}")
        # Always replace the build output, even if the prebuilt file has an older mtime.
        shutil.copy2(candidate, package_dir / candidate.name)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        if not self.editable_mode:
            outputs.append(str(Path(self.build_lib) / "rs_jxl" / native._native_name()))
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
        missing = [name for name in sources.LICENSE_FILES
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
        native._build_target()
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
        arch, _ = native._host_arch()
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
        sources._require_sources(ROOT)
        super().run()

    def make_distribution(self):
        included = {Path(name).as_posix() for name in self.filelist.files}
        missing = sorted(set(sources.REQUIRED_SOURCES) - included)
        if missing:
            raise RuntimeError("Incomplete sdist manifest; missing:\n  " + "\n  ".join(missing))
        super().make_distribution()
