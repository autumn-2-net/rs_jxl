"""Source-tree completeness and third-party notice manifests."""

BUILD_SUPPORT_FILES = (
    "build_support/__init__.py", "build_support/native.py",
    "build_support/sources.py", "build_support/commands.py",
)
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
) + LICENSE_FILES + BUILD_SUPPORT_FILES


def _require_sources(root):
    missing = [name for name in REQUIRED_SOURCES if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Incomplete source packaging; missing:\n  " + "\n  ".join(missing))
