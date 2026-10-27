"""Package a built CLI with notices; never gather arbitrary build-tree files."""

import argparse
import hashlib
import os
from pathlib import Path
import tarfile
import zipfile

from prepare_ci_version import release_version


ROOT = Path(__file__).resolve().parents[1]
NOTICES = (
    "README.md",
    "docs/CI.md",
    "docs/ENCODING_PARAMETERS.md",
    "docs/PORTING_KERNELS.md",
    "vendor/jxl-rs/LICENSE",
    "vendor/jxl-rs/PATENTS",
    "vendor/jxl-rs/AUTHORS",
    "vendor/jxl-rs/PROVENANCE.md",
    "vendor/libsais-rs/LICENSE",
    "vendor/libsais-rs/PROVENANCE.md",
) + tuple(
    f"vendor/jxl-rs/{crate}/LICENSE"
    for crate in ("jxl", "jxl_macros", "jxl_simd", "jxl_transforms")
)


def archive_cli(root, cli, destination, platform, version=None):
    root, cli, destination = Path(root), Path(cli), Path(destination)
    if platform not in ("windows-x86_64", "linux-x86_64"):
        raise ValueError(f"Unsupported CI platform: {platform}")
    if version is not None:
        version = release_version(version)
    binary_name = "rs-jxl.exe" if platform == "windows-x86_64" else "rs-jxl"
    files = [(cli, binary_name)] + [(root / name, name) for name in NOTICES]
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = f"rs-jxl-{version}-{platform}" if version is not None else f"rs-jxl-{platform}"
    if platform == "windows-x86_64":
        result = destination / f"{prefix}.zip"
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, name in files:
                archive.write(path, f"{prefix}/{name}")
    else:
        result = destination / f"{prefix}.tar.gz"
        with tarfile.open(result, "w:gz") as archive:
            for path, name in files:
                info = archive.gettarinfo(str(path), f"{prefix}/{name}")
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if name == binary_name else 0o644
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    return result


def write_checksums(destination, files):
    lines = []
    for path in sorted(files):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.name}\n")
    result = Path(destination) / "SHA256SUMS"
    result.write_text("".join(lines), encoding="ascii")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows-x86_64", "linux-x86_64"), required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--version", default=os.environ.get("RS_JXL_RELEASE_VERSION"))
    args = parser.parse_args()
    wheels = sorted(args.dist.glob("rs_jxl-*.whl"))
    if len(wheels) != 1:
        parser.error(f"Expected exactly one rs_jxl wheel in {args.dist}, found {len(wheels)}")
    archive = archive_cli(ROOT, args.cli, args.dist, args.platform, args.version)
    checksums = write_checksums(args.dist, [archive, *wheels])
    for path in [*wheels, archive, checksums]:
        print(f"Artifact: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
