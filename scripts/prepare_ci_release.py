"""Validate downloaded CI artifacts and assemble a new, flat release directory."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import stat

from package_ci import write_checksums
from prepare_ci_version import release_version


PLATFORMS = {"windows-x86_64": ".zip", "linux-x86_64": ".tar.gz"}


def checked_path(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError(f"Path traversal is not allowed: {path}")
    path = path.absolute()
    # Inspect ancestors without resolving links, including Windows junctions.
    for candidate in reversed((path, *path.parents)):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f"Linked paths are not allowed: {candidate}")
        if candidate != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Expected a directory: {candidate}")
    return path


def matching_wheel(name, version, platform):
    prefix = f"rs_jxl-{version}-py3-none-"
    if platform == "windows-x86_64":
        return name == f"{prefix}win_amd64.whl"
    tag = r"manylinux(?:_[0-9]+_[0-9]+|1|2010|2014)_x86_64"
    match = re.fullmatch(re.escape(prefix) + rf"({tag}(?:\.{tag})*)\.whl", name)
    return match is not None and "manylinux_2_28_x86_64" in match[1].split(".")


def validate_artifact(directory, version, platform):
    checked_path(directory)
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError(f"Expected an artifact directory: {directory}")
    files = {path.name: path for path in directory.iterdir()}
    wheels = [name for name in files if matching_wheel(name, version, platform)]
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one matching wheel in {directory}")
    archive = f"rs-jxl-{version}-{platform}{PLATFORMS[platform]}"
    data_names = {wheels[0], archive}
    if set(files) != data_names | {"SHA256SUMS"}:
        raise ValueError(f"Missing or unexpected files in {directory}")
    for path in files.values():
        checked_path(path)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Expected a regular artifact file: {path}")

    checksums = {}
    for line in files["SHA256SUMS"].read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed checksum record in {directory}")
        digest, name = match.groups()
        # Only known basenames are allowed; never open a path from a record.
        if name not in data_names or name in checksums:
            raise ValueError(f"Unexpected or duplicate checksum record: {name}")
        checksums[name] = digest.lower()
    if set(checksums) != data_names:
        raise ValueError(f"Missing checksum records in {directory}")
    for name, expected in checksums.items():
        digest = hashlib.sha256()
        with files[name].open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {files[name]}")
    return [files[name] for name in sorted(data_names)]


def prepare(artifacts, output, version):
    version = release_version(version)
    artifacts, output = checked_path(artifacts), checked_path(output)
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    if artifacts in output.parents:
        raise ValueError("Output must be outside the input artifact directory")
    expected = {f"rs-jxl-{version}-{platform}" for platform in PLATFORMS}
    if {path.name for path in artifacts.iterdir()} != expected:
        raise ValueError("Missing or unexpected platform artifact directories")
    sources = []
    for platform in PLATFORMS:
        sources.extend(validate_artifact(artifacts / f"rs-jxl-{version}-{platform}",
                                         version, platform))
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("Release asset filenames collide")

    # Validate both platforms before creating any output; never reuse a directory.
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for source in sources:
        destination = output / source.name
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        copied.append(destination)
    return [*copied, write_checksums(output, copied)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    for path in prepare(args.artifacts, args.output, args.version):
        print(f"Release asset: {path}")


if __name__ == "__main__":
    main()
