"""Set a release version only in the disposable CI checkout, without resolving dependencies."""

import ast
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def release_version(value):
    value = value.strip()
    if value.startswith("v"):
        value = value[1:]
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value):
        raise ValueError("Version must be X.Y.Z or vX.Y.Z with no leading zeroes")
    if any(int(part) > 2**64 - 1 for part in value.split(".")):
        raise ValueError("Version components must fit Cargo's unsigned 64-bit integers")
    return value


def prepare(root, requested):
    import tomlkit

    root = Path(root)
    cargo_path = root / "Cargo.toml"
    python_path = root / "pyproject.toml"
    lock_path = root / "Cargo.lock"
    init_path = root / "python/rs_jxl/__init__.py"
    cargo = tomlkit.parse(cargo_path.read_text(encoding="utf-8"))
    project = tomlkit.parse(python_path.read_text(encoding="utf-8"))
    lock = tomlkit.parse(lock_path.read_text(encoding="utf-8"))
    current = str(cargo["workspace"]["package"]["version"])
    version = release_version(requested if requested.strip() else current)

    source = init_path.read_text(encoding="utf-8")
    assignments = [node for node in ast.parse(source).body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "__version__"]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant):
        raise ValueError("Expected one literal __version__ assignment")
    assignment = assignments[0]
    if assignment.lineno != assignment.end_lineno:
        raise ValueError("Expected a single-line __version__ assignment")
    if project["project"]["version"] != current or assignment.value.value != current:
        raise ValueError("Cargo, Python package and wheel source versions disagree")

    members = set()
    for member in cargo["workspace"]["members"]:
        manifest = tomlkit.parse((root / member / "Cargo.toml").read_text(encoding="utf-8"))
        if manifest["package"]["version"] != {"workspace": True}:
            raise ValueError(f"Workspace member does not inherit its version: {member}")
        members.add(str(manifest["package"]["name"]))
    packages = [package for package in lock["package"]
                if package["name"] in members and "source" not in package]
    if {package["name"] for package in packages} != members or len(packages) != len(members):
        raise ValueError("Lockfile does not contain exactly the workspace packages")
    if any(package["version"] != current for package in packages):
        raise ValueError("Workspace package versions disagree with the lockfile")

    if version == current:
        return version
    cargo["workspace"]["package"]["version"] = version
    project["project"]["version"] = version
    for package in packages:
        package["version"] = version
    # Cargo may qualify a dependency when another source has the same crate name.
    for package in lock["package"]:
        dependencies = package.get("dependencies", [])
        for index, dependency in enumerate(dependencies):
            if dependency in {f"{name} {current}" for name in members}:
                dependencies[index] = f"{dependency.split()[0]} {version}"
    lines = source.splitlines(keepends=True)
    lines[assignment.lineno - 1] = f'__version__ = "{version}"\n'
    # All validation precedes writes. Registry versions and checksums are untouched.
    for path, document in ((cargo_path, cargo), (python_path, project), (lock_path, lock)):
        path.write_text(tomlkit.dumps(document), encoding="utf-8")
    init_path.write_text("".join(lines), encoding="utf-8")
    return version


def main():
    requested = os.environ.get("RS_JXL_BUILD_VERSION", "")
    version = prepare(ROOT, requested)
    for variable, key in (("GITHUB_OUTPUT", "version"), ("GITHUB_ENV", "RS_JXL_RELEASE_VERSION")):
        if os.environ.get(variable):
            with Path(os.environ[variable]).open("a", encoding="utf-8") as output:
                output.write(f"{key}={version}\n")
                if variable == "GITHUB_OUTPUT":
                    output.write(f"publish_release={str(bool(requested.strip())).lower()}\n")
    print(f"Build version: {version}")


if __name__ == "__main__":
    main()
