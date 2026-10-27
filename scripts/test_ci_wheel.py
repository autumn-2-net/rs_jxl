"""Install one built wheel in a fresh venv and test without source import paths."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import venv


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    args = parser.parse_args()
    wheels = sorted(args.wheel_dir.resolve().glob("rs_jxl-*.whl"))
    if len(wheels) != 1:
        parser.error(f"Expected exactly one rs_jxl wheel, found {len(wheels)}")
    cli = args.cli.resolve(strict=True)
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "RS_JXL_NATIVE_LIB", "RS_JXL_DJXL", "RS_JXL_TEST_CUDA"):
        env.pop(name, None)
    # A fresh environment must exercise the wheel's own library, not target/release.
    with tempfile.TemporaryDirectory(prefix="rs-jxl-wheel-ci-") as directory:
        work = Path(directory)
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        def run(*command):
            subprocess.run([str(python), "-I", "-B", *map(str, command)],
                           cwd=work, env=env, check=True)

        run("-m", "pip", "install", "--disable-pip-version-check", str(wheels[0]),
            "pillow", "pytest", "packaging", "setuptools>=68", "wheel>=0.38")
        run(ROOT / "scripts/smoke_ci.py", "--cli", cli)
        run("-m", "pytest", "--import-mode=prepend", "-p", "no:cacheprovider",
            ROOT / "python/tests", "-q")
    print("Installed-wheel tests passed.", flush=True)


if __name__ == "__main__":
    main()
