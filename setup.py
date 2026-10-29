"""Build a platform wheel containing the Rust C ABI, without a Python ABI tag."""

from pathlib import Path
import sys

from setuptools import setup

# PEP 517 frontends need not put the source tree on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_support.commands import BinaryDistribution, bdist_wheel, build_py, editable_wheel, sdist


if __name__ == "__main__":
    setup(cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel,
                   "editable_wheel": editable_wheel, "sdist": sdist},
          distclass=BinaryDistribution)
