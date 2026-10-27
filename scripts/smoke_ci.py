"""Smoke-test an installed wheel and CLI using only small generated fixtures."""

import argparse
from importlib.metadata import version as package_version
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def installed_codec():
    require(sys.flags.isolated, "Run this script with python -I")
    require("RS_JXL_NATIVE_LIB" not in os.environ,
            "Unset RS_JXL_NATIVE_LIB: this test must load the wheel's bundled library")

    import rs_jxl
    from rs_jxl import _ffi

    prefix = Path(sys.prefix).resolve(strict=True)
    sites = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
    sites = {path for path in sites if path.is_relative_to(prefix)}
    package = Path(rs_jxl.__file__).resolve(strict=True).parent
    require(any(package.is_relative_to(path) for path in sites),
            f"rs_jxl is not installed in the active Python prefix {prefix}: {package}")
    for name, module in tuple(sys.modules.items()):
        if name == "rs_jxl" or name.startswith("rs_jxl."):
            require(getattr(module, "__file__", None) is not None,
                    f"Package module has no file: {name}")
            require(Path(module.__file__).resolve(strict=True).is_relative_to(package),
                    f"Package module is outside the installed package: {name}: {module.__file__}")
    native = _ffi._library_path().resolve(strict=True)
    require(native.parent == package and native.name == _ffi.native_name(),
            f"Native library is not bundled in the installed package: {native}")
    require(_ffi._native is None, "Native library was loaded before installation checks")
    _ffi.library()  # Load the verified bundled path and validate the C ABI.
    require(rs_jxl.__version__ == package_version("rs-jxl"),
            "Python __version__ differs from wheel metadata")
    expected = os.environ.get("RS_JXL_RELEASE_VERSION", rs_jxl.__version__)
    require(rs_jxl.__version__ == expected, "Installed wheel has the wrong build version")
    print(f"PASS installed package: {package}", flush=True)
    print(f"PASS bundled native ABI: {native}", flush=True)
    return rs_jxl


def exact_pixels(actual, expected, label):
    require(isinstance(actual, np.ndarray), f"{label}: output is not a NumPy array")
    require(actual.shape == expected.shape,
            f"{label}: shape {actual.shape} != {expected.shape}")
    require(actual.dtype == expected.dtype,
            f"{label}: dtype {actual.dtype} != {expected.dtype}")
    require(np.array_equal(actual, expected), f"{label}: pixel samples differ")


def run_cli(cli, directory, *arguments):
    command = [str(cli), *(str(value) for value in arguments)]
    print(f"RUN CLI: {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=directory, stdin=subprocess.DEVNULL,
                   check=True, timeout=120)


def smoke(codec, cli, directory):
    samples = np.arange(5 * 7 * 4, dtype=np.uint32).reshape(5, 7, 4)
    rgba8 = ((samples * 37 + 11) % 256).astype(np.uint8)
    rgba16 = ((samples * 997 + 123) % 65536).astype(np.uint16)
    for source in (rgba8, rgba16):
        source[::2, ::2, 3] = 0
        source[1, 1, 3] = np.iinfo(source.dtype).max
        require(np.any(source[..., :3][source[..., 3] == 0] != 0),
                "Fixture must contain nonzero RGB under zero alpha")
        encoded = codec.encode(source, effort=3, threads=1)
        exact_pixels(codec.decode(encoded, threads=1), source, f"Python {source.dtype}")
        print(f"PASS Python {source.dtype} RGBA, including hidden RGB", flush=True)

    with codec.Encoder(threads=1) as encoder, codec.Decoder(threads=1) as decoder:
        encoded = encoder.encode(rgba8, effort=3)
        exact_pixels(decoder.decode(encoded), rgba8, "Python contexts")
    print("PASS Python encoder/decoder contexts", flush=True)

    frames = [rgba8.copy() for _ in range(3)]
    frames[1][0, 0, 0] ^= 1  # A hidden-RGB-only change must survive planning.
    frames[2][1, 2, 1] ^= 7
    frames.append(frames[-1].copy())
    durations = [1, 2, 3, 4]
    encoded = codec.encode_multiframe(frames, durations=durations, strategy="auto",
                                     reference="auto", effort=3, threads=1)
    decoded, metadata = codec.decode(encoded, frame=None, return_info=True, threads=1)
    require(isinstance(decoded, list) and len(decoded) == len(frames),
            "Python automatic multiframe: wrong displayed frame count")
    require([frame["duration"] for frame in metadata["frames"]] == durations,
            "Python automatic multiframe: durations differ")
    for index, (actual, expected) in enumerate(zip(decoded, frames)):
        exact_pixels(actual, expected, f"Python automatic multiframe {index}")
    print("PASS Python automatic multiframe: exact frames and durations", flush=True)

    jpeg = directory / "input.jpg"
    Image.fromarray(rgba8[..., :3]).save(jpeg, format="JPEG", quality=90,
                                       subsampling=0, progressive=False, optimize=False)
    original_jpeg = jpeg.read_bytes()
    python_jxl = directory / "python-jpeg.jxl"
    codec.transcode_jpeg(jpeg, python_jxl, effort=3, threads=1)
    require(codec.reconstruct_jpeg(python_jxl, threads=1) == original_jpeg,
            "Python JPEG reconstruction differs from the original bytes")
    print("PASS Python JPEG byte-exact reconstruction", flush=True)

    png = directory / "input.png"
    png_jxl = directory / "cli-png.jxl"
    restored_png = directory / "restored.png"
    Image.fromarray(rgba8).save(png, format="PNG")
    run_cli(cli, directory, "encode", png, png_jxl, "--source-effort", "3", "--threads", "1")
    exact_pixels(codec.decode(png_jxl, threads=1), rgba8, "CLI PNG to Python")
    run_cli(cli, directory, "decode", png_jxl, restored_png, "--threads", "1")
    with Image.open(restored_png) as image:
        exact_pixels(np.array(image), rgba8, "CLI PNG round trip")
    print("PASS CLI PNG lossless round trip, including hidden RGB", flush=True)

    jpeg_jxl = directory / "cli-jpeg.jxl"
    restored_jpeg = directory / "restored.jpg"
    run_cli(cli, directory, "transcode-jpeg", jpeg, jpeg_jxl, "--effort", "3", "--threads", "1")
    require(codec.reconstruct_jpeg(jpeg_jxl, threads=1) == original_jpeg,
            "CLI JPEG transcode to Python reconstruction differs")
    run_cli(cli, directory, "decode", jpeg_jxl, restored_jpeg,
            "--reconstruct-jpeg", "--threads", "1")
    require(restored_jpeg.read_bytes() == original_jpeg,
            "CLI JPEG reconstruction differs from the original bytes")
    restored_python_jpeg = directory / "restored-python.jpg"
    run_cli(cli, directory, "decode", python_jxl, restored_python_jpeg,
            "--reconstruct-jpeg", "--threads", "1")
    require(restored_python_jpeg.read_bytes() == original_jpeg,
            "Python JPEG transcode to CLI reconstruction differs")
    print("PASS CLI JPEG byte-exact reconstruction and Python interoperability", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path, help="Absolute path to the built rs-jxl CLI")
    args = parser.parse_args()
    require(args.cli.is_absolute(), "--cli must be an absolute path")
    cli = args.cli.resolve(strict=True)
    require(cli.is_file(), f"CLI is not a file: {cli}")
    codec = installed_codec()
    reported = subprocess.run([str(cli), "--version"], check=True, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    require(reported == f"rs-jxl {codec.__version__}",
            f"CLI and wheel versions disagree: {reported!r}, {codec.__version__!r}")
    with tempfile.TemporaryDirectory(prefix="rs-jxl-ci-") as directory:
        smoke(codec, cli, Path(directory))
    print("PASS all installed-wheel and CLI smoke checks", flush=True)


if __name__ == "__main__":
    main()
