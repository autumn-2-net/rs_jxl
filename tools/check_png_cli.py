"""Opt-in CLI acceptance on real 8-bit PNGs; Pillow is a test-only dependency.

Times include process startup, PNG decoding, encoding, and stdout collection.
The independent decoder is only an oracle, never an encoder runtime dependency.
16-bit samples are covered by the Rust reference suite without Pillow narrowing.
"""

import argparse
from io import BytesIO
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from time import perf_counter

from PIL import Image


def run(command, *, timeout=120, show_stderr=False, **kwargs):
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as error:
        if error.stderr:
            sys.stderr.write(error.stderr.decode(errors="replace"))
        raise
    if result.returncode:
        raise RuntimeError(f"{command!r}\n{result.stderr.decode(errors='replace')}")
    if show_stderr and result.stderr:
        sys.stderr.write(result.stderr.decode(errors="replace"))
    return result.stdout


def png_samples(data):
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Missing PNG IHDR")
    length, kind, _, _, depth, _, _, _, _ = struct.unpack_from(
        ">I4sIIBBBBB", data, 8
    )
    if length != 13 or kind != b"IHDR" or depth > 8:
        raise ValueError("This helper accepts only 1/2/4/8-bit PNG; use Rust tests for 16-bit")
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
            raise ValueError("This helper accepts only 1/2/4/8-bit PNG; use Rust tests for 16-bit")
        if image.mode == "1":
            image = image.convert("L")
        elif image.mode == "P":
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        elif "transparency" in image.info:
            image = image.convert("LA" if image.mode == "L" else "RGBA")
        return image.size, image.mode, image.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, action="append", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--timeout", type=float, default=120, help="Per-process limit in seconds")
    parser.add_argument("--profile", choices=("summary", "trace"), help="Show encoder stage diagnostics on stderr")
    parser.add_argument("--skip-stdin-check", action="store_true", help="Do not re-encode the first image via stdin")
    parser.add_argument("--encoder-arg", action="append", default=[], help="Repeat as --encoder-arg=--option and --encoder-arg=value")
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    environment = dict(os.environ)
    if args.profile:
        environment["RS_JXL_PROFILE"] = "1" if args.profile == "summary" else "trace"
    encoder = str(args.encoder.resolve(strict=True))
    decoders = [str(path.resolve(strict=True)) for path in args.decoder]
    print("file | dimensions | threads | PNG bytes | JXL bytes | bpp | CLI seconds | MPix/s")
    with tempfile.TemporaryDirectory(prefix="rs-jxl-png-cli-") as directory:
        directory = Path(directory)
        for index, source in enumerate(args.images):
            source = source.resolve(strict=True)
            original = source.read_bytes()
            expected = png_samples(original)
            width, height = expected[0]
            pixels = width * height
            for threads in args.threads:
                command = [encoder, "encode", str(source), "-", "--strip-metadata", "--threads", str(threads)]
                command.extend(args.encoder_arg)
                start = perf_counter()
                encoded = run(command, timeout=args.timeout, env=environment, show_stderr=bool(args.profile))
                elapsed = perf_counter() - start
                variants = [("file", encoded)]
                if index == 0 and not args.skip_stdin_check:
                    command[2] = "-"
                    variants.append(("stdin", run(command, input=original, timeout=args.timeout,
                                                  env=environment, show_stderr=bool(args.profile))))
                for variant, data in variants:
                    jxl = directory / "input.jxl"
                    jxl.write_bytes(data)
                    for decoder_index, decoder in enumerate(decoders):
                        output = directory / f"output-{decoder_index}.png"
                        output.unlink(missing_ok=True)
                        run([decoder, str(jxl), str(output), "--bits_per_sample=8", "--num_threads=1", "--quiet"], timeout=args.timeout)
                        actual = png_samples(output.read_bytes())
                        if actual != expected:
                            raise AssertionError(f"pixel mismatch: {source}, {variant}, {threads=}, {decoder}")
                print(f"{source.name} | {width}x{height} | {threads} | {len(original)} | {len(encoded)} | "
                      f"{8 * len(encoded) / pixels:.6f} | {elapsed:.4f} | {pixels / elapsed / 1e6:.3f}", flush=True)
    print("PASS: full samples matched, including RGB under alpha=0; temporary outputs removed.")


if __name__ == "__main__":
    main()
