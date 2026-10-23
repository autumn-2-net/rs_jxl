"""Opt-in archival CLI acceptance using official djxl as a decoder only.

Every decoded JPEG is compared to the complete original byte sequence. No
encoder-output equality is required. Timings include the Rust process startup,
input read, coefficient encoding, and output collection, but not verification.
Pillow only reads image dimensions for bpp; it is not part of the transcode.
"""

import argparse
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

from PIL import Image


def run(command, *, timeout, data=None, success=True):
    result = subprocess.run(command, input=data, capture_output=True, timeout=timeout)
    if (result.returncode == 0) != success:
        raise AssertionError(
            f"unexpected exit {result.returncode}: {command!r}\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def verify(data, original, directory, decoders, timeout):
    encoded = directory / "oracle-input.jxl"
    encoded.write_bytes(data)
    for index, decoder in enumerate(decoders):
        restored = directory / f"restored-{index}.jpg"
        restored.unlink(missing_ok=True)
        run([decoder, str(encoded), str(restored), "--num_threads=1", "--quiet"],
            timeout=timeout)
        if restored.read_bytes() != original:
            raise AssertionError(f"original JPEG bytes differ after {decoder}")
    return len(decoders)


def check_boundaries(encoder, original, directory, decoders, timeout, source_search=False):
    # Only private scratch paths are passed to --force or intentionally rejected.
    source = directory / "jpeg-with-wrong-extension.png"
    output = directory / "published.jxl"
    source.write_bytes(original)
    command = [encoder, "transcode-jpeg", str(source), str(output)]
    if source_search:
        command.append("--source-search")
    run(command, timeout=timeout)
    checks = verify(output.read_bytes(), original, directory, decoders, timeout)
    marker = b"existing output must survive a failed transcode"
    output.write_bytes(marker)
    run(command, timeout=timeout, success=False)
    if output.read_bytes() != marker:
        raise AssertionError("unforced output was overwritten")
    run(command + ["--force"], timeout=timeout)
    checks += verify(output.read_bytes(), original, directory, decoders, timeout)
    run([encoder, "transcode-jpeg", str(source), str(source), "--force"],
        timeout=timeout, success=False)
    if source.read_bytes() != original:
        raise AssertionError("same-path invocation changed the original JPEG")
    output.write_bytes(marker)
    for invalid in (b"not JPEG", original[:2]):
        source.write_bytes(invalid)
        run(command + ["--force"], timeout=timeout, success=False)
        if output.read_bytes() != marker:
            raise AssertionError("failed encoding modified an existing output")
        result = run([encoder, "transcode-jpeg", "-", "-"], data=invalid,
                     timeout=timeout, success=False)
        if result:
            raise AssertionError("failed encoding wrote partial binary stdout")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, action="append", required=True)
    parser.add_argument("--effort", type=int, nargs="+", default=[3, 9])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--source-search", action="store_true")
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("timeout must be finite and positive")
    if any(not 1 <= value <= 11 for value in args.effort):
        parser.error("effort must be 1..11")
    if any(not 1 <= value <= 256 for value in args.threads):
        parser.error("threads must be 1..256")
    encoder = str(args.encoder.resolve(strict=True))
    decoders = [str(path.resolve(strict=True)) for path in args.decoder]
    checks = 0
    print("file | dimensions | effort | threads | JPEG bytes | JXL bytes | ratio | bpp | CLI seconds | MPix/s")
    with tempfile.TemporaryDirectory(prefix="rs-jxl-jpeg-cli-") as scratch:
        directory = Path(scratch)
        for index, source in enumerate(args.images):
            source = source.resolve(strict=True)
            if source.stat().st_size > 16 << 20:
                raise ValueError("this acceptance helper caps each input at 16 MiB")
            original = source.read_bytes()
            with Image.open(BytesIO(original)) as image:
                if image.format != "JPEG":
                    raise ValueError(f"not JPEG: {source}")
                width, height = image.size
            pixels = width * height
            for effort in args.effort:
                for threads in args.threads:
                    command = [encoder, "transcode-jpeg", str(source), "-",
                               "--effort", str(effort), "--threads", str(threads)]
                    if args.source_search:
                        command.append("--source-search")
                    start = perf_counter()
                    encoded = run(command, timeout=args.timeout)
                    seconds = perf_counter() - start
                    checks += verify(encoded, original, directory, decoders, args.timeout)
                    if index == 0:
                        command[2] = "-"
                        streamed = run(command, timeout=args.timeout, data=original)
                        checks += verify(streamed, original, directory, decoders, args.timeout)
                    print(f"{source.name} | {width}x{height} | {effort} | {threads} | "
                          f"{len(original)} | {len(encoded)} | {len(encoded)/len(original):.3%} | "
                          f"{8*len(encoded)/pixels:.6f} | {seconds:.4f} | "
                          f"{pixels/seconds/1e6:.3f}", flush=True)
            if index == 0:
                checks += check_boundaries(encoder, original, directory, decoders, args.timeout,
                                           args.source_search)
    print(f"PASS: {checks} original-JPEG byte comparisons; file/stdin/stdout and failure boundaries checked.")


if __name__ == "__main__":
    main()
