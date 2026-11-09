"""Alternate in-memory JPEG reconstruction through two native DLLs/shared libs.

The public Decoder API is timed, including FFI and the returned bytes copy,
but excluding input IO, library/context creation, and original-byte validation.
The only mocked operation selects each context's library at construction; each
context then owns its actual native handle and matching result/free functions.
No codec operations are mocked. Run in a dedicated process, never an app host.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

from PIL import Image

from compare_cpp_port import clean_environment, file_sha256


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before", type=Path, required=True)
    cli.add_argument("--after", type=Path, required=True)
    cli.add_argument("--case", type=Path, nargs=2, action="append", required=True,
                     metavar=("JXL", "ORIGINAL_JPEG"))
    cli.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    cli.add_argument("--repeats", type=int, default=20, help="Measured calls per library/case")
    cli.add_argument("--warmups", type=int, default=3, help="Unmeasured calls per library/case")
    cli.add_argument("--out", type=Path, required=True, help="New JSON report path")
    args = cli.parse_args()
    if (args.repeats < 1 or args.warmups < 0 or len(set(args.threads)) != len(args.threads)
            or any(not 1 <= count <= 256 for count in args.threads)):
        cli.error("Invalid repetition/thread settings")
    if args.out.exists():
        cli.error("Output already exists")

    env, environment = clean_environment(os.environ)
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import rs_jxl
    from rs_jxl import _ffi

    report = dict(status="running", measurement="Public in-memory Decoder.reconstruct_jpeg call",
                  environment=environment, config=vars(args), binaries={}, cases=[], summaries=[])
    identities = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        libraries = {}
        for name in ("before", "after"):
            path = getattr(args, name).resolve(strict=True)
            digest = file_sha256(path)
            identities.append((path, digest))
            lib = _ffi.ffi.dlopen(str(path))
            _ffi._validate_abi(lib, path)
            libraries[name] = lib
            report["binaries"][name] = dict(path=str(path), sha256=digest)

        for index, (encoded_path, original_path) in enumerate(args.case):
            encoded_path = encoded_path.resolve(strict=True)
            original_path = original_path.resolve(strict=True)
            for path in (encoded_path, original_path):
                identities.append((path, file_sha256(path)))
            encoded, original = encoded_path.read_bytes(), original_path.read_bytes()
            with Image.open(original_path) as image:
                if image.format != "JPEG":
                    raise ValueError(f"Original is not JPEG: {original_path}")
                width, height = image.size
                progressive = bool(image.info.get("progressive"))
            case = dict(index=index, label=original_path.name, jxl=str(encoded_path),
                        original=str(original_path), width=width, height=height,
                        progressive=progressive, pixels=width * height,
                        jxl_bytes=len(encoded), bpp=8 * len(encoded) / (width * height))
            report["cases"].append(case)
            for threads in args.threads:
                row = dict(case_index=index, label=case["label"], threads=threads,
                           bpp=case["bpp"], samples={name: [] for name in libraries},
                           verified_calls={name: 0 for name in libraries})
                report["summaries"].append(row)
                with ExitStack() as stack:
                    decoders = {}
                    for name, lib in libraries.items():
                        with patch.object(_ffi, "library", return_value=lib):
                            decoders[name] = stack.enter_context(rs_jxl.Decoder(threads=threads))
                    for repetition in range(-args.warmups, args.repeats):
                        names = list(libraries)
                        if (repetition + index + threads) % 2:
                            names.reverse()
                        for name in names:
                            start = time.perf_counter()
                            restored = decoders[name].reconstruct_jpeg(encoded)
                            seconds = time.perf_counter() - start
                            if restored != original:
                                raise ValueError(f"Original JPEG mismatch: {case['label']}/{name}")
                            row["verified_calls"][name] += 1
                            del restored
                            if repetition >= 0:
                                row["samples"][name].append(seconds)
                row["decoders"] = {}
                for name, samples in row["samples"].items():
                    median = statistics.median(samples)
                    row["decoders"][name] = dict(seconds=median, min_seconds=min(samples),
                                                 max_seconds=max(samples),
                                                 mpix_s=case["pixels"] / 1e6 / median)
                before = row["decoders"]["before"]["seconds"]
                after = row["decoders"]["after"]["seconds"]
                row["speedup"] = before / after
                print(f"{case['label']} {threads}T: {before * 1000:.3f} -> "
                      f"{after * 1000:.3f} ms; {row['speedup']:.3f}x; "
                      f"{row['decoders']['after']['mpix_s']:.3f} MPix/s", flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            for path, digest in identities:
                if file_sha256(path) != digest:
                    raise ValueError(f"Benchmark input/library changed: {path}")
        except OSError as error:
            report.update(status="failed", error=str(error))
            raise
        except ValueError as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            with args.out.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, default=str)
                output.write("\n")


if __name__ == "__main__":
    main()
