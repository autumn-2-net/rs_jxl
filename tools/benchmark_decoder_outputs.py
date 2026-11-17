"""Compare decoder DLLs on a fixed manifest of original arrays / JPEG bytes.

No encoding, corpus discovery or builds. Expected .npy files must be prepared
before timing. The manifest identifies original-sample and baseline-render
oracles separately; baseline rendering is not a lossless correctness claim.
Every warmup and timed result is checked outside the timer. MPix counts all
returned full frames, not channels or compressed bytes.
"""

import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np

from compare_cpp_port import clean_environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(actual, expected):
    actual = actual if isinstance(actual, list) else [actual]
    if len(actual) != len(expected):
        raise AssertionError("decoded frame count differs")
    for frame, source in zip(actual, expected):
        if (frame.shape != source.shape or frame.dtype != source.dtype
                or not np.array_equal(frame.view(np.uint8), source.view(np.uint8))):
            raise AssertionError("decoded original sample words, dimensions or type differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--affinity", type=lambda s: int(s, 0), default=0,
                        help="Windows process affinity mask; zero retains scheduler default")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out.exists() or args.repeats < 1 or args.warmups < 0
            or any(not 1 <= n <= 256 for n in args.threads)
            or len(set(args.threads)) != len(args.threads)):
        parser.error("use a new output path and valid repetition/thread settings")
    if args.affinity:
        if os.name != "nt":
            raise RuntimeError("process affinity option currently requires Windows")
        kernel = ctypes.windll.kernel32
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1), args.affinity):
            raise RuntimeError("cannot set requested process affinity")
    env, environment = clean_environment(os.environ)
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import rs_jxl
    from rs_jxl import _ffi

    manifest = args.manifest.resolve(strict=True)
    spec = json.loads(manifest.read_text(encoding="utf-8-sig"))
    identities = [(manifest, digest(manifest))]
    report = {"status": "running", "config": vars(args), "environment": environment,
              "measurement": "reusable public Python Decoder call, including returned output",
              "verification": "every call, original array words or JPEG bytes outside timer",
              "binaries": {}, "cases": [], "summaries": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        libraries = {}
        for name in ("before", "after"):
            path = getattr(args, name).resolve(strict=True)
            identities.append((path, digest(path)))
            lib = _ffi.ffi.dlopen(str(path))
            _ffi._validate_abi(lib, path)
            libraries[name] = lib
            report["binaries"][name] = {"path": str(path), "sha256": digest(path)}
        for case_index, case in enumerate(spec["cases"]):
            source = (manifest.parent / case["jxl"]).resolve(strict=True)
            identities.append((source, digest(source)))
            encoded = source.read_bytes()
            op = case.get("operation", "decode")
            if op not in ("decode", "reconstruct_jpeg"):
                raise ValueError(f"unsupported operation: {op}")
            expected = []
            for item in case["expected"]:
                path = (manifest.parent / item).resolve(strict=True)
                identities.append((path, digest(path)))
                expected.append(path.read_bytes() if op == "reconstruct_jpeg"
                                else np.load(path, allow_pickle=False))
            pixels = (case["width"] * case["height"] if op == "reconstruct_jpeg"
                      else sum(frame.shape[0] * frame.shape[1] for frame in expected))
            if op == "reconstruct_jpeg" and len(expected) != 1:
                raise ValueError("JPEG requires exactly one original file")
            report["cases"].append({**case, "jxl_bytes": len(encoded), "pixels": pixels,
                                    "jxl_sha256": digest(source)})
            for threads in args.threads:
                row = {"label": case["label"], "operation": op, "threads": threads,
                       "oracle": case["oracle"], "pixels": pixels,
                       "samples": {name: [] for name in libraries}, "verified_calls": {}}
                report["summaries"].append(row)
                with ExitStack() as stack:
                    decoders = {}
                    for name, lib in libraries.items():
                        with patch.object(_ffi, "library", return_value=lib):
                            decoders[name] = stack.enter_context(rs_jxl.Decoder(threads=threads))
                    for repetition in range(-args.warmups, args.repeats):
                        names = list(libraries)
                        if (repetition + case_index) % 2:
                            names.reverse()
                        for name in names:
                            start = time.perf_counter()
                            actual = getattr(decoders[name], op)(encoded, **case.get("options", {}))
                            elapsed = time.perf_counter() - start
                            if op == "reconstruct_jpeg":
                                if actual != expected[0]:
                                    raise AssertionError("original JPEG bytes differ")
                            else:
                                exact(actual, expected)
                            del actual
                            row["verified_calls"][name] = row["verified_calls"].get(name, 0) + 1
                            if repetition >= 0:
                                row["samples"][name].append(elapsed)
                row["decoders"] = {}
                for name, values in row["samples"].items():
                    seconds = statistics.median(values)
                    row["decoders"][name] = {"seconds": seconds, "mpix_s": pixels / 1e6 / seconds,
                                              "min_seconds": min(values), "max_seconds": max(values)}
                before, after = (row["decoders"][name] for name in ("before", "after"))
                row["speedup"] = before["seconds"] / after["seconds"]
                print(f'{case["label"]} {threads}T: {before["seconds"] * 1000:.3f} -> '
                      f'{after["seconds"] * 1000:.3f} ms; {before["mpix_s"]:.3f} -> '
                      f'{after["mpix_s"]:.3f} MPix/s; {row["speedup"]:.3f}x', flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            for path, before in identities:
                if digest(path) != before:
                    raise AssertionError(f"input/library changed: {path}")
        except BaseException as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            report["files"] = [{"path": str(path), "sha256": value} for path, value in identities]
            with args.out.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, default=str)
                output.write("\n")


if __name__ == "__main__":
    main()
