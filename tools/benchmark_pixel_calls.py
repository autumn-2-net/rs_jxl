"""A/B reusable in-memory pixel APIs through two actual CFFI libraries.

No builds or corpus discovery. Run in a dedicated process. Only context
construction patches _ffi.library; every encode/decode uses its bound DLL.
Timers exclude IO, context construction, hashing and sample verification.
Encode includes array transport and output copies, with native codec_seconds
reported separately. Decode includes the returned NumPy array. Both DLLs decode
one fixed before-encoded JXL per case, also shared across thread settings.
Every candidate (including warmups) is cross-decoded by both libraries.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

from compare_cpp_port import clean_environment, file_sha256, png_payload, png_samples, sha256


NAMES = ("before", "after")


def parse_args(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before", type=Path, required=True)
    cli.add_argument("--after", type=Path, required=True)
    cli.add_argument("--case", type=Path, action="append", required=True, metavar="ORIGINAL_PNG")
    cli.add_argument("--effort", type=int, default=9)
    cli.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    cli.add_argument("--repeats", type=int, default=3, help="Measured calls per DLL/case/thread setting")
    cli.add_argument("--warmups", type=int, default=1)
    cli.add_argument("--out", type=Path, required=True, help="New JSON report path (never overwritten)")
    args = cli.parse_args(argv)
    if (not 1 <= args.effort <= 11 or args.repeats < 1 or args.warmups < 0
            or len(set(args.threads)) != len(args.threads)
            or any(not 1 <= count <= 256 for count in args.threads)):
        cli.error("Invalid effort/repetition/thread settings")
    if args.out.exists():
        cli.error("Output already exists")
    return args


def load_case(index, path):
    import numpy as np

    path = path.resolve(strict=True)
    data = path.read_bytes()
    pixels = png_samples(data)  # Pixels(width, height, depth, rgba), not a tuple.
    _, info = png_payload(data)
    color = info["color_type"]
    # Retain gray/GA/RGB/RGBA; palette expands to RGBA, tRNS to GA/RGBA.
    channels = {0: 2 if "transparency_key" in info else 1,
                2: 4 if "transparency_key" in info else 3, 3: 4, 4: 2, 6: 4}[color]
    if channels == 2:
        array = pixels.rgba[:, :, [0, 3]]
    else:
        array = pixels.rgba[:, :, :channels]
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    case = dict(index=index, label=path.name, original=str(path), original_sha256=sha256(data),
                original_bytes=len(data), width=pixels.width, height=pixels.height,
                depth=pixels.depth, pixels=pixels.width * pixels.height,
                png_color_type=color, channels=channels, shape=list(array.shape),
                dtype=str(array.dtype), pixel_bytes=array.nbytes,
                pixel_sha256=sha256(memoryview(array).cast("B")))
    return case, array


def verify_samples(expected, actual):
    import numpy as np

    if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError("Decoded shape/dtype differs from original samples (no narrowing allowed)")
    for y in range(expected.shape[0]):
        mismatch = np.flatnonzero(actual[y] != expected[y])
        if mismatch.size:
            x, channel = divmod(int(mismatch[0]), expected.shape[2])
            raise ValueError(f"Original sample mismatch x={x} y={y} channel={channel}: "
                             f"{int(expected[y, x, channel])} != {int(actual[y, x, channel])}; "
                             "hidden RGB is included")


def codec_seconds(meta):
    report = meta.get("report") if isinstance(meta, dict) else None
    value = report.get("codec_seconds") if isinstance(report, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Missing/invalid encode report.codec_seconds")
    return value


def encoded_info(encoded, pixels):
    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError("Encoder must return nonempty in-memory bytes")
    return dict(bytes=len(encoded), bpp=8 * len(encoded) / pixels, sha256=sha256(encoded))


def call_order(repetition, case_index, threads):
    return NAMES[::-1] if (repetition + case_index + threads) % 2 else NAMES


def timing_summary(samples, key, pixels):
    values = [sample[key] for sample in samples if not sample["warmup"]]
    median = statistics.median(values)
    return dict(seconds=median, min_seconds=min(values), max_seconds=max(values),
                mpix_s=pixels / 1e6 / median if median > 0 else None)


def measure_case(args, case, array, encoders, decoders, row, baseline):
    threads, pixels = row["threads"], case["pixels"]
    row["phase"] = "baseline"
    if baseline is None:
        baseline, meta = encoders["before"].encode(
            array, effort=args.effort, source_search=False, return_info=True)
        case["baseline"] = dict(encoder="before", threads=threads, codec_seconds=codec_seconds(meta),
                                **encoded_info(baseline, pixels))
    for name in NAMES:
        verify_samples(array, decoders[name].decode(baseline))
        row["verified_calls"]["baseline"][name] += 1

    row["phase"] = "encode"
    for repetition in range(-args.warmups, args.repeats):
        order = call_order(repetition, case["index"], threads)
        candidates = []
        for name in order:
            row["active_call"] = dict(library=name, repetition=repetition)
            start = time.perf_counter()
            encoded, meta = encoders[name].encode(
                array, effort=args.effort, source_search=False, return_info=True)
            seconds = time.perf_counter() - start
            sample = dict(repetition=repetition, warmup=repetition < 0, seconds=seconds,
                          codec_seconds=codec_seconds(meta), **encoded_info(encoded, pixels))
            row["encode"][name]["samples"].append(sample)
            candidates.append((name, encoded))
        # Cross-validation never runs inside either API timer. Byte identity is
        # not an acceptance criterion, including between repeated encodes.
        for encoder, encoded in candidates:
            for decoder in order:
                row["active_call"] = dict(encoder=encoder, decoder=decoder, repetition=repetition)
                verify_samples(array, decoders[decoder].decode(encoded))
                row["verified_calls"]["cross_decode"][encoder][decoder] += 1
        del candidates, encoded

    row["phase"] = "decode"
    for repetition in range(-args.warmups, args.repeats):
        for name in call_order(repetition, case["index"], threads):
            row["active_call"] = dict(library=name, repetition=repetition)
            start = time.perf_counter()
            actual = decoders[name].decode(baseline)
            seconds = time.perf_counter() - start
            row["decode"][name]["samples"].append(dict(
                repetition=repetition, warmup=repetition < 0, seconds=seconds))
            verify_samples(array, actual)
            row["verified_calls"]["decode"][name] += 1
            del actual

    for name in NAMES:
        for phase in ("encode", "decode"):
            entry = row[phase][name]
            entry["whole_call"] = timing_summary(entry["samples"], "seconds", pixels)
        entry = row["encode"][name]
        entry["codec"] = timing_summary(entry["samples"], "codec_seconds", pixels)
    row["speedup"] = {
        phase: row[phase]["before"]["whole_call"]["seconds"] / row[phase]["after"]["whole_call"]["seconds"]
        for phase in ("encode", "decode")}
    before = row["encode"]["before"]["codec"]["seconds"]
    after = row["encode"]["after"]["codec"]["seconds"]
    row["speedup"]["encode_codec"] = before / after if after > 0 else None
    row.update(status="passed", phase="complete")
    row.pop("active_call", None)
    return baseline


def benchmark(args):
    env, environment = clean_environment(os.environ)
    os.environ.clear()
    os.environ.update(env)
    report = dict(status="running", created=datetime.now(timezone.utc).isoformat(),
                  config=vars(args), environment=environment, python=sys.version,
                  measurement=dict(encode="Whole Encoder.encode(array) call; native codec_seconds separately",
                                   decode="Whole Decoder.decode(fixed before JXL) call returning NumPy",
                                   excluded="IO, construction, hashing, verification; warmups not aggregated",
                                   pixels="Original integer samples including hidden RGB; metadata excluded"),
                  binaries={}, cases=[], summaries=[], errors=[])
    identities = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the report before doing work, without ever replacing an old one.
    with args.out.open("x", encoding="utf-8") as output:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
            import rs_jxl
            from rs_jxl import _ffi

            libraries = {}
            for name in NAMES:
                path = getattr(args, name).resolve(strict=True)
                digest = file_sha256(path)
                identities.append((path, digest))
                report["binaries"][name] = dict(path=str(path), sha256=digest)
                lib = _ffi.ffi.dlopen(str(path))
                _ffi._validate_abi(lib, path)
                libraries[name] = lib
            for index, path in enumerate(args.case):
                case, array = load_case(index, path)
                report["cases"].append(case)
                identities.append((Path(case["original"]), case["original_sha256"]))
                baseline = None
                for threads in args.threads:
                    row = dict(case_index=index, label=case["label"], threads=threads, status="running",
                               encode={name: dict(samples=[]) for name in NAMES},
                               decode={name: dict(samples=[]) for name in NAMES},
                               verified_calls=dict(baseline={name: 0 for name in NAMES},
                                                   cross_decode={e: {d: 0 for d in NAMES} for e in NAMES},
                                                   decode={name: 0 for name in NAMES}))
                    report["summaries"].append(row)
                    with ExitStack() as stack:
                        encoders, decoders = {}, {}
                        for name, lib in libraries.items():
                            with patch.object(_ffi, "library", return_value=lib):
                                encoders[name] = stack.enter_context(rs_jxl.Encoder(threads=threads))
                                decoders[name] = stack.enter_context(rs_jxl.Decoder(threads=threads))
                        baseline = measure_case(args, case, array, encoders, decoders, row, baseline)
                    print(f"{case['label']} {threads}T: encode {row['speedup']['encode']:.3f}x; "
                          f"decode {row['speedup']['decode']:.3f}x", flush=True)
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            for path, digest in identities:
                try:
                    if file_sha256(path) != digest:
                        report["errors"].append(f"Input/library changed during measurement: {path}")
                except OSError as error:
                    report["errors"].append(str(error))
            if report["errors"]:
                report["status"] = "failed"
                for row in report["summaries"]:
                    if row["status"] == "running":
                        row["status"] = "failed"
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


def main(argv=None):
    args = parse_args(argv)
    report = benchmark(args)
    print(report["status"], args.out, *report["errors"], sep="\n")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
