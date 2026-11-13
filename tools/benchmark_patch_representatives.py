"""Source Patch reference-instance ablation, not a detector or scheduler change.

First/Medoid/entropy share all native e9 settings, source gates, grouping and
atlas packing. Complete outputs are measured, including larger/slower trials.
Official djxl checks original samples. A bounded candidate budget of zero is
exhaustive for that dimension, not a production input limit.
"""

import argparse
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import zlib

import benchmark_patch_ma as patch
from representation_inputs import load_verified_cases

VARIANTS = ("off", "first", "medoid", "entropy")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--occurrences", type=int, default=16)
    p.add_argument("--pixels", type=int, default=128)
    p.add_argument("--parallel-gather", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or min(args.candidates, args.occurrences, args.pixels) < 0):
        p.error("Invalid thread/repeat/budget/timeout")
    if "first" not in args.variants or len(set(args.variants)) != len(args.variants):
        p.error("Distinct variants including first are required")
    if not args.case and args.manifest is None and not args.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        p.error("Report already exists; not overwritten")
    return args


def command(args, source, output, variant, profile=False):
    if variant not in VARIANTS:
        raise ValueError("Unknown representative variant")
    base = "off" if variant == "off" else "source"
    if args.parallel_gather:
        base += "-gather"
    result = patch.command(args, source, output, base, profile)
    if variant != "off":
        result += ["--source-representative", variant,
                   "--representative-candidates", str(args.candidates),
                   "--representative-occurrences", str(args.occurrences),
                   "--representative-pixels", str(args.pixels)]
    return result


def validate_measurement(args, m, variant, size, width, height):
    base = "off" if variant == "off" else "source"
    patch.validate_measurement(m, base + ("-gather" if args.parallel_gather else ""), size, width, height)
    work = m.get("representative_work")
    if variant == "off":
        if work is not None:
            raise ValueError("Off route cannot have representative work")
        return
    if not isinstance(work, dict) or work.get("mode") != variant:
        raise ValueError("Missing or mismatched representative receipt")
    for field, expected in [("candidates_budget", args.candidates),
                            ("occurrences_budget", args.occurrences), ("pixels_budget", args.pixels)]:
        if type(work.get(field)) is not int or work[field] != expected:
            raise ValueError("Representative budget not executed")
    for field in ("retained_groups", "occurrences", "searched_groups", "variable_sample_groups",
                  "changed_groups", "candidates", "sampled_occurrences", "sampled_pixels",
                  "emitted_plans", "atlas_pixels", "dictionary_positions", "float_fallback_groups"):
        if type(work.get(field)) is not int or work[field] < 0:
            raise ValueError(f"Invalid representative counter {field}")
    for field in ("selection_seconds", "baseline_proxy", "selected_proxy"):
        value = work.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid representative score/timer {field}")
    if (not 0 <= work["changed_groups"] <= work["variable_sample_groups"] <= work["searched_groups"] <= work["retained_groups"]
            or work["selected_proxy"] > work["baseline_proxy"]
            or work["emitted_plans"] > 1
            or bool(work["emitted_plans"]) != bool(work["atlas_pixels"])
            or bool(work["emitted_plans"]) != bool(work["dictionary_positions"])):
        raise ValueError("Inconsistent representative work")
    if variant == "first" and (work["searched_groups"] or work["changed_groups"] or work["selection_seconds"]):
        raise ValueError("First must not perform representative search")


def fixtures(folder):
    """Quantized-equal groups with unequal exact samples and a poor first instance."""
    paths = []
    width, height = 129, 83
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    for depth in (8, 16):
        for channels, kind in ((1, 0), (2, 4), (3, 2), (4, 6)):
            scale = 1 if depth == 8 else 257
            raw = bytearray()
            for y in range(height):
                row = []
                for x in range(width):
                    v = 0
                    for x0, y0, level in ((73, 17, 82), (17, 37, 87), (49, 57, 83), (97, 37, 82)):
                        if x0 <= x < x0 + 5 and y0 <= y < y0 + 5:
                            v = level * scale
                    row.extend([v] * (1 if channels < 3 else 3))
                    if channels in (2, 4):
                        row.append(0)  # All color must survive even when invisible.
                raw.append(0)
                raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(row), *row))
            path = folder / f"representative-{depth}-{channels}.png"
            header = struct.pack(">IIBBBBB", width, height, depth, kind, 0, 0, 0)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
            paths.append(path)
    return paths


def benchmark(args):
    # Private harness module: existing scripts/tests keep their own dispatch.
    spec = importlib.util.spec_from_file_location("_patch_representative_runner", Path(patch.__file__))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.load_cases = load_verified_cases
    runner.command = command
    runner.validate_measurement = lambda m, v, s, w, h: validate_measurement(args, m, v, s, w, h)
    if not args.synthetic:
        return runner.benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="representative-fixtures-", dir=args.out.parent) as folder:
        args.case = args.case + fixtures(Path(folder))
        return runner.benchmark(args)


def main(argv=None):
    try:
        report = benchmark(arguments(argv))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
