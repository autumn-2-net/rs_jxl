"""Partial-channel palette trials using the private offsets/refinement runner.

All six variants use ordinary native e9, full-frame buffering and the same
explicit palette cap. Official djxl checks original samples, including hidden
RGB and 16-bit precision. No builds; size/time regressions are reportable.
"""

import argparse
import math
from pathlib import Path
import struct
import sys
import zlib

import benchmark_fast_offsets as offsets
import benchmark_refinement as refinement


MODES = ("rg", "yco", "ycg", "cocg", "ya-cocg")
VARIANTS = ("baseline",) + tuple(f"group-{mode}" for mode in MODES)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True, help="Prebuilt encoder_lab")
    parser.add_argument("--decoder", "--official-djxl", "--djxl", dest="decoder",
                        type=Path, required=True, help="Caller-attested official djxl")
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--effort", type=int, choices=(9,), default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--palette-group-colors", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(variants=list(VARIANTS), frozen=None, buffering=0, lz=0)
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not 2 <= args.palette_group_colors <= 70911
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid threads/repeats/warmups/palette-group-colors/timeout")
    if not args.case and args.manifest is None and not args.synthetic:
        parser.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        parser.error("Report already exists; it will not be overwritten")
    return args


def mode_for(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return None if variant == "baseline" else variant.removeprefix("group-")


def command(args, source, output, variant):
    mode = mode_for(variant)
    if args.effort != 9 or args.buffering != 0 or args.lz != 0:
        raise ValueError("Palette groups require ordinary e9, buffering 0 and default LZ policy")
    result = refinement.command(args, source, output, "baseline")
    result += ["--palette-group-colors", str(args.palette_group_colors)]
    if mode is not None:
        result += ["--palette-grouping", mode]
    return result


validate_measurement = offsets.validate_measurement


def validate_receipt(measurement, variant, colors=4096):
    mode = mode_for(variant)
    offsets.validate_receipt(measurement, "baseline")
    if "palette_grouping" not in measurement:
        raise ValueError("Missing palette_grouping receipt")
    receipt = measurement["palette_grouping"]
    if mode is None and receipt is None:
        return
    if (not isinstance(receipt, dict) or receipt.get("mode") != mode
            or type(receipt.get("colors")) is not int or receipt["colors"] != colors):
        raise ValueError("Requested palette grouping/cap was not executed")
    for key in ("attempts", "applied", "tables"):
        if type(receipt.get(key)) is not int or receipt[key] < 0:
            raise ValueError(f"Invalid palette grouping counter: {key}")
    tables_per_application = 2 if mode == "ya-cocg" else 1
    if (receipt["applied"] > receipt["attempts"]
            or receipt["tables"] != tables_per_application * receipt["applied"]
            or (mode is None and any(receipt[key] for key in ("attempts", "applied", "tables")))):
        raise ValueError("Inconsistent palette grouping work counts")


def validate_sizes(variants):
    # Compare the same repetitions, but do not require a real candidate to win.
    if set(variants) != set(VARIANTS):
        raise ValueError("Missing palette grouping variants")
    reference = [sample["repetition"] for sample in variants["baseline"]]
    if not reference or len(set(reference)) != len(reference):
        raise ValueError("Missing or duplicate paired repetitions")
    for samples in variants.values():
        repetitions = [sample["repetition"] for sample in samples]
        if len(repetitions) != len(reference) or set(repetitions) != set(reference):
            raise ValueError("Missing or duplicate paired repetitions")


def _chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def _rgb(y, co, cg):
    # Inverse of the native integer RCT6, including its floor shifts.
    t = y - (cg >> 1)
    b = t - (co >> 1)
    return b + co, t + cg, b


def fixtures(folder):
    paths = refinement.fixtures(folder)
    for depth in (8, 16):
        scale = 1 if depth == 8 else 256
        maximum = (1 << depth) - 1
        for mode in MODES:
            for channels in ((4,) if mode == "ya-cocg" else (3, 4)):
                width, height = 128, 128 if mode == "ya-cocg" else 64
                raw = bytearray()
                for pair in range(height):
                    row = []
                    for free in range(width):
                        if mode == "rg":
                            rgb = (scale * (32 + pair), scale * (48 + pair), scale * free)
                        else:
                            y, co, cg = 104 + pair % 16, 2 * (pair // 16) - 2, free - 64
                            if mode == "ycg":
                                co, cg = cg, co
                            elif mode in ("cocg", "ya-cocg"):
                                y, co, cg = 48 + free, 2 * (pair % 16) - 14, 2 * (pair // 16) - 6
                            rgb = _rgb(scale * y, scale * co, scale * cg)
                        # Nonzero low bits prevent a 16-bit fixture from being
                        # merely an 8-bit image expanded by multiplication by 257.
                        if depth == 16:
                            rgb = tuple(value + 37 for value in rgb)
                        row.extend(rgb)
                        if channels == 4:
                            row.append((0, maximum // 2, maximum)[free % 3])
                    raw.append(0)
                    raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(row), *row))
                header = struct.pack(">IIBBBBB", width, height, depth, 2 if channels == 3 else 6, 0, 0, 0)
                path = folder / f"pair-{mode}-{depth}bit-c{channels}.png"
                with path.open("xb") as output:
                    output.write(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
                                 + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))
                paths.append(path)
    return paths


def make_runner():
    runner = offsets.make_runner()
    runner.command = command
    runner.validate_measurement = validate_measurement
    runner.validate_receipt = validate_receipt
    runner.validate_sizes = validate_sizes
    runner.add_ratios = offsets.fast.add_ratios
    runner.fixtures = fixtures
    return runner


def benchmark(args):
    runner = make_runner()
    runner.validate_receipt = lambda measurement, variant: validate_receipt(
        measurement, variant, args.palette_group_colors)
    return runner.benchmark(args)


def main(argv=None):
    try:
        report = benchmark(arguments(argv))
    except (OSError, ValueError) as error:
        print(f"Benchmark/report failure: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
