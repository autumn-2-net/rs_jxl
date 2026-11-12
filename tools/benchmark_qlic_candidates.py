"""Compare prebuilt patch_lab candidates using native time and official djxl.

No builds, downloads or discovery. Requires numpy/opencv-python at runtime for
original 8/16-bit PNG samples, including RGB under zero alpha. Native seconds
exclude process startup and PNG/input/output IO. Regressions are reported only.
Use --repeats 1 for smoke measurements or the default 3 for per-case medians.
"""

import argparse
import json
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
import zlib

from benchmark_fast_context import load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
from benchmark_palette_runs import prepare_png
from corpus_inputs import png_framing
import benchmark_tile_patches as tiles
import compare_cpp_port as port


SOURCE_BLOCKS = {
    "source-ref8": ("reference", 8), "source-ref16": ("reference", 16),
    "source-res8": ("residual", 8), "source-res16": ("residual", 16),
    "source-both8": ("both", 8), "source-both16": ("both", 16),
    "source-ref16-parallel": ("reference", 16),
    "source-res16-parallel": ("residual", 16),
    "source-both16-parallel": ("both", 16),
}
PARALLEL_MODES = ("baseline-parallel", "source-parallel", "source-ref16-parallel",
                  "source-res16-parallel", "source-both16-parallel")
MODES = ("baseline", "source", "tiles", "offgrid", "block8", "block16", *SOURCE_BLOCKS,
         "baseline-parallel", "source-parallel")
BLOCK_COUNTERS = ("trees", "pixels", "views", "requested_tiles", "leaves",
                  "spatial_splits", "channel_splits", "stream_splits", "coarsened_views")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True, help="Prebuilt patch_lab executable")
    p.add_argument("--decoder", "--official-djxl", "--djxl", dest="decoder", type=Path,
                   required=True, help="Caller-attested official djxl executable")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tile-size", type=int)
    p.add_argument("--tile-admission", choices=tiles.ADMISSIONS, default="screened")
    p.add_argument("--block-leaves", type=int, help="Override the block planner's work budget")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300, help="Harness process guard only")
    p.add_argument("--out", type=Path, required=True, help="New JSON report, never overwritten")
    args = p.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not positive_number(args.timeout)
            or (args.tile_size is not None and args.tile_size < 1)
            or (args.block_leaves is not None and args.block_leaves < 1)):
        p.error("Invalid threads/tile-size/repeats/warmups/timeout")
    if len(set(args.modes)) != len(args.modes) or "baseline" not in args.modes:
        p.error("Use distinct modes including baseline")
    if not args.case and args.manifest is None and not args.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return args


def native_mode(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown candidate: {mode}")
    if mode in SOURCE_BLOCKS:
        return "source"
    if mode in PARALLEL_MODES:
        return mode.removesuffix("-parallel")
    return "tiles" if mode == "offgrid" else "baseline" if mode.startswith("block") else mode


def block_settings(mode):
    if mode in SOURCE_BLOCKS:
        return SOURCE_BLOCKS[mode]
    if mode in ("block8", "block16"):
        return "residual", int(mode[5:])
    return None, None


def command(args, source, output, mode):
    result = tiles.command(args, source, output, native_mode(mode))
    if mode == "offgrid":
        result.append("--off-grid")
    target, size = block_settings(mode)
    if size is not None:
        result += ["--block-size", str(size)]
        if mode in SOURCE_BLOCKS:
            result += ["--block-target", target]
        if args.block_leaves is not None:
            result += ["--block-leaves", str(args.block_leaves)]
    if mode in PARALLEL_MODES:
        result.append("--parallel-gather")
    return result


def fixtures(folder):
    """Retain format/control fixtures and add shifted repeats and mixed directions.

    Write native big-endian 16-bit PNG samples directly, never via Pillow.
    Shifted copies include all RGBA samples and one grid-aligned source seed.
    """
    paths = tiles.fixtures(folder)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    for depth in (8, 16):
        top = (1 << depth) - 1
        for pattern in ("shifted-tiles", "mixed-direction"):
            width, height = 149, 113
            rng = random.Random(811 + depth)
            raw = bytearray()
            for y in range(height):
                row = []
                for x in range(width):
                    local = next(((x - sx, y - sy) for sx, sy in
                                  ((0, 0), (37, 5), (73, 5), (109, 5),
                                   (5, 41), (37, 41), (73, 41), (109, 41),
                                   (17, 79), (49, 79), (81, 79), (113, 79))
                                  if 0 <= x - sx < 16 and 0 <= y - sy < 16), None)
                    for c in range(4):
                        if pattern == "shifted-tiles":
                            if local is None:
                                value = rng.randrange(top + 1)
                            else:
                                u, v = local
                                word = u * 0x45d9f3b ^ v * 0x119de1f3 ^ (c + 1) * 0x9e3779b9
                                value = ((word ^ (word >> 16)) * 0x45d9f3b) & top
                            hidden = local is not None and sum(local) % 5 == 0
                        else:
                            # Adjacent regions favor top, left, gradient and noisy predictors.
                            region = int(x >= 73) + 2 * int(y >= 55)
                            value = ((x * 1237 + c * 701) if region == 0 else
                                     (y * 1879 + c * 911) if region == 1 else
                                     (x * 277 + y * 619 + c * 1013) if region == 2 else
                                     rng.randrange(top + 1)) & top
                            hidden = (x + y) % 7 == 0
                        row.append(0 if c == 3 and hidden else value)
                raw.append(0)
                raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(row), *row))
            header = struct.pack(">IIBBBBB", width, height, depth, 6, 0, 0, 0)
            path = folder / f"{pattern}-{depth}-rgba.png"
            path.write_bytes(port.PNG_SIGNATURE + chunk(b"IHDR", header)
                             + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
            paths.append(path)
    return paths


def validate_measurement(m, mode, size, width, height, tile_size=16, tile_admission="screened", block_leaves=None):
    mapped = native_mode(mode)
    tiles.validate_measurement(m, mapped, size, width, height, tile_size, tile_admission)
    if mode in PARALLEL_MODES:
        if m.get("parallel_gather") is not True:
            raise ValueError("Missing parallel gather activation receipt")
    elif "parallel_gather" in m and m["parallel_gather"] is not False:
        raise ValueError("Unexpected parallel gather activation")
    if any(type(m.get(k)) is not int for k in ("width", "height")):
        raise ValueError("Image dimensions must be integers")
    if "block_size" not in m or "block_work" not in m:
        raise ValueError("Missing block receipt (inactive receipts must be null)")
    target, expected_block = block_settings(mode)
    if (m["block_size"] != expected_block
            or (expected_block is not None and type(m["block_size"]) is not int)):
        raise ValueError("Requested block size differs from receipt")
    work = m["block_work"]
    budget = (block_leaves or 4096) if expected_block else None
    if ("block_plan_leaves" not in m or m["block_plan_leaves"] != budget
            or (budget is not None and type(m["block_plan_leaves"]) is not int)):
        raise ValueError("Missing/mismatched block planning budget")
    # Older off-mode receipts predate block_target and retain their flat schema.
    if "block_target" in m and m["block_target"] != target:
        raise ValueError("Requested block target differs from receipt")
    if expected_block is None:
        if work is not None:
            raise ValueError("Inactive block work must be null")
    elif mode in SOURCE_BLOCKS:
        if m.get("block_target") != target:
            raise ValueError("Missing/mismatched source block target")
        if not isinstance(work, dict) or set(work) != {"reference", "residual"}:
            raise ValueError("Source block work must separate reference and residual receipts")
        for part in ("reference", "residual"):
            part_work = work[part]
            requested = target in (part, "both")
            if not requested:
                if part_work is not None:
                    raise ValueError("Inactive source block work must be null")
                continue
            validate_block_work(part_work)
            configured = part == "residual" or m["source_detector_enabled"]
            if part_work.get("configured") is not configured:
                raise ValueError("Missing/mismatched source block configuration")
            if (part_work.get("used") is not (part_work["trees"] > 0)
                    or (not configured and part_work["trees"] != 0)):
                raise ValueError("Inconsistent source block actual usage")
    else:
        validate_block_work(work)
    if mode in ("tiles", "offgrid"):
        work = m["tile_work"]
        if (type(work.get("tile_size")) is not int
                or work.get("off_grid") is not (mode == "offgrid")):
            raise ValueError("Missing/mismatched off-grid tile receipt")
        if any(type(work.get(k)) is not int or work[k] < 0
               for k in ("windows_examined", "discovery_limited")):
            raise ValueError("Invalid tile discovery counter")


def validate_block_work(work):
    if not isinstance(work, dict) or any(
            type(work.get(k)) is not int or work[k] < 0 for k in BLOCK_COUNTERS):
        raise ValueError("Missing/invalid block work counter")
    predictors = work.get("predictor_leaves")
    if (not isinstance(predictors, list) or len(predictors) != 4
            or any(type(n) is not int or n < 0 for n in predictors)):
        raise ValueError("Block predictor_leaves must contain four nonnegative integers")
    # A forest of full binary trees has leaves = trees + splits. Work is on
    # transformed views; pixels/views need not equal original image counts.
    if (sum(predictors) != work["leaves"]
            or work["leaves"] != work["trees"] + sum(work[k] for k in
                ("spatial_splits", "channel_splits", "stream_splits"))
            or work["coarsened_views"] > work["views"]
            or (work["trees"] == 0 and (any(work[k] for k in BLOCK_COUNTERS)
                                           or any(predictors)))):
        raise ValueError("Inconsistent block work accounting")


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(), cases=[], errors=[],
                  timing="Native discovery+admission+encode only; excludes startup and input/output IO",
                  correctness="Official djxl vs original 8/16-bit samples including hidden RGB",
                  official_decoder_provenance="Caller-attested; executable path and SHA256 recorded",
                  regression_policy="Size/time regressions reported, not failures",
                  ratio_contract="mode / baseline; aggregate uses sums of per-case medians")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            for name in ("encoder", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(args, name, path)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                digest = port.file_sha256(args.manifest)
                identities.append((args.manifest, digest))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=digest)
            with tempfile.TemporaryDirectory(prefix="qlic-candidates-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                external = set(sources)
                if args.synthetic:
                    sources += fixtures(temporary / "synthetic")
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    if source in external:
                        identities.append((source, digest))
                    payload, expected, preparation = prepare_png(source, original, temporary)
                    _, info = png_framing(original)
                    info["mode"] = preparation
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(payload)
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=info, snapshot_sha256=port.sha256(payload),
                                variants={mode: [] for mode in args.modes}, summaries={})
                    report["cases"].append(case)
                    for repeat in range(-args.warmups, args.repeats):
                        for mode in trial_order(args.modes, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][mode].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                sample["encoder"] = port.run_process(
                                    command(args, snapshot, jxl, mode), env, args.timeout)
                                port.check_process(sample["encoder"])
                                sample["decoder"] = port.run_process(
                                    port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                port.check_process(sample["decoder"])
                                port.compare_pixels(expected, port.png_samples(png.read_bytes()))
                                measurement = json.loads(sample["encoder"]["stdout"])
                                validate_measurement(measurement, mode, jxl.stat().st_size,
                                                     expected.width, expected.height, args.tile_size or 16,
                                                     args.tile_admission, args.block_leaves)
                                sample.update(verified=True, measurement=measurement)
                            print(f"{source.name} {mode} repeat={repeat}: {measurement['bytes']} bytes "
                                  f"{measurement['seconds']:.6f}s selected={measurement['selected_mode']} verified",
                                  flush=True)
                    case["summaries"] = {mode: summarize(samples, pixels)
                                         for mode, samples in case["variants"].items()}
                    tiles.add_comparisons(case["summaries"])
                report["aggregate"] = tiles.aggregate(report["cases"], args.modes)
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if port.file_sha256(path) == digest:
                        continue
                    error = f"Input/binary changed during measurement: {path}"
                except OSError as failure:
                    error = str(failure)
                report["errors"].append(error)
            if report["errors"] or not report["temporary_outputs_cleaned"]:
                report["status"] = "failed"
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


def main(argv=None):
    args = arguments(argv)
    try:
        report = benchmark(args)
    except (OSError, ValueError) as error:
        print(f"Benchmark/report failure: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
