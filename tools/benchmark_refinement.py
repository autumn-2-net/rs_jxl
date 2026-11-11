"""Focused post-training trials; compare original PNG samples with official djxl.

Native-call timings exclude image input, process startup and verification.
No persistent encoded files, corpus discovery, build commands or JXL hash goal.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import tempfile
import zlib

from benchmark_fast_context import add_ratios, load_cases, summarize, validate_measurement
from benchmark_ma_models import trial_order
from benchmark_palette_runs import prepare_png
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process, sha256,
)

OFFSET_MODES = ("mean", "median", "entropy", "distribution")
VARIANTS = ("baseline", "frozen", "entropy", "leaves", "combined",
            "offset-mean", "offset-median", "offset-entropy", "offset-distribution", "context-map")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--frozen", type=Path)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true",
                   help="Also check generated 8/16-bit gray, gray-alpha, RGB and RGBA fixtures")
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=["baseline", "entropy", "leaves", "combined"])
    p.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--lz", type=int, choices=(0, 1, 2), default=0,
                   help="Use the same native LZ77 policy for baseline and every trial")
    p.add_argument("--buffering", type=int, choices=(-1, 0, 1, 2))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--leaf-offset-samples", type=int, default=1024,
                   help="Per-leaf fitting reservoir only; not an MA or input population cap")
    p.add_argument("--context-map-shortlist", type=int, default=8)
    p.add_argument("--context-map-merges", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or args.leaf_offset_samples < 1
            or args.context_map_shortlist < 1 or args.context_map_merges < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error("Invalid threads/repeats/warmups/timeout/leaf-offset-samples/context-map limits")
    if "baseline" not in args.variants or len(set(args.variants)) != len(args.variants):
        p.error("Use distinct variants including baseline")
    if "frozen" in args.variants and args.frozen is None:
        p.error("frozen variant requires --frozen")
    if not args.case and args.manifest is None and not args.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return args


def fixtures(folder):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    paths = []
    width, height = 67, 51
    for depth in (8, 16):
        maximum = (1 << depth) - 1
        for color, channels in ((0, 1), (4, 2), (2, 3), (6, 4)):
            raw = bytearray()
            for y in range(height):
                values = []
                for x in range(width):
                    for c in range(channels):
                        value = (x * 71 + y * 137 + x * y * (c + 1) * 29 + c * 101) & maximum
                        if channels in (2, 4) and c == channels - 1:
                            value = (0, maximum // 2, maximum)[(x + y) % 3]
                        values.append(value)
                raw.append(0)
                raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(values), *values))
            header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0)
            data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                    + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
            path = folder / f"synthetic-{depth}bit-c{channels}.png"
            path.write_bytes(data)
            paths.append(path)
    return paths


def command(args, source, output, variant):
    encoder = args.frozen if variant == "frozen" else args.encoder
    result = [str(encoder), str(source), "--effort", str(args.effort),
              "--threads", str(args.threads), "--model-costs", "--output", str(output)]
    if args.buffering is not None:
        result += ["--buffering", str(args.buffering)]
    if args.lz:
        result += ["--lz", str(args.lz)]
    if variant in ("entropy", "combined"):
        result += ["--refine-entropy"]
    if variant in ("leaves", "combined"):
        result += ["--refine-leaves"]
    if variant == "context-map":
        result += ["--refine-context-map", "--context-map-shortlist", str(args.context_map_shortlist),
                   "--context-map-merges", str(args.context_map_merges)]
    if variant in tuple(f"offset-{mode}" for mode in OFFSET_MODES):
        result += ["--leaf-offset", variant.removeprefix("offset-"),
                   "--leaf-offset-samples", str(args.leaf_offset_samples)]
    return result


def validate_receipt(measurement, variant):
    if variant == "frozen":
        return
    if "refinement" not in measurement:
        raise ValueError("Missing refinement receipt")
    receipt = measurement["refinement"]
    if variant == "baseline":
        if receipt is not None:
            raise ValueError("Baseline unexpectedly used refinement")
        return
    if (not isinstance(receipt, dict)
            or receipt.get("entropy") is not (variant in ("entropy", "combined"))
            or receipt.get("leaves") is not (variant in ("leaves", "combined"))):
        raise ValueError("Requested refinement flags were not executed")
    offset_mode = variant.removeprefix("offset-") if variant.startswith("offset-") else None
    if "offset_mode" not in receipt or receipt["offset_mode"] != offset_mode:
        raise ValueError("Requested offset mode was not executed")
    for key in ("domains", "models", "refit_proposals", "refit_wins", "entropy_wins", "saved_bits"):
        if type(receipt.get(key)) is not int or receipt[key] < 0:
            raise ValueError(f"Invalid refinement counter {key}")
    if (receipt["domains"] < 1 or receipt["models"] < receipt["domains"]
            or receipt["refit_wins"] > receipt["refit_proposals"]
            or receipt["refit_proposals"] > receipt["domains"]):
        raise ValueError("Invalid refinement work counts")
    if not receipt["leaves"] and receipt["refit_proposals"]:
        raise ValueError("Unrequested leaf refitting")
    if not receipt["entropy"] and receipt["entropy_wins"]:
        raise ValueError("Unrequested entropy comparison")
    offset_counters = ("offset_proposals", "offset_wins", "offset_leaves", "offset_samples",
                       "offset_population", "offset_candidates")
    for key in offset_counters:
        if type(receipt.get(key)) is not int or receipt[key] < 0:
            raise ValueError(f"Invalid refinement counter {key}")
    if (receipt["offset_wins"] > receipt["offset_proposals"]
            or receipt["offset_proposals"] > receipt["domains"]
            or receipt["offset_samples"] > receipt["offset_population"]):
        raise ValueError("Invalid offset work counts")
    if offset_mode is None and any(receipt[key] for key in offset_counters):
        raise ValueError("Unrequested offset fitting")
    for key in ("refit_seconds", "model_seconds", "write_seconds", "offset_seconds"):
        value = receipt.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid refinement timer {key}")
    if offset_mode is None and receipt["offset_seconds"] != 0:
        raise ValueError("Unrequested offset fitting timer")
    context_map_fields = ("context_map", "context_map_shortlist", "context_map_merge_limit",
                          "context_map_proposals", "context_map_wins", "context_map_proposed_merges",
                          "context_map_selected_merges", "context_map_pairs", "context_map_map_trials",
                          "context_map_skipped", "context_map_seconds")
    # Older encoders do not include these fields for the existing variants.
    if variant == "context-map" or any(key in receipt for key in context_map_fields):
        if receipt.get("context_map") is not (variant == "context-map"):
            raise ValueError("Requested context-map refinement was not executed")
        for key in context_map_fields[1:-1]:
            minimum = 1 if key in ("context_map_shortlist", "context_map_merge_limit") else 0
            if type(receipt.get(key)) is not int or receipt[key] < minimum:
                raise ValueError(f"Invalid refinement counter {key}")
        if (receipt["context_map_wins"] > receipt["context_map_proposals"]
                or receipt["context_map_selected_merges"] > receipt["context_map_proposed_merges"]):
            raise ValueError("Invalid context-map work counts")
        seconds = receipt.get("context_map_seconds")
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Invalid refinement timer context_map_seconds")
        if not receipt["context_map"] and any(receipt[key] for key in context_map_fields[3:]):
            raise ValueError("Unrequested context-map refinement")


def validate_sizes(variants):
    baseline = {sample["repetition"]: sample["measurement"]["bytes"]
                for sample in variants["baseline"]}
    for variant, samples in variants.items():
        if variant in ("baseline", "frozen"):
            continue
        for sample in samples:
            if sample["measurement"]["bytes"] > baseline[sample["repetition"]]:
                raise ValueError(f"Actual output size regressed: {variant}, "
                                 f"repeat={sample['repetition']}")


def decoder_summary(samples):
    times = [sample["decoder"]["wall_seconds"] for sample in samples
             if not sample["warmup"] and sample["verified"]]
    if not times or any(isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError("Invalid decoder wall timing")
    return dict(decoder_wall_seconds=statistics.median(times),
                min_decoder_wall_seconds=min(times), max_decoder_wall_seconds=max(times))


def benchmark(args):
    report = dict(status="running", config=vars(args).copy(), cases=[], errors=[],
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  timing="Native encode only; serial order-rotated trials, median after warmups",
                  decoder_timing="Serial djxl process wall time includes startup and input/output; warmups excluded",
                  correctness="Original samples including hidden RGB, decoded by official djxl",
                  scope="Fixed native candidate at effort; not the outer scheduler",
                  timeout="Process guard only; no encoder timeout policy")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        try:
            env, report["environment"] = clean_environment(os.environ)
            identities = []
            for key in ("encoder", "decoder", "frozen"):
                path = getattr(args, key)
                if path is not None:
                    path = path.resolve(strict=True)
                    setattr(args, key, path)
                    digest = file_sha256(path)
                    identities.append((path, digest))
                    report[key] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                digest = file_sha256(args.manifest)
                identities.append((args.manifest, digest))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=digest)
            with tempfile.TemporaryDirectory(prefix="refinement-", dir=args.out.parent) as scratch:
                cases = load_cases(args) if args.case or args.manifest else []
                real_sources = set(cases)
                if args.synthetic:
                    cases += fixtures(Path(scratch))
                for index, source in enumerate(cases):
                    original = source.read_bytes()
                    digest = sha256(original)
                    if source in real_sources:
                        identities.append((source, digest))
                    payload, expected, preparation = prepare_png(source, original, Path(scratch))
                    pixels = expected.width * expected.height
                    snapshot = Path(scratch) / "input.png"
                    snapshot.write_bytes(payload)
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=preparation,
                                variants={v: [] for v in args.variants})
                    report["cases"].append(case)
                    for repeat in range(-args.warmups, args.repeats):
                        for variant in trial_order(args.variants, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][variant].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=scratch) as trial:
                                jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                sample["encoder"] = run_process(command(args, snapshot, jxl, variant), env, args.timeout)
                                check_process(sample["encoder"])
                                sample["decoder"] = run_process(decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                check_process(sample["decoder"])
                                compare_pixels(expected, png_samples(png.read_bytes()))
                                measurement = json.loads(sample["encoder"]["stdout"])
                                validate_measurement(measurement, "baseline", "auto", jxl.stat().st_size, pixels)
                                validate_receipt(measurement, variant)
                                sample.update(verified=True, measurement=measurement)
                            print(f"{source.name} {variant} repeat={repeat}: "
                                  f"{measurement['bytes']} bytes {measurement['seconds']:.6f}s verified", flush=True)
                    case["summaries"] = {v: summarize(samples, pixels) for v, samples in case["variants"].items()}
                    add_ratios(case["summaries"])
                    for variant, samples in case["variants"].items():
                        case["summaries"][variant].update(decoder_summary(samples))
                    for summary in case["summaries"].values():
                        summary["ratios_to_baseline"]["decoder_wall_seconds"] = (
                            summary["decoder_wall_seconds"]
                            / case["summaries"]["baseline"]["decoder_wall_seconds"])
                    validate_sizes(case["variants"])
                    report["completed"] = len(report["cases"])
                    output.seek(0)
                    json.dump(report, output, indent=2, default=str, allow_nan=False)
                    output.truncate()
                    output.flush()
            for path, digest in identities:
                if file_sha256(path) != digest:
                    raise ValueError(f"Input or binary changed during benchmark: {path}")
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["status"] = "failed"
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            output.seek(0)
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
            output.truncate()
    return report


def main(argv=None):
    report = benchmark(arguments(argv))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
