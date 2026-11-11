"""Compare frozen shallow MA plans with one Distribution offset proposal.

Reuses the refinement runner's metadata-neutral PNG preparation, original-pixel
djxl oracle, native timings, rotated serial trials and failure reports. No builds.
Decoder provenance is caller-attested; executable/input identities are recorded.
"""

import argparse
import importlib.util
import math
from pathlib import Path
import sys

import benchmark_fast_context as fast
import benchmark_refinement as refinement


VARIANTS = ("baseline", "fast-basic-off", "fast-basic-off-distribution",
            "fast-full-static", "fast-full-static-distribution")
PAIRS = {"fast-basic-off-distribution": "fast-basic-off",
         "fast-full-static-distribution": "fast-full-static"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--leaf-offset-samples", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(variants=list(VARIANTS), frozen=None, warmups=0,
                        buffering=None, lz=0)
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.repeats < 1
            or args.leaf_offset_samples < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid threads/repeats/samples/timeout")
    if not args.case and args.manifest is None and not args.synthetic:
        parser.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        parser.error("Report already exists; it will not be overwritten")
    return args


def parameters(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == "baseline":
        return None
    full = variant.startswith("fast-full-")
    return dict(mode="auto12", bias="static" if full else "off", scan="auto",
                predictors="full" if full else "basic")


def command(args, source, output, variant):
    result = refinement.command(args, source, output, "baseline")
    plan = parameters(variant)
    if plan is not None:
        result += ["--fast-context", plan["mode"], "--context-bias", plan["bias"],
                   "--context-scan", plan["scan"], "--context-predictors", plan["predictors"]]
    if variant in PAIRS:
        result += ["--leaf-offset", "distribution", "--leaf-offset-samples",
                   str(args.leaf_offset_samples)]
    return result


def validate_measurement(measurement, variant, scan, encoded_size, pixels):
    # The shared runner passes baseline here; requested-plan validation follows
    # in validate_receipt, where the actual variant is available.
    if not isinstance(measurement, dict):
        raise ValueError("Encoder JSON must be an object")
    fast.validate_measurement(dict(measurement, fast_context=None),
                              "baseline", scan, encoded_size, pixels)


def validate_receipt(measurement, variant):
    plan = parameters(variant)
    # Reuse fast-context's full counter checks as well as its flag validation.
    fast_variant = "baseline" if plan is None else (
        "auto12-static" if plan["bias"] == "static" else "auto12")
    pixels = measurement["bytes"] * 8 / measurement["bpp"]
    fast.validate_measurement(measurement, fast_variant, "auto", measurement["bytes"], pixels)
    if plan is not None and measurement["fast_context"].get("predictors") != plan["predictors"]:
        raise ValueError("Requested predictor set was not executed")
    if measurement.get("ma_seed") is not None:
        raise ValueError("Unrequested MA warm start")
    refinement.validate_receipt(measurement, "offset-distribution" if variant in PAIRS else "baseline")


def validate_sizes(variants):
    for refined, baseline in PAIRS.items():
        reference = {s["repetition"]: s["measurement"]["bytes"] for s in variants[baseline]}
        samples = variants[refined]
        if len(reference) != len(variants[baseline]) or len(samples) != len(reference):
            raise ValueError("Missing or duplicate paired repetitions")
        if {s["repetition"] for s in samples} != set(reference) or len(
                {s["repetition"] for s in samples}) != len(samples):
            raise ValueError("Missing or duplicate paired repetitions")
        for sample in samples:
            if sample["measurement"]["bytes"] > reference[sample["repetition"]]:
                raise ValueError(f"Actual output size regressed: {refined} vs {baseline}, "
                                 f"repeat={sample['repetition']}")


def add_ratios(summaries):
    fast.add_ratios(summaries)
    for refined, baseline in PAIRS.items():
        summaries[refined]["paired_baseline"] = baseline
        summaries[refined]["ratios_to_fast_baseline"] = {
            key: summaries[refined][key] / summaries[baseline][key]
            for key in ("bytes", "seconds", "bpp", "mpix_per_second")}


def make_runner():
    # A private module instance keeps these narrow hooks out of the shared
    # runner's globals, including when multiple harnesses are imported by tests.
    spec = importlib.util.spec_from_file_location("_fast_offsets_runner", refinement.__file__)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.command = command
    runner.validate_measurement = validate_measurement
    runner.validate_receipt = validate_receipt
    runner.validate_sizes = validate_sizes
    runner.add_ratios = add_ratios
    return runner


def benchmark(args):
    return make_runner().benchmark(args)


def main(argv=None):
    report = benchmark(arguments(argv))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
