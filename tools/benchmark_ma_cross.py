"""Compare ordinary native e9 MA learning with up to two previous channels.

Reuse the private offsets/refinement runner, original-pixel official djxl
verification and serial rotated trials. No builds or automatic timing runs.
RCT, learning fraction, palette and buffering retain the ordinary defaults.
"""

import sys

import benchmark_fast_offsets as offsets
import benchmark_refinement as refinement


VARIANTS = ("baseline", "ma-prev1", "ma-prev2")


def arguments(argv=None):
    args = offsets.arguments(argv)
    args.variants = list(VARIANTS)
    return args


def previous_channels(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return VARIANTS.index(variant)


def command(args, source, output, variant):
    requested = previous_channels(variant)
    result = refinement.command(args, source, output, "baseline")
    if requested:
        result += ["--previous-channels", str(requested)]
    return result


validate_measurement = offsets.validate_measurement


def validate_receipt(measurement, variant):
    requested = previous_channels(variant)
    offsets.validate_receipt(measurement, "baseline")
    for key in ("max_previous_channels", "ma_previous_channels"):
        if type(measurement.get(key)) is not int:
            raise ValueError(f"Missing/invalid {key} receipt")
    if measurement["max_previous_channels"] != requested:
        raise ValueError("Requested previous-channel limit was not executed")
    # Projection may remove unavailable references, notably for grayscale.
    if not 0 <= measurement["ma_previous_channels"] <= requested:
        raise ValueError("Invalid projected MA previous-channel count")
    for key in ("palette_grouping", "match_extension", "lz_scout", "lz_quick"):
        if measurement.get(key) is not None:
            raise ValueError(f"Unrequested experiment: {key}")
    if measurement.get("tree_learning", "legacy") != "legacy":
        raise ValueError("Unrequested tree-learning experiment")


def validate_sizes(variants):
    # A real ordinary-MA candidate may grow; require matched trials, not wins.
    if set(variants) != set(VARIANTS):
        raise ValueError("Missing MA cross variants")
    reference = [sample["repetition"] for sample in variants["baseline"]]
    if not reference or len(set(reference)) != len(reference):
        raise ValueError("Missing or duplicate paired repetitions")
    for samples in variants.values():
        repetitions = [sample["repetition"] for sample in samples]
        if len(repetitions) != len(reference) or set(repetitions) != set(reference):
            raise ValueError("Missing or duplicate paired repetitions")


def make_runner():
    runner = offsets.make_runner()
    runner.command = command
    runner.validate_measurement = validate_measurement
    runner.validate_receipt = validate_receipt
    runner.validate_sizes = validate_sizes
    runner.add_ratios = offsets.fast.add_ratios
    return runner


def benchmark(args):
    return make_runner().benchmark(args)


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
