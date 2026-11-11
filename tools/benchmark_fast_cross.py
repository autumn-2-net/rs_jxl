"""Compare shallow auto12 plans with optional previous-channel residual splits.

Reuse the offsets/refinement harness for original 8/16-bit samples (including
hidden RGB), official djxl verification, actual bytes and native encode timing.
No builds. Decoder provenance is caller-attested. Proxy selection may grow files.
"""

import sys

import benchmark_fast_offsets as offsets


VARIANTS = ("baseline", "fast-basic-off", "fast-basic-cross",
            "fast-full-static", "fast-full-cross")
PAIRS = {"fast-basic-cross": "fast-basic-off",
         "fast-full-cross": "fast-full-static"}


def arguments(argv=None):
    args = offsets.arguments(argv)
    args.variants = list(VARIANTS)
    return args


def base_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return PAIRS.get(variant, variant)


def parameters(variant):
    plan = offsets.parameters(base_variant(variant))
    return None if plan is None else dict(plan, cross_channel=variant in PAIRS)


def command(args, source, output, variant):
    result = offsets.command(args, source, output, base_variant(variant))
    if variant in PAIRS:
        result += ["--context-cross-channel"]
    return result


validate_measurement = offsets.validate_measurement


def validate_receipt(measurement, variant):
    offsets.validate_receipt(measurement, base_variant(variant))
    plan = parameters(variant)
    if plan is None:
        return
    receipt = measurement["fast_context"]
    if receipt.get("cross_channel") is not plan["cross_channel"]:
        raise ValueError("Requested cross_channel flag was not executed")
    count = receipt.get("cross_channel_plans")
    if type(count) is not int or count < 0:
        raise ValueError("Invalid fast-context counter: cross_channel_plans")
    # Axis counters partition channel plans, not MA trees. Zero cross plans is
    # valid for gray, missing references or a proxy that prefers ordinary axes.
    if not plan["cross_channel"] and count:
        raise ValueError("Unrequested cross_channel plans")


def validate_sizes(variants):
    # Check pairing only: neither shallow plans nor their extra proxy-selected
    # axis promise non-growth against ordinary e9 or the matching fast plan.
    for cross, baseline in PAIRS.items():
        reference = [sample["repetition"] for sample in variants[baseline]]
        trials = [sample["repetition"] for sample in variants[cross]]
        if (len(set(reference)) != len(reference)
                or len(set(trials)) != len(trials) or set(reference) != set(trials)):
            raise ValueError("Missing or duplicate paired repetitions")


def add_ratios(summaries):
    offsets.fast.add_ratios(summaries)
    for cross, baseline in PAIRS.items():
        summaries[cross]["paired_baseline"] = baseline
        summaries[cross]["ratios_to_fast_baseline"] = {
            key: summaries[cross][key] / summaries[baseline][key]
            for key in ("bytes", "seconds", "bpp", "mpix_per_second")}


def make_runner():
    runner = offsets.make_runner()
    runner.command = command
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
