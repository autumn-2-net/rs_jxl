"""Summarize separate source-patch diagnostic trials, never benchmark timings.

Detail records are summed within each trial, then medianed per case/variant.
Corpus totals are sums of those per-case medians (not pooled image medians).
Missing detail records are errors, including for no-background detector calls.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


DETAIL_PREFIX = "[rs-jxl profile] source.patch.detail:"
STAGES = (
    "seed_scan", "background_setup", "background_flood", "component_scan",
    "fingerprint_delta", "group_lookup", "finalize_delta",
)
TIMING_HEADER = "stage | calls | total seconds | longest seconds"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, where):
    require(type(value) in (int, float), f"{where}: expected a number")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    require(valid, f"{where}: expected a finite nonnegative number")
    return value


def numeric_map(value, where):
    require(isinstance(value, dict) and bool(value), f"{where}: expected nonempty object")
    for key, item in value.items():
        require(isinstance(key, str) and bool(key.strip()), f"{where}: invalid key")
        number(item, f"{where}.{key}")
    return value


def add_values(target, values, where):
    for key, value in values.items():
        target[key] = number(target.get(key, 0) + value, f"{where}.{key} sum")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_trial(stderr, where):
    require(isinstance(stderr, str), f"{where}: missing encoder stderr")
    result = dict(stages_ns={}, counts={}, timing_ns={}, timing_calls={}, detail_records=0)
    keys = None
    in_table = False
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith(DETAIL_PREFIX):
            detail = json.loads(line[len(DETAIL_PREFIX):], object_pairs_hook=unique_object)
            require(isinstance(detail, dict), f"{where}: detail must be an object")
            stages = numeric_map(detail.get("stages_ns"), f"{where}.stages_ns")
            counts = numeric_map(detail.get("counts"), f"{where}.counts")
            require(set(STAGES) <= stages.keys(), f"{where}: missing required detail stages")
            current = (set(stages), set(counts))
            require(keys is None or keys == current, f"{where}: inconsistent stage/count keys")
            keys = current
            add_values(result["stages_ns"], stages, where)
            add_values(result["counts"], counts, where)
            result["detail_records"] += 1
        elif line == TIMING_HEADER:
            in_table = True
        elif in_table and "|" in line:
            fields = [part.strip() for part in line.split("|")]
            require(len(fields) == 4 and bool(fields[0]), f"{where}: malformed timing row")
            stage, calls, total, longest = fields
            require(calls.isascii() and calls.isdigit(), f"{where}: invalid timing calls")
            calls = int(calls)
            total = number(float(total), f"{where}.{stage} total seconds")
            number(float(longest), f"{where}.{stage} longest seconds")
            add_values(result["timing_ns"], {stage: total * 1e9}, where)
            add_values(result["timing_calls"], {stage: calls}, where)
        elif in_table and line:
            in_table = False
    require(result["detail_records"] > 0, f"{where}: missing source.patch.detail records")
    return result


def check_sample(sample, where, diagnostic=False):
    require(isinstance(sample, dict), f"{where}: expected trial object")
    require(sample.get("verified") is True and not sample.get("error"),
            f"{where}: unverified or failed trial")
    if diagnostic:
        require(sample.get("profile") is True and sample.get("included_in_timing") is False,
                f"{where}: not a separate diagnostic trial")
    for name in ("encoder", "decoder"):
        process = sample.get(name)
        require(isinstance(process, dict) and type(process.get("returncode")) is int
                and process["returncode"] == 0 and not process.get("error"),
                f"{where}: failed or missing {name} process")


def medians(trials, where):
    result = {}
    for field in ("stages_ns", "counts", "timing_ns", "timing_calls"):
        keys = trials[0][field].keys()
        require(all(trial[field].keys() == keys for trial in trials),
                f"{where}: inconsistent {field} keys across trials")
        result[field] = {key: number(statistics.median([t[field][key] for t in trials]),
                                     f"{where}.{field}.{key} median") for key in keys}
    result["detail_records"] = statistics.median(t["detail_records"] for t in trials)
    return result


def summarize(report):
    require(isinstance(report, dict), "Expected report object")
    require(report.get("schema_version") == 1, "Unsupported report schema_version")
    require(report.get("status") == "passed" and report.get("errors") == [],
            "Report failed or contains errors")
    profiling = report.get("profiling", {})
    require(isinstance(profiling, dict) and profiling.get("enabled") is True
            and profiling.get("included_in_timing") is False, "Separate profiling is not enabled")
    repeats = profiling.get("runs_per_case_version")
    if repeats is not None:
        require(type(repeats) is int and repeats > 0, "Invalid diagnostic repetition count")
    encoders = report.get("encoders")
    require(isinstance(encoders, dict) and bool(encoders), "Missing encoder labels")
    labels = list(encoders)
    cases = report.get("cases")
    require(isinstance(cases, list) and bool(cases), "Empty or missing corpus")
    output = dict(schema_version=1,
                  aggregation="sum records per trial; median trials per case; sum per-case medians",
                  timing_note="Inclusive diagnostic process totals; nested/parallel times overlap",
                  cases=[], corpus={})
    detail_keys = None
    for index, case in enumerate(cases):
        where = f"case {index}"
        require(isinstance(case, dict), f"{where}: expected object")
        source = case.get("source")
        require(isinstance(source, str) and bool(source), f"{where}: missing source")
        diagnostics = case.get("diagnostics")
        require(isinstance(diagnostics, dict) and diagnostics.keys() == encoders.keys(),
                f"{where}: missing or extra diagnostic labels")
        # Do not trust a passed status if an underlying native trial failed.
        if "variants" in case:
            variants = case["variants"]
            require(isinstance(variants, dict) and variants.keys() == encoders.keys(),
                    f"{where}: invalid native variant labels")
            for label, samples in variants.items():
                require(isinstance(samples, list) and bool(samples), f"{where}: empty native trials")
                for sample in samples:
                    check_sample(sample, f"{where}/{label} native")
        item = dict(source=source, variants={})
        for label in labels:
            context = f"{where}/{label}"
            samples = diagnostics[label]
            require(isinstance(samples, list) and bool(samples), f"{context}: empty diagnostics")
            require(repeats is None or len(samples) == repeats, f"{context}: missing diagnostic trials")
            trials = []
            for trial_index, sample in enumerate(samples):
                location = f"{context}/trial {trial_index}"
                check_sample(sample, location, diagnostic=True)
                trial = parse_trial(sample["encoder"].get("stderr"), location)
                current = (set(trial["stages_ns"]), set(trial["counts"]))
                require(detail_keys is None or detail_keys == current,
                        f"{location}: inconsistent stage/count keys across corpus")
                detail_keys = current
                trials.append(trial)
            median = medians(trials, context)
            item["variants"][label] = dict(trials=trials, median=median)
        output["cases"].append(item)
    for label in labels:
        totals = dict(stages_ns={}, counts={}, timing_ns={}, timing_calls={}, detail_records=0)
        for case in output["cases"]:
            median = case["variants"][label]["median"]
            for field in ("stages_ns", "counts", "timing_ns", "timing_calls"):
                # Timing scopes not executed in a case contribute zero, never missing detail.
                add_values(totals[field], median[field], label)
            totals["detail_records"] += median["detail_records"]
        output["corpus"][label] = totals
    return output


def print_table(summary):
    for label, totals in summary["corpus"].items():
        stages = list(totals["stages_ns"])
        print(f"\n{label}: detail stages (ms); corpus = sum of per-case medians")
        print("case | " + " | ".join(stages))
        rows = [("CORPUS", totals)] + [
            (case["source"], case["variants"][label]["median"]) for case in summary["cases"]]
        for name, values in rows:
            print(name + " | " + " | ".join(f"{values['stages_ns'][stage] / 1e6:.3f}"
                                            for stage in stages))
        scopes = [stage for stage in totals["timing_ns"]
                  if stage.startswith(("source.patch", "patch.", "encode.patch"))]
        if scopes:
            print("Inclusive patch scopes (ms; overlapping, not additive)")
            print("case | " + " | ".join(scopes))
            for name, values in rows:
                print(name + " | " + " | ".join(
                    f"{values['timing_ns'].get(stage, 0) / 1e6:.3f}" for stage in scopes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, help="New summary JSON; never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.out is not None and args.out.exists():
            raise ValueError("Output already exists; it will not be overwritten")
        with args.report.open(encoding="utf-8") as source:
            summary = summarize(json.load(source, object_pairs_hook=unique_object))
        if args.out is not None:
            with args.out.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(summary, output, indent=2, allow_nan=False)
                output.write("\n")
        print_table(summary)
    except (OSError, ValueError, OverflowError) as error:
        print(f"Profile summary failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
