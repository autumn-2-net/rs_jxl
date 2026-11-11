"""Five-way Chain Batch ablation using a prebuilt encoder and official djxl.

Same fixed family/e9/threads in every mode. Mandatory separate profiles prove
activation before serial, rotated native-call timings. No builds or discovery.
Every artifact is decoded to exact original samples, including invisible RGB;
codestream hashes are never a correctness criterion. All regressions are kept.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import benchmark_chain_interleave as shared


VARIANTS = ("ordinary", "legacy", "scheduling-only", "adaptive-only", "combined")
NOTE_NAMES = (*shared.NOTE_NAMES, "lz77.source.chain_schedule")
counter = shared.counter


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True, help="Prebuilt encoder_lab; never built here")
    parser.add_argument("--decoder", "--official-djxl", dest="decoder", type=Path, required=True,
                        help="Caller-attested official djxl; binary identity recorded")
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--effort", type=int, choices=(9,), default=9)
    parser.add_argument("--family", choices=("automatic", "literal", "cross6"), default="automatic")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--warmups", type=int, choices=range(4), default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--profile", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.case and args.manifest is None:
        parser.error("Provide --case or --manifest")
    if not 1 <= args.threads <= 256 or not shared.positive_number(args.timeout):
        parser.error("Invalid threads/timeout")
    if args.out.exists():
        parser.error("Report already exists; it will not be overwritten")
    args.variants = list(VARIANTS)
    return args


def configuration(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation: {variant}")
    return dict(batch=variant != "ordinary", lanes=1,
                streams=variant in ("scheduling-only", "combined"),
                adaptive=variant in ("adaptive-only", "combined"))


def command(args, source, output, variant):
    config = configuration(variant)
    result = [str(args.encoder), str(source), "--effort", str(args.effort),
              "--threads", str(args.threads), "--family", args.family,
              "--output", str(output)]
    if config["batch"]:
        result += ["--chain-batch", "--chain-batch-lanes", "1"]
        if not config["streams"]:
            result.append("--chain-batch-legacy-schedule")
        if not config["adaptive"]:
            result.append("--chain-batch-fixed-window")
    return result


def profile_notes(stderr):
    return shared.profile_notes(stderr, NOTE_NAMES)


def validate_route(notes, variant):
    config = configuration(variant)
    routes, batches, work, schedules = (notes[name] for name in NOTE_NAMES)
    chains = [r for r in routes if r.get("method") == "Optc256"
              and r.get("mode") == "SymbolId" and r.get("indexed") == "false"
              and r.get("bucket") == "false" and counter(r, "tokens") > 0]
    if not chains:
        raise ValueError("No ordinary Optc256 SymbolId chain route observed")
    result = dict(verified=True, **config, batch_streams=0, roots=0, tokens=0, computed=0,
                  consumed=0, overflow_requeries=0, demand_queries=0, unused=0,
                  interleaved_positions=0, cooperative_schedules=0)
    if not config["batch"]:
        if batches or work or any(r.get("chain_batch") != "false" for r in chains):
            raise ValueError("Ordinary route unexpectedly used chain batch")
        if schedules:
            raise ValueError("Ordinary route unexpectedly emitted chain batch scheduling")
        return result
    active = [r for r in chains if r.get("chain_batch") == "true"]
    if not active or not batches or not work:
        raise ValueError("Requested chain batch did no observed work; choose an eligible original")
    signatures = [Counter(counter(r, "tokens") for r in rows)
                  for rows in (active, batches, work)]
    if signatures[0] != signatures[1] or signatures[1] != signatures[2]:
        raise ValueError("Incomplete/disagreeing chain batch route/work receipts")
    if not schedules:
        raise ValueError("Missing chain schedule receipt")
    for row in schedules:
        if row.get("streams_enabled") != str(config["streams"]).lower():
            raise ValueError("Observed schedule differs from requested ablation")
        roots, streams, eligible = (counter(row, k) for k in ("roots", "streams", "eligible"))
        if roots == 0 or streams == 0 or not 0 < eligible <= streams or roots > streams:
            raise ValueError("Invalid scheduling roots/streams/eligible counters")
        if row.get("cooperative") not in ("true", "false"):
            raise ValueError("Missing/invalid cooperative schedule flag")
        if row["cooperative"] == "true" and not config["streams"]:
            raise ValueError("Legacy schedule unexpectedly used cooperative roots")
        if roots > 1 and row["cooperative"] != "true":
            raise ValueError("Multiple chain roots without cooperative dispatch")
        result["cooperative_schedules"] += row["cooperative"] == "true"
        result["roots"] += roots
    for row in batches + work:
        if counter(row, "lanes") != 1:
            raise ValueError("Ablation requires observed lanes=1")
        if row.get("adaptive") != str(config["adaptive"]).lower():
            raise ValueError("Observed adaptive configuration differs from requested ablation")
    for row in work:
        tokens, computed, consumed, overflow, positions = (
            counter(row, k) for k in ("tokens", "computed", "consumed",
                                      "overflow_requeries", "interleaved_positions"))
        if computed > tokens or not 0 < consumed <= tokens or overflow > consumed or positions:
            raise ValueError("Inconsistent chain batch work counters")
        demand, unused = counter(row, "demand_queries"), counter(row, "unused")
        if (demand > consumed or unused > computed
                or computed + demand != consumed + unused
                or overflow > consumed - demand):
            raise ValueError("Inconsistent speculative/demand/unused accounting")
        result["demand_queries"] += demand
        result["unused"] += unused
        for key, value in (("tokens", tokens), ("computed", computed), ("consumed", consumed),
                           ("overflow_requeries", overflow)):
            result[key] += value
    result["batch_streams"] = len(batches)
    return result


def benchmark(args):
    return shared.benchmark(args, make_command=command, read_notes=profile_notes,
                            check_route=validate_route, baseline="ordinary", comparison="legacy")


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
