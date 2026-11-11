"""Bounded ordinary vs parallel-only Chain Batch comparison with prebuilt binaries.

Always e9 and the same selected family. Mandatory separate profiles establish
actual activity per case, not a CLI flag; shared serial alternating timings,
official djxl exact-original sample checks, hashes and cleanup remain unchanged.
No builds, corpus discovery, production toggles or performance filtering.

With --encoder lab.exe --official-djxl djxl.exe --profile --out NEW.json:
  --case hot-screenshot.png --threads 8 --expect active
  --case hot-screenshot.png --threads 1 --expect inactive
  --manifest mixed-other.json --threads 8 --expect either
Expectations apply to the parallel profile of EVERY selected case.
"""

import argparse
from pathlib import Path
import sys

import benchmark_chain_batch as batch
import benchmark_chain_interleave as shared


VARIANTS = ("ordinary", "parallel")
EXPECTATIONS = ("active", "inactive", "either")
NOTE_NAMES = batch.NOTE_NAMES
profile_notes = batch.profile_notes
counter = batch.counter


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True,
                        help="Prebuilt development encoder_lab; never built here")
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
    parser.add_argument("--expect", choices=EXPECTATIONS, default="either",
                        help="Required observed parallel activity per case (ordinary is always inactive)")
    parser.add_argument("--profile", action="store_true", required=True,
                        help="Separate mandatory route proof, excluded from timing summaries")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.case and args.manifest is None:
        parser.error("Provide --case or --manifest")
    if not 1 <= args.threads <= 256 or not shared.positive_number(args.timeout):
        parser.error("Invalid threads/timeout")
    if args.threads == 1 and args.expect == "active":
        parser.error("Parallel-only Batch cannot be active at threads=1")
    if args.out.exists():
        parser.error("Report already exists; it will not be overwritten")
    args.variants = list(VARIANTS)
    return args


def command(args, source, output, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown mode: {variant}")
    result = batch.command(args, source, output, "ordinary")
    if variant == "parallel":
        result.append("--chain-batch-parallel")
    return result


def chain_route(row):
    return (row.get("method") == "Optc256" and row.get("mode") == "SymbolId"
            and row.get("indexed") == "false" and row.get("bucket") == "false"
            and counter(row, "tokens") > 0)


def validate_inactive_schedule(row):
    # MT scheduling intent is allowed on bypasses, but is never a work receipt.
    if row.get("streams_enabled") != "true" or row.get("cooperative") not in ("true", "false"):
        raise ValueError("Invalid inactive parallel schedule flags")
    roots, streams, eligible = (counter(row, key) for key in ("roots", "streams", "eligible"))
    if not 0 < roots <= streams or eligible > streams:
        raise ValueError("Invalid inactive parallel schedule counters")
    if roots > 1 and row["cooperative"] != "true":
        raise ValueError("Multiple chain roots without cooperative dispatch")


def validate_route(notes, variant, *, threads=8, expect="either"):
    if variant not in VARIANTS or expect not in EXPECTATIONS or not 1 <= threads <= 256:
        raise ValueError("Invalid mode, expectation or threads")
    routes, batches, work, schedules = (notes[name] for name in NOTE_NAMES)
    if variant == "parallel" and threads == 1 and schedules:
        raise ValueError("Parallel-only mode emitted Batch scheduling at threads=1")
    for row in routes:
        if row.get("chain_batch") not in ("true", "false"):
            raise ValueError("Missing/invalid route chain_batch receipt")
        if row["chain_batch"] == "true" and not chain_route(row):
            raise ValueError("Batch reported on an ineligible source route")
    observed = bool(batches or work or any(row["chain_batch"] == "true" for row in routes))
    if observed:
        if variant == "ordinary":
            raise ValueError("Ordinary mode unexpectedly used Batch")
        if threads == 1:
            raise ValueError("Parallel-only mode used Batch at threads=1")
        # Existing combined validation owns per-stream accounting and completeness.
        result = batch.validate_route(notes, "combined")
        workers = [counter(row, "workers") for row in batches]
        if any(worker <= 1 or worker > threads for worker in workers):
            raise ValueError("Every active Batch must report 1 < workers <= threads")
    else:
        if variant == "ordinary" and schedules:
            raise ValueError("Ordinary mode unexpectedly emitted Batch scheduling")
        for row in schedules:
            validate_inactive_schedule(row)
        # The ordinary validator requires Optc256. Small/ineligible images need
        # not reach it; absence is valid only after ruling out ALL Batch evidence.
        if any(chain_route(row) for row in routes):
            ordinary_notes = dict(notes, **{NOTE_NAMES[3]: []})
            result = batch.validate_route(ordinary_notes, "ordinary")
        else:
            result = dict(verified=True, batch=False, lanes=1, streams=False, adaptive=False,
                          batch_streams=0, roots=0, tokens=0, computed=0, consumed=0,
                          overflow_requeries=0, demand_queries=0, unused=0,
                          interleaved_positions=0, cooperative_schedules=0)
        workers = []
    if variant == "parallel" and ((expect == "active" and not observed)
                                  or (expect == "inactive" and observed)):
        raise ValueError(f"Parallel expectation {expect} disagrees with observed activity={observed}")
    result.update(active=observed, workers=workers, requested_parallel=variant == "parallel",
                  schedule_count=len(schedules),
                  schedule_eligible=sum(counter(row, "eligible") for row in schedules),
                  expectation=expect if variant == "parallel" else "inactive",
                  disposition="active" if observed else ("bypass" if variant == "parallel" else "ordinary"))
    return result


def benchmark(args):
    def check_route(notes, variant):
        return validate_route(notes, variant, threads=args.threads, expect=args.expect)

    return shared.benchmark(args, make_command=command, read_notes=profile_notes,
                            check_route=check_route, baseline="ordinary", comparison="parallel")


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
