"""Small before/after comparison of prebuilt encoder_lab checkpoints.

Both binaries receive the same e9/family/threads/--chain-batch-parallel flags.
Separate mandatory profiles prove actual eligible Batch work (or an explicitly
selected inactive/either control). Shared serial alternating trials retain all
regressions and check exact original pixels with caller-attested official djxl.
No builds, corpus discovery or JXL-byte identity requirement.
"""

import argparse
from pathlib import Path
import sys

import benchmark_chain_interleave as shared
import benchmark_chain_parallel as parallel


VARIANTS = ("before", "after")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in VARIANTS:
        parser.add_argument(f"--{name}", type=Path, required=True,
                            help="Prebuilt encoder_lab; never built here")
    parser.add_argument("--official-djxl", dest="decoder", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--family", choices=("automatic", "literal", "cross6"), default="automatic")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expect", choices=parallel.EXPECTATIONS, default="active",
                        help="Observed activity required in BOTH checkpoints; default active")
    parser.add_argument("--warmups", type=int, choices=range(4), default=0)
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--profile", action="store_true", required=True,
                        help="Mandatory separate route proof, excluded from timings")
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
    args.encoder, args.effort, args.variants = args.before, 9, list(VARIANTS)
    return args


def command(args, source, output, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown checkpoint: {variant}")
    selected = argparse.Namespace(**vars(args))
    selected.encoder = getattr(args, variant)
    return parallel.command(selected, source, output, "parallel")


def benchmark(args):
    identities = {}

    def checkpoint_command(options, source, output, variant):
        # Run inside the shared failure/report boundary, before either binary.
        if not identities:
            for name in VARIANTS:
                path = getattr(options, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(options, name, path)
                identities[name] = dict(path=str(path), sha256=shared.port.file_sha256(path))
        return command(options, source, output, variant)

    def check_route(notes, variant):
        return parallel.validate_route(notes, "parallel", threads=args.threads, expect=args.expect)

    def finalize_report(report):
        report["scope"] = "Prebuilt before/after; same fixed family/e9/threads and parallel-only Batch flags"
        report.pop("interleaved_positions", None)
        report.update(identities)
        for identity in identities.values():
            identity["verified_unchanged"] = False
            try:
                identity["sha256_after"] = shared.port.file_sha256(Path(identity["path"]))
                if identity["sha256_after"] != identity["sha256"]:
                    raise ValueError(f"Checkpoint binary changed during benchmark: {identity['path']}")
                identity["verified_unchanged"] = True
            except (OSError, ValueError) as error:
                report["errors"].append(f"{type(error).__name__}: {error}")

    return shared.benchmark(args, make_command=checkpoint_command,
                            read_notes=parallel.profile_notes, check_route=check_route,
                            baseline="before", comparison="after", finalize_report=finalize_report)


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
