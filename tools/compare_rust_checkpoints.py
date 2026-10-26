"""Alternating, pixel-verified measurements of two already-built Rust CLIs."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from compare_cpp_port import (
    MODES, binary_info, clean_environment, measure, png_payload, png_samples,
)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, action="append", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Optional positive MA sample cap; omitted uses each CLI's default")
    parser.add_argument("--mode", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 256 or not 1 <= args.effort <= 10
            or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or (args.sample_limit is not None and args.sample_limit < 1)):
        parser.error("invalid effort, worker, repeat, sample or timeout limit")
    if args.json.exists():
        parser.error("refusing to overwrite an existing report")
    if not args.json.parent.is_dir():
        parser.error("report parent directory must already exist")
    paths = {name: getattr(args, name).resolve(strict=True) for name in ("before", "after")}
    decoders = [path.resolve(strict=True) for path in args.decoder]
    environment, env_record = clean_environment(os.environ)
    report = dict(status="running", rows=[], summaries=[], environment=env_record,
                  threads=args.threads, effort=args.effort, warmups=args.warmups,
                  repeats=args.repeats, sample_limit=args.sample_limit,
                  timing="serial CLI wall including startup/IO, excluding validation",
                  binaries={name: binary_info(path, environment, args.timeout)
                            for name, path in paths.items()},
                  decoders=[binary_info(path, environment, args.timeout) for path in decoders])
    try:
        with tempfile.TemporaryDirectory(prefix="rsjxl-checkpoints-") as temp:
            root = Path(temp)
            for image in args.images:
                data = image.read_bytes()
                payload, _ = png_payload(data)
                expected = png_samples(data)
                source = root / "input.png"
                source.write_bytes(payload)
                for mode in args.mode:
                    rows = []
                    for repeat in range(-args.warmups, args.repeats):
                        order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                        for name in order:
                            with tempfile.TemporaryDirectory(dir=root, prefix="trial-") as trial:
                                row = measure("rust", paths[name], decoders, source, expected,
                                              Path(trial), args.threads, args.effort, mode,
                                              args.sample_limit, environment, environment, args.timeout)
                            row.update(input=str(image), mode=mode, version=name,
                                       repeat=repeat, warmup=repeat < 0)
                            report["rows"].append(row)
                            if row["status"] != "passed":
                                raise RuntimeError(row.get("error", "pixel validation failed"))
                            if repeat >= 0:
                                rows.append(row)
                            print(f"{image.name} {mode} {name} round={repeat}: "
                                  f"{row['bytes']} B {row['bpp']:.6f} bpp "
                                  f"{row['wall_seconds']:.3f}s {row['mpix_per_second']:.3f} MPix/s", flush=True)
                    summary = dict(input=str(image), mode=mode)
                    for name in paths:
                        selected = [row for row in rows if row["version"] == name]
                        seconds = statistics.median(row["wall_seconds"] for row in selected)
                        summary[name] = dict(
                            bytes=statistics.median(row["bytes"] for row in selected),
                            bpp=statistics.median(row["bpp"] for row in selected),
                            seconds=seconds, mpix_per_second=selected[0]["mpix"] / seconds,
                        )
                    summary["speedup"] = summary["before"]["seconds"] / summary["after"]["seconds"]
                    report["summaries"].append(summary)
                    print("MEDIAN " + json.dumps(summary), flush=True)
        report["status"] = "passed"
    except (OSError, ValueError, RuntimeError) as error:
        report.update(status="failed", error=str(error))
    finally:
        with args.json.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    if report["status"] != "passed":
        raise SystemExit(report["error"])


if __name__ == "__main__":
    main()
