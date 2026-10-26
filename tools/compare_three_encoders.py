"""Pixel-verified official cjxl / optimized C++ / Rust CLI comparison."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from compare_cpp_port import (
    binary_info, check_process, clean_environment, compare_pixels,
    decoder_command, encoder_command, measure, png_payload, png_samples,
    run_process,
)


def official_trial(binary, decoders, source, expected, directory, args, environment):
    output = directory / "encoded.jxl"
    # Default upstream e9, without the fork's outer scheduler or explicit -I.
    command = encoder_command("cpp", binary, source, output, args.threads,
                              args.effort, "full-scheduler", args.sample_limit)
    process = run_process(command, environment, args.timeout)
    check_process(process)
    size = output.stat().st_size
    if not size:
        raise ValueError("Official encoder produced empty output")
    pixels = expected.width * expected.height
    row = dict(process=process, bytes=size, bpp=8 * size / pixels,
               wall_seconds=process["wall_seconds"], mpix=pixels / 1e6,
               decodes=[], mode="official-default")
    for index, decoder in enumerate(decoders):
        decoded = directory / f"decoded-{index}.png"
        result = run_process(decoder_command(decoder, output, decoded, expected.depth),
                             environment, args.timeout)
        row["decodes"].append(result)
        check_process(result)
        compare_pixels(expected, png_samples(decoded.read_bytes()))
        result["pixels_match_including_hidden_rgb"] = True
    row["status"] = "passed"
    return row


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "cpp", "rust"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--decoder", type=Path, action="append", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--effort", type=int, default=9)
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Optional positive Rust MA sample cap; omitted uses the CLI default")
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
            or (args.sample_limit is not None and args.sample_limit < 1)
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("invalid measurement limits")
    if args.json.exists() or not args.json.parent.is_dir():
        parser.error("report must be a new file in an existing directory")
    names = ("official", "cpp", "rust")
    paths = {name: getattr(args, name).resolve(strict=True) for name in names}
    decoders = [path.resolve(strict=True) for path in args.decoder]
    environment, env_record = clean_environment(os.environ)
    cpp_environment, cpp_env_record = clean_environment(os.environ, cpp=True)
    report = dict(status="running", rows=[], summaries=[], environment=env_record,
                  cpp_environment=cpp_env_record,
                  threads=args.threads, effort=args.effort, repeats=args.repeats,
                  warmups=args.warmups, sample_limit=args.sample_limit,
                  timing="serial CLI wall including startup/IO, excluding verification",
                  binaries={name: binary_info(path, environment, args.timeout)
                            for name, path in paths.items()},
                  decoders=[binary_info(path, environment, args.timeout) for path in decoders])
    try:
        with tempfile.TemporaryDirectory(prefix="rsjxl-three-way-") as temp:
            root = Path(temp)
            for image in args.images:
                data = image.read_bytes()
                payload, _ = png_payload(data)
                expected = png_samples(data)
                source = root / "input.png"
                source.write_bytes(payload)
                rows = []
                for repeat in range(-args.warmups, args.repeats):
                    start = repeat % len(names)
                    for name in names[start:] + names[:start]:
                        with tempfile.TemporaryDirectory(dir=root, prefix="trial-") as trial:
                            if name == "official":
                                row = official_trial(paths[name], decoders, source, expected,
                                                     Path(trial), args, environment)
                            else:
                                row = measure(name, paths[name], decoders, source, expected,
                                              Path(trial), args.threads, args.effort,
                                              "full-scheduler", args.sample_limit,
                                              cpp_environment if name == "cpp" else environment,
                                              environment, args.timeout)
                                row["mode"] = "full-scheduler"
                        row.update(input=str(image), encoder=name,
                                   repeat=repeat, warmup=repeat < 0)
                        report["rows"].append(row)
                        if row["status"] != "passed":
                            raise RuntimeError(row.get("error", "pixel validation failed"))
                        if repeat >= 0:
                            rows.append(row)
                        print(f"{image.name} {name} round={repeat}: {row['bytes']} B "
                              f"{row['bpp']:.6f} bpp {row['wall_seconds']:.3f}s", flush=True)
                summary = dict(input=str(image))
                for name in names:
                    selected = [row for row in rows if row["encoder"] == name]
                    seconds = statistics.median(row["wall_seconds"] for row in selected)
                    summary[name] = dict(
                        bytes=statistics.median(row["bytes"] for row in selected),
                        bpp=statistics.median(row["bpp"] for row in selected),
                        seconds=seconds, mpix_per_second=selected[0]["mpix"] / seconds,
                    )
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
