"""Compare two prebuilt palette_lab binaries with identical native e9 policies.

Uses mode=default (no run-order experiment, no patches). Native call time excludes
startup and IO; every trial is checked against original pixels by official djxl.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from benchmark_fast_context import add_ratios, load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
from benchmark_palette_runs import prepare_png, validate_measurement
from benchmark_tile_patches import aggregate
import compare_cpp_port as port


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[], type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 256 or not 1 <= args.repeats <= 100
            or not 0 <= args.warmups <= 100 or not positive_number(args.timeout)):
        parser.error("Invalid timing controls")
    if not args.case and args.manifest is None:
        parser.error("Provide --case or --manifest")
    if args.out.exists():
        parser.error("Report already exists")
    args.palette_colors, args.full_frame = None, False
    return args


def command(encoder, source, output, args):
    return [str(encoder), str(source), "--output", str(output), "--mode", "default",
            "--effort", str(args.effort), "--threads", str(args.threads)]


def benchmark(args):
    report = dict(status="failed", config=vars(args).copy(), cases=[], errors=[],
                  timing="One native encode per trial; rotated order; medians excluding warmups",
                  correctness="Original pixel samples including hidden RGB; independent official djxl",
                  scope="Fixed projected effort policy, patches disabled; not outer searchbest",
                  profiles="Diagnostic only; overlapping inclusive thread times are not wall times")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    variants = ("baseline", "candidate")
    with args.out.open("x", encoding="utf-8") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            if args.profile:
                env["RS_JXL_PROFILE"] = "1"
                report["environment"]["explicit"]["RS_JXL_PROFILE"] = "1"
            for name in ("before", "after", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable: {path}")
                setattr(args, name, path)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                identities.append((args.manifest, port.file_sha256(args.manifest)))
            with tempfile.TemporaryDirectory(prefix="lz-hotspots-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                for index, source in enumerate(load_cases(args)):
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    identities.append((source, digest))
                    payload, expected, preparation = prepare_png(source, original, temporary)
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(payload)
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=preparation, variants={key: [] for key in variants})
                    report["cases"].append(case)
                    policy = None
                    for repeat in range(-args.warmups, args.repeats):
                        for variant in trial_order(variants, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][variant].append(sample)
                            encoder = args.before if variant == "baseline" else args.after
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                sample["encoder"] = port.run_process(command(encoder, snapshot, jxl, args), env, args.timeout)
                                port.check_process(sample["encoder"])
                                sample["decoder"] = port.run_process(
                                    port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                port.check_process(sample["decoder"])
                                port.compare_pixels(expected, port.png_samples(png.read_bytes()))
                                measurement = json.loads(sample["encoder"]["stdout"])
                                validate_measurement(measurement, "default", jxl.stat().st_size,
                                                     expected.width, expected.height, args)
                                observed = {key: measurement[key] for key in ("configured", "projected", "overrides")}
                                if policy is not None and observed != policy:
                                    raise ValueError("Native policy changed between binaries or trials")
                                policy = observed
                                sample.update(verified=True, measurement=measurement)
                            print(f"{source.name} {variant} repeat={repeat}: {measurement['bytes']} bytes "
                                  f"{measurement['seconds']:.6f}s verified", flush=True)
                    case["summaries"] = {key: summarize(samples, pixels)
                                         for key, samples in case["variants"].items()}
                    add_ratios(case["summaries"])
                report["aggregate"] = aggregate(report["cases"], variants)
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if port.file_sha256(path) == digest:
                        continue
                    message = f"Input or binary changed: {path}"
                except OSError as error:
                    message = str(error)
                report["errors"].append(message)
            if report["errors"] or not report["temporary_outputs_cleaned"]:
                report["status"] = "failed"
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


def main(argv=None):
    report = benchmark(arguments(argv))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
