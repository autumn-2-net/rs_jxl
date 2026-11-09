"""Measure explicit MA description-cost/prefix experiments and verify original pixels.

Uses the development encoder_lab, not a changed production default. Encoder
timing comes from its native call; startup, PNG input/output and independent
djxl validation are excluded. Profiling is diagnostic only, never a speed claim.
Nonzero prefix levels require the archived shared-prefix prototype executable;
the unsuccessful prefix learner was removed from maintained encoder sources.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--modes", nargs="+", choices=["off", "quarter", "half", "full"],
                        default=["off", "quarter", "half", "full"])
    parser.add_argument("--prefix-levels", nargs="+", type=int, choices=range(256), default=[0])
    parser.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or len(set(args.modes)) != len(args.modes)
            or len(set(args.prefix_levels)) != len(args.prefix_levels) or args.out.exists()):
        parser.error("Invalid settings or output already exists")
    return args


def trial_order(variants, repetition, case_index):
    offset = (repetition + case_index) % len(variants)
    order = variants[offset:] + variants[:offset]
    # Keep the rotating first entry: reversing the entire list would cancel
    # alternation when comparing exactly two variants.
    return order[:1] + order[:0:-1] if repetition % 2 else order


def validate_measurement(measurement, mode, encoded_size, prefix_levels=0):
    seconds = measurement.get("seconds")
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Missing/invalid native encoder time")
    if measurement.get("bytes") != encoded_size:
        raise ValueError("Reported size differs from actual codestream")
    if measurement.get("description_cost", "").lower() != mode or measurement.get("search") is not False:
        raise ValueError("Encoder did not confirm the requested direct experiment")
    confirmed_prefix = measurement.get("ma_prefix_levels", 0)
    if (isinstance(confirmed_prefix, bool) or not isinstance(confirmed_prefix, int)
            or confirmed_prefix != prefix_levels):
        raise ValueError("Encoder did not confirm the requested MA prefix levels")
    costs = measurement.get("model_costs")
    if not isinstance(costs, dict) or any(
        isinstance(costs.get(k), bool) or not isinstance(costs.get(k), int) or costs[k] < 0
        for k in ["tree_bits", "entropy_header_bits", "total_bits"]
    ):
        raise ValueError("Missing/invalid emitted model costs")
    if costs["total_bits"] != costs["tree_bits"] + costs["entropy_header_bits"]:
        raise ValueError("Model costs do not sum")
    if costs["total_bits"] > encoded_size * 8:
        raise ValueError("Model receipt includes more bits than the entire output")


def profile_rows(stderr):
    result = {}
    for line in stderr.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == 4 and parts[0].startswith("ma."):
            result[parts[0]] = dict(calls=int(parts[1]), seconds=float(parts[2]),
                                    longest_seconds=float(parts[3]))
    return result


def summarize(samples, pixels):
    measured = [sample for sample in samples if not sample["warmup"]]
    times = [sample["measurement"]["seconds"] for sample in measured]
    sizes = sorted({sample["measurement"]["bytes"] for sample in measured})
    seconds = statistics.median(times)
    return dict(seconds=seconds, min_seconds=min(times), max_seconds=max(times),
                sizes=sizes, bpp=[8 * size / pixels for size in sizes],
                mpix_s=pixels / 1e6 / seconds)


def benchmark(args):
    variants = [(mode, levels) for mode in args.modes for levels in args.prefix_levels]

    def variant_key(mode, levels):
        return mode if args.prefix_levels == [0] else f"{mode}/prefix-{levels}"

    env, removed = clean_environment(os.environ)
    if args.profile:
        env.update(RS_JXL_PROFILE="1", RS_JXL_PROFILE_MA_NODES="1")
    report = dict(status="running", created=datetime.now(timezone.utc).isoformat(),
                  config=vars(args), environment=removed, cases=[], errors=[],
                  measurement="Native encoder call only; startup/IO/verification excluded",
                  profiling_is_not_benchmark=args.profile)
    identities = []
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        try:
            for name in ["encoder", "decoder"]:
                path = getattr(args, name).resolve(strict=True)
                setattr(args, name, path)
                digest = file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            with tempfile.TemporaryDirectory(prefix="ma-models-", dir=args.out.parent) as temporary:
                temporary = Path(temporary)
                for case_index, source in enumerate(args.case):
                    source = source.resolve(strict=True)
                    digest = file_sha256(source)
                    identities.append((source, digest))
                    expected = png_samples(source.read_bytes())
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                depth=expected.depth, width=expected.width, height=expected.height,
                                modes={variant_key(*variant): [] for variant in variants}, summaries={})
                    report["cases"].append(case)
                    for repetition in range(-args.warmups, args.repeats):
                        order = trial_order(variants, repetition, case_index)
                        for mode, levels in order:
                            key = variant_key(mode, levels)
                            stem = f"{case_index}-{repetition}-{mode}-prefix-{levels}"
                            jxl, png = temporary / f"{stem}.jxl", temporary / f"{stem}.png"
                            command = [str(args.encoder), str(source), "--effort", str(args.effort),
                                       "--threads", str(args.threads), "--description-cost", mode,
                                       "--model-costs", "--output", str(jxl)]
                            if levels:
                                command.extend(["--ma-prefix-levels", str(levels)])
                            encoded = run_process(command, env, args.timeout)
                            sample = dict(repetition=repetition, warmup=repetition < 0,
                                          config=dict(description_cost=mode, ma_prefix_levels=levels),
                                          encoder=encoded, verified=False)
                            case["modes"][key].append(sample)
                            check_process(encoded)
                            measurement = json.loads(encoded["stdout"])
                            validate_measurement(measurement, mode, jxl.stat().st_size, levels)
                            sample["measurement"] = measurement
                            sample["profile"] = profile_rows(encoded["stderr"])
                            decoded = run_process(decoder_command(args.decoder, jxl, png, expected.depth),
                                                  env, args.timeout)
                            sample["decoder"] = decoded
                            check_process(decoded)
                            compare_pixels(expected, png_samples(png.read_bytes()))
                            sample["verified"] = True
                            print(f"{source.name} {key}: {measurement['bytes']} bytes "
                                  f"{measurement['seconds']:.3f}s verified", flush=True)
                    case["summaries"] = {mode: summarize(samples, pixels)
                                         for mode, samples in case["modes"].items()}
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
            report["status"] = "failed"
        finally:
            for path, digest in identities:
                try:
                    if file_sha256(path) == digest:
                        continue
                    error = f"Input/binary changed during measurement: {path}"
                except OSError as failure:
                    error = str(failure)
                if error:
                    report["status"] = "failed"
                    report["errors"].append(error)
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


if __name__ == "__main__":
    result = benchmark(arguments())
    print(result["status"], result["errors"], flush=True)
    raise SystemExit(result["status"] != "passed")
