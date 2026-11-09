"""Focused exact LZ77 extension experiments; official djxl checks original pixels.

Serial, order-rotated native encoder timings exclude input/output and process
startup. No builds, corpus discovery, byte-identity objective or default change.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from benchmark_fast_context import add_ratios, load_cases, summarize, validate_measurement
from benchmark_ma_models import trial_order
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process, sha256,
)

VARIANTS = ("baseline", "frozen", "scalar", "avx2", "avx512",
            "scalar-batch", "avx2-batch", "avx512-batch")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--frozen", type=Path, help="Optional pre-change encoder_lab control")
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=["baseline", "scalar", "avx2", "avx512", "avx2-batch", "avx512-batch"])
    p.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error("Invalid threads/repeats/warmups/timeout")
    if "baseline" not in args.variants or len(set(args.variants)) != len(args.variants):
        p.error("Use distinct variants including baseline")
    if "frozen" in args.variants and args.frozen is None:
        p.error("frozen variant requires --frozen")
    if not args.case and args.manifest is None:
        p.error("Provide --case or --manifest")
    if args.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return args


def command(args, source, output, variant):
    encoder = args.frozen if variant == "frozen" else args.encoder
    result = [str(encoder), str(source), "--effort", str(args.effort),
              "--threads", str(args.threads), "--model-costs", "--output", str(output)]
    if variant not in ("baseline", "frozen"):
        kernel, _, batch = variant.partition("-")
        result += ["--match-extension", kernel]
        if batch:
            result += ["--batch-matches"]
    return result


def validate_receipt(measurement, variant):
    receipt = measurement.get("match_extension")
    if variant == "frozen":
        return
    if "match_extension" not in measurement:
        raise ValueError("Missing match_extension receipt")
    if variant == "baseline":
        if receipt is not None:
            raise ValueError("Baseline unexpectedly used extension experiment")
        return
    kernel, _, batch = variant.partition("-")
    if not isinstance(receipt, dict) or receipt.get("mode") != kernel or receipt.get("batch") is not bool(batch):
        raise ValueError("Mismatch between requested kernel and executed kernel")
    for key in ("extensions", "batch_groups", "fast_matchers", "chain_matchers"):
        if type(receipt.get(key)) is not int or receipt[key] < 0:
            raise ValueError(f"Invalid extension counter {key}")
    if not batch and receipt["batch_groups"] != 0:
        raise ValueError("Unrequested batched matching")
    if receipt["batch_groups"] * 4 > receipt["extensions"]:
        raise ValueError("Inconsistent batch counters")
    if (receipt.get("active") is not (receipt["extensions"] != 0)
            or receipt.get("batch_active") is not (receipt["batch_groups"] != 0)):
        raise ValueError("Activity flags disagree with actual work counters")


def benchmark(args):
    report = dict(status="failed", config=vars(args).copy(), cases=[], errors=[],
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  timing="Native encoder only; serial trials, rotating order, median after warmups",
                  correctness="Original samples including alpha=0 RGB; official djxl, not JXL hashes",
                  scope="Fixed candidate at requested effort, not the outer scheduler",
                  timeout="Harness process guard only, no encoder deadline or pruning")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        try:
            env, report["environment"] = clean_environment(os.environ)
            identities = []
            for key in ("encoder", "decoder", "frozen"):
                path = getattr(args, key)
                if path is not None:
                    path = path.resolve(strict=True)
                    setattr(args, key, path)
                    digest = file_sha256(path)
                    identities.append((path, digest))
                    report[key] = dict(path=str(path), sha256=digest)
            with tempfile.TemporaryDirectory(prefix="lz-extension-", dir=args.out.parent) as scratch:
                for index, source in enumerate(load_cases(args)):
                    original = source.read_bytes()
                    expected = png_samples(original)
                    pixels = expected.width * expected.height
                    snapshot = Path(scratch) / "input.png"
                    snapshot.write_bytes(original)
                    case = dict(source=str(source), sha256=sha256(original), pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                variants={v: [] for v in args.variants})
                    report["cases"].append(case)
                    for repeat in range(-args.warmups, args.repeats):
                        for variant in trial_order(args.variants, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][variant].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=scratch) as trial:
                                jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                sample["encoder"] = run_process(command(args, snapshot, jxl, variant), env, args.timeout)
                                check_process(sample["encoder"])
                                sample["decoder"] = run_process(decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                check_process(sample["decoder"])
                                compare_pixels(expected, png_samples(png.read_bytes()))
                                measurement = json.loads(sample["encoder"]["stdout"])
                                validate_measurement(measurement, "baseline", "auto", jxl.stat().st_size, pixels)
                                validate_receipt(measurement, variant)
                                sample.update(verified=True, measurement=measurement)
                            print(f"{source.name} {variant} repeat={repeat}: "
                                  f"{measurement['bytes']} bytes {measurement['seconds']:.6f}s verified", flush=True)
                    case["summaries"] = {v: summarize(samples, pixels) for v, samples in case["variants"].items()}
                    add_ratios(case["summaries"])
            for path, digest in identities:
                if file_sha256(path) != digest:
                    raise ValueError(f"Binary changed during benchmark: {path}")
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


def main(argv=None):
    report = benchmark(arguments(argv))
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
