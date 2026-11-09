"""Paired opt-in PR5007 MA learning trials against original PNG samples.

Native encode timings exclude startup, image I/O and official djxl validation.
Size/time regressions are measurements, not failures. No builds, corpus discovery
or JXL-byte equality checks. Official decoder provenance is caller-attested.
Experimental learners require effort >= 4; efforts 1..3 use fixed trees.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from benchmark_fast_context import (
    add_ratios, load_cases, summarize, validate_measurement as validate_native,
)
from benchmark_ma_models import trial_order
from benchmark_refinement import command as native_command, fixtures
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process, sha256,
)

VARIANTS = ("baseline", "cost", "symmetric", "frozen")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--decoder", "--official-djxl", dest="decoder", type=Path, required=True)
    p.add_argument("--frozen", type=Path, help="Optional old encoder_lab; adds frozen to default variants")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--variants", nargs="+", choices=VARIANTS)
    p.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--buffering", type=int, choices=(-1, 0, 1, 2))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=180, help="Harness process guard only")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.variants is None:
        args.variants = list(VARIANTS if args.frozen is not None else VARIANTS[:-1])
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error("Invalid threads/repeats/warmups/timeout")
    if "baseline" not in args.variants or len(set(args.variants)) != len(args.variants):
        p.error("Use distinct variants including baseline")
    if "frozen" in args.variants and args.frozen is None:
        p.error("frozen variant requires --frozen")
    if args.effort < 4 and any(v in ("cost", "symmetric") for v in args.variants):
        p.error("cost/symmetric require effort >= 4; efforts 1..3 use fixed trees")
    if not args.case and args.manifest is None and not args.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return args


def command(args, source, output, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown tree-learning variant: {variant}")
    if variant in ("cost", "symmetric") and args.effort < 4:
        raise ValueError("cost/symmetric require effort >= 4; efforts 1..3 use fixed trees")
    result = native_command(args, source, output, "frozen" if variant == "frozen" else "baseline")
    if variant in ("cost", "symmetric"):
        result += ["--tree-learning", variant]
    return result


def validate_measurement(measurement, variant, encoded_size, pixels):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown tree-learning variant: {variant}")
    if not isinstance(measurement, dict):
        raise ValueError("Encoder JSON must be an object")
    # Reuse native timing/size/model-cost checks without requiring an unrelated
    # fast-context receipt from older binaries.
    native = dict(measurement)
    native.setdefault("fast_context", None)
    validate_native(native, "baseline", "auto", encoded_size, pixels)
    expected = "legacy" if variant in ("baseline", "frozen") else variant
    old_frozen = variant == "frozen" and "tree_learning" not in measurement
    if not old_frozen and measurement.get("tree_learning") != expected:
        raise ValueError(f"Missing/mismatched tree_learning receipt: expected {expected!r}")
    if variant == "frozen" and "tree_learning_work" not in measurement:
        return
    if "tree_learning_work" not in measurement:
        raise ValueError("Missing tree_learning_work receipt")
    work = measurement["tree_learning_work"]
    if variant in ("baseline", "frozen"):
        if work is not None:
            raise ValueError("Legacy tree_learning_work must be null")
        return
    if not isinstance(work, dict):
        raise ValueError("Experimental tree_learning_work must be an object")
    for key in ("learners", "records", "cost_splits", "symmetric_splits"):
        if type(work.get(key)) is not int or work[key] < 0:
            raise ValueError(f"Invalid tree_learning_work counter: {key}")
    if variant == "cost" and work["symmetric_splits"] != 0:
        raise ValueError("cost mode unexpectedly reports symmetric_splits")


def aggregate(cases, variants):
    """Ratios of corpus totals of per-case medians, not means of ratios."""
    if not cases:
        raise ValueError("Cannot aggregate an empty case list")
    pixels = sum(case["pixels"] for case in cases)
    result = {}
    for variant in variants:
        size = sum(case["summaries"][variant]["bytes"] for case in cases)
        seconds = sum(case["summaries"][variant]["seconds"] for case in cases)
        result[variant] = dict(bytes=size, seconds=seconds, bpp=size * 8 / pixels,
                               mpix_per_second=pixels / 1e6 / seconds)
    add_ratios(result)
    return result


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(),
                  created_utc=datetime.now(timezone.utc).isoformat(), cases=[], errors=[],
                  timing="Native encode only; serial order rotation; medians exclude warmups",
                  correctness="Original PNG samples including hidden RGB and depth; official djxl",
                  official_decoder_provenance="Caller-attested; path and SHA256 pinned",
                  ratio_contract="variant / baseline; aggregate ratios use sums of per-case medians",
                  regression_policy="Size/time regressions are reported, not rejected",
                  timeout_contract="Process guard only; no encoder timeout policy")
    identities = []
    temporary = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = clean_environment(os.environ)
            names = ["encoder", "decoder"] + (["frozen"] if "frozen" in args.variants else [])
            for name in names:
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(args, name, path)
                digest = file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                identities.append((args.manifest, file_sha256(args.manifest)))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=identities[-1][1])
            with tempfile.TemporaryDirectory(prefix="tree-learning-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                if args.synthetic:
                    sources += fixtures(temporary)
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    expected = png_samples(original)
                    pixels = expected.width * expected.height
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(original)
                    case = dict(source=str(source), sha256=sha256(original), pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                variants={v: [] for v in args.variants})
                    report["cases"].append(case)
                    for repeat in range(-args.warmups, args.repeats):
                        for variant in trial_order(args.variants, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][variant].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                sample["encoder"] = run_process(command(args, snapshot, jxl, variant), env, args.timeout)
                                check_process(sample["encoder"])
                                sample["decoder"] = run_process(decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                check_process(sample["decoder"])
                                compare_pixels(expected, png_samples(png.read_bytes()))
                                measurement = json.loads(sample["encoder"]["stdout"])
                                validate_measurement(measurement, variant, jxl.stat().st_size, pixels)
                                sample.update(measurement=measurement, verified=True)
                            print(f"{source.name} {variant} repeat={repeat}: "
                                  f"{measurement['bytes']} bytes {measurement['seconds']:.6f}s verified", flush=True)
                    case["summaries"] = {v: summarize(samples, pixels) for v, samples in case["variants"].items()}
                    add_ratios(case["summaries"])
                report["aggregate"] = aggregate(report["cases"], args.variants)
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if file_sha256(path) == digest:
                        continue
                    error = f"Binary/manifest changed during benchmark: {path}"
                except OSError as failure:
                    error = str(failure)
                report["errors"].append(error)
            if report["errors"] or not report["temporary_outputs_cleaned"]:
                report["status"] = "failed"
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


def main(argv=None):
    args = arguments(argv)
    try:
        report = benchmark(args)
    except (OSError, ValueError) as error:
        print(f"Benchmark/report failure: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
