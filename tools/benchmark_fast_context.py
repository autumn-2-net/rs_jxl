"""Benchmark frozen encoder_lab fast-context experiments against original PNG pixels.

Native-call seconds exclude startup, PNG I/O and independent official djxl checks.
The timeout is a harness guard, not an encoder heuristic. No builds or discovery
are performed. Decoder provenance is caller-attested. Scan backends are compared
by separate invocations/reports; baseline never receives experimental flags.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

from benchmark_ma_models import trial_order
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process, sha256,
)


VARIANTS = ("baseline", "leaf", "gradient12", "directional12", "auto12",
            "auto12-static", "auto12-adaptive")
SCANS = ("auto", "scalar", "avx2", "avx512")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True,
                        help="Explicit frozen encoder_lab executable; never built by this tool")
    parser.add_argument("--decoder", "--official-djxl", dest="decoder", type=Path, required=True,
                        help="Caller-attested official djxl executable")
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--scan", choices=SCANS, default="auto")
    parser.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.case and args.manifest is None:
        parser.error("At least one --case or --manifest is required")
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid threads/repeats/warmups/timeout")
    if len(set(args.variants)) != len(args.variants) or "baseline" not in args.variants:
        parser.error("Variants must be distinct and include baseline for per-case ratios")
    if args.scan != "auto" and args.variants == ["baseline"]:
        parser.error("Explicit --scan requires an experimental variant")
    if args.out.exists():
        parser.error("Output already exists; reports are never overwritten")
    return args


def load_cases(args):
    paths = list(args.case)
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        cases = manifest.get("cases") if isinstance(manifest, dict) else None
        if not isinstance(cases, list) or not cases:
            raise ValueError("Manifest must contain a nonempty cases list")
        for case in cases:
            original = case.get("original") if isinstance(case, dict) else None
            if not isinstance(original, str) or not original:
                raise ValueError("Each manifest case must have an original PNG path")
            path = Path(original)
            paths.append(path if path.is_absolute() else args.manifest.resolve().parent / path)
    resolved = [path.resolve(strict=True) for path in paths]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("Input cases must be nonempty and distinct")
    return resolved


def variant_parameters(variant, scan="auto"):
    if variant not in VARIANTS or scan not in SCANS:
        raise ValueError("Unknown fast-context variant/scan")
    if variant == "baseline":
        return None
    mode, separator, bias = variant.partition("-")
    return dict(mode=mode, bias=bias if separator else "off", scan=scan)


def encoder_command(args, source, output, variant):
    command = [str(args.encoder), str(source), "--effort", str(args.effort),
               "--threads", str(args.threads), "--model-costs", "--output", str(output)]
    parameters = variant_parameters(variant, args.scan)
    if parameters is not None:
        command += ["--fast-context", parameters["mode"], "--context-bias", parameters["bias"],
                    "--context-scan", parameters["scan"]]
    return command


def positive_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def validate_measurement(measurement, variant, scan, encoded_size, pixels):
    if not isinstance(measurement, dict):
        raise ValueError("Encoder JSON must be an object")
    seconds = measurement.get("seconds")
    if not positive_number(seconds):
        raise ValueError("Missing/invalid native encoder time")
    size = measurement.get("bytes")
    if type(size) is not int or size <= 0 or size != encoded_size:
        raise ValueError("Reported size differs from actual nonempty codestream")
    for key, expected in (("bpp", size * 8 / pixels),
                          ("mpix_per_second", pixels / 1e6 / seconds)):
        actual = measurement.get(key)
        if not positive_number(actual) or not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"Missing/inconsistent {key}")
    if measurement.get("search") is not False:
        raise ValueError("Encoder did not confirm search=false")
    costs = measurement.get("model_costs")
    if not isinstance(costs, dict) or any(
        type(costs.get(key)) is not int or costs[key] < 0
        for key in ("tree_bits", "entropy_header_bits", "total_bits")
    ):
        raise ValueError("Missing/invalid emitted model costs")
    if (costs["total_bits"] != costs["tree_bits"] + costs["entropy_header_bits"]
            or costs["total_bits"] > size * 8):
        raise ValueError("Model costs do not sum or exceed codestream size")
    parameters = variant_parameters(variant, scan)
    if "fast_context" not in measurement:
        raise ValueError("Missing fast_context receipt (baseline must explicitly report null)")
    receipt = measurement["fast_context"]
    if parameters is None:
        if receipt is not None:
            raise ValueError("Baseline unexpectedly used fast context")
        return
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in parameters.items()):
        raise ValueError("Fast-context receipt parameters differ from requested experiment")
    for key in ("trees", "leaves", "pixels", "offset_leaves", "horizontal", "vertical", "single"):
        if type(receipt.get(key)) is not int or receipt[key] < 0:
            raise ValueError(f"Invalid fast-context counter: {key}")
    if any(receipt[key] == 0 for key in ("trees", "leaves", "pixels")):
        raise ValueError("Fast-context experiment has a zero work receipt")
    # Mode counters describe channel plans, not MA trees. There is no separate
    # channel-plan total in the CLI receipt, so do not equate this sum to trees.
    if receipt["horizontal"] + receipt["vertical"] + receipt["single"] == 0:
        raise ValueError("Fast-context experiment has no channel-plan modes")


def summarize(samples, pixels):
    measured = [sample["measurement"] for sample in samples
                if not sample["warmup"] and sample["verified"]]
    times = [item["seconds"] for item in measured]
    sizes = [item["bytes"] for item in measured]
    seconds, size = statistics.median(times), statistics.median(sizes)
    return dict(seconds=seconds, min_seconds=min(times), max_seconds=max(times),
                bytes=size, sizes=sorted(set(sizes)), bpp=8 * size / pixels,
                mpix_per_second=pixels / 1e6 / seconds)


def add_ratios(summaries):
    baseline = summaries["baseline"]
    for summary in summaries.values():
        summary["ratios_to_baseline"] = {
            key: summary[key] / baseline[key]
            for key in ("bytes", "seconds", "bpp", "mpix_per_second")
        }


def benchmark(args):
    report = dict(schema_version=1, status="failed", created_utc=datetime.now(timezone.utc).isoformat(),
                  config=vars(args).copy(), cases=[], errors=[],
                  timing="Native encoder call only; median seconds; warmups excluded",
                  pixel_contract="Original PNG samples including depth and hidden RGB",
                  official_decoder_provenance="Caller-attested; path and SHA256 recorded",
                  ratio_contract="variant / per-case baseline; throughput uses median native time",
                  timeout_contract="Harness process guard, not an encoder heuristic")
    identities = []
    temporary = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new report before execution, including failure reports.
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = clean_environment(os.environ)
            for name in ("encoder", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(args, name, path)
                digest = file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                digest = file_sha256(args.manifest)
                identities.append((args.manifest, digest))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=digest)
            sources = load_cases(args)
            with tempfile.TemporaryDirectory(prefix="fast-context-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = sha256(original)
                    identities.append((source, digest))
                    expected = png_samples(original)
                    pixels = expected.width * expected.height
                    # All variants see exactly the same original PNG snapshot.
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(original)
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                variants={variant: [] for variant in args.variants}, summaries={})
                    report["cases"].append(case)
                    for repetition in range(-args.warmups, args.repeats):
                        for variant in trial_order(args.variants, repetition, index):
                            sample = dict(repetition=repetition, warmup=repetition < 0,
                                          parameters=variant_parameters(variant, args.scan), verified=False)
                            case["variants"][variant].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, png = Path(trial) / "output.jxl", Path(trial) / "decoded.png"
                                encoded = run_process(encoder_command(args, snapshot, jxl, variant),
                                                      env, args.timeout)
                                sample["encoder"] = encoded
                                check_process(encoded)
                                # Validate every produced artifact, even if its JSON receipt is wrong.
                                decoded = run_process(decoder_command(args.decoder, jxl, png, expected.depth),
                                                      env, args.timeout)
                                sample["decoder"] = decoded
                                check_process(decoded)
                                compare_pixels(expected, png_samples(png.read_bytes()))
                                sample["verified"] = True
                                measurement = json.loads(encoded["stdout"])
                                validate_measurement(measurement, variant, args.scan, jxl.stat().st_size, pixels)
                                sample["measurement"] = measurement
                            print(f"{source.name} {variant} scan={args.scan} warmup={sample['warmup']}: "
                                  f"{measurement['bytes']} bytes {measurement['seconds']:.6f}s verified",
                                  flush=True)
                    case["summaries"] = {variant: summarize(samples, pixels)
                                         for variant, samples in case["variants"].items()}
                    add_ratios(case["summaries"])
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if file_sha256(path) == digest:
                        continue
                    error = f"Input/binary changed during measurement: {path}"
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
