"""Compare frozen patch_lab binaries with identical --mode source settings.

No builds or discovery. Native timings exclude startup and PNG IO. Official
djxl checks original samples, including hidden RGB. --profile adds separate
diagnostic encodes per case/version, never included in timing summaries.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace

from benchmark_fast_context import load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
from benchmark_palette_runs import prepare_png
import benchmark_qlic_candidates as qlic
import compare_cpp_port as port
from corpus_inputs import png_framing


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--decoder", "--official-djxl", "--djxl", type=Path, required=True,
                   help="Caller-attested official djxl executable")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--profile", action="store_true", help="Separate diagnostic runs only")
    p.add_argument("--profile-repeats", type=int, default=1,
                   help="Independent diagnostic repetitions when --profile is enabled")
    p.add_argument("--profile-detail", action="store_true",
                   help="Enable Source Patch substage timers in diagnostic runs only")
    p.add_argument("--out", type=Path, required=True, help="New JSON report; never overwritten")
    args = p.parse_args(argv)
    if args.profile_detail and not args.profile:
        p.error("--profile-detail requires --profile")
    encoders = {}
    for assignment in args.encoder:
        label, separator, path = assignment.partition("=")
        if not separator or not label.strip() or label != label.strip() or not path.strip():
            p.error("Expected --encoder LABEL=PATH")
        if label in encoders:
            p.error("Encoder labels must be distinct")
        encoders[label] = Path(path)
    if len(encoders) < 2 or len({path.resolve() for path in encoders.values()}) != len(encoders):
        p.error("Provide at least two distinct encoder labels and paths")
    args.encoder = encoders
    if (not 1 <= args.threads <= 256 or not 1 <= args.repeats <= 100
            or not 0 <= args.warmups <= 100 or not 1 <= args.profile_repeats <= 100
            or not positive_number(args.timeout)):
        p.error("Invalid threads/repeats/warmups/timeout")
    if not args.case and args.manifest is None and not args.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if args.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return args


def command(args, label, source, output, profile=False):
    options = SimpleNamespace(**(vars(args) | {"encoder": args.encoder[label]}))
    result = qlic.command(options, source, output, "source")
    return result + ["--profile"] if profile else result


def aggregate(cases, labels):
    if not cases:
        raise ValueError("Cannot aggregate an empty corpus")
    pixels = sum(case["pixels"] for case in cases)
    result = {}
    for label in labels:
        summaries = [case["summaries"][label] for case in cases]
        size = sum(s["bytes"] for s in summaries)
        seconds = sum(s["seconds"] for s in summaries)
        result[label] = dict(total_bytes=size, total_median_seconds=seconds,
                             mean_bpp=statistics.mean(s["bpp"] for s in summaries),
                             weighted_bpp=8 * size / pixels,
                             mpix_per_second=pixels / 1e6 / seconds)
    return result


def run_trial(args, label, snapshot, expected, temporary, env, sample, profile=False):
    # Keep failed process records on the sample; the caller writes a failure report.
    with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as folder:
        jxl, png = Path(folder) / "out.jxl", Path(folder) / "out.png"
        encoder_env = dict(env)
        if profile:
            encoder_env["RS_JXL_PROFILE"] = "1"
            if args.profile_detail:
                encoder_env["RS_JXL_PATCH_SOURCE_PROFILE"] = "1"
        sample["encoder"] = port.run_process(
            command(args, label, snapshot, jxl, profile), encoder_env, args.timeout)
        port.check_process(sample["encoder"])
        sample["decoder"] = port.run_process(
            port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
        port.check_process(sample["decoder"])
        port.compare_pixels(expected, port.png_samples(png.read_bytes()))
        measurement = json.loads(sample["encoder"]["stdout"])
        qlic.validate_measurement(measurement, "source", jxl.stat().st_size,
                                  expected.width, expected.height)
        sample.update(verified=True, measurement=measurement)


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(),
                  cases=[], errors=[],
                  timing="Native seconds; per-case medians exclude warmups, profiling, startup and PNG IO",
                  aggregate_contract="All cases required; sums of per-case median bytes/seconds; mean and pixel-weighted bpp",
                  correctness="Official djxl vs original 8/16-bit samples including hidden RGB",
                  official_decoder_provenance="Caller-attested; path and SHA256 recorded",
                  profiling=dict(enabled=args.profile, included_in_timing=False,
                                 runs_per_case_version=args.profile_repeats if args.profile else 0,
                                 explicit=({"RS_JXL_PROFILE": "1"}
                                           | ({"RS_JXL_PATCH_SOURCE_PROFILE": "1"}
                                              if args.profile_detail else {}))
                                 if args.profile else {}))
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            report["encoders"] = {}
            for label, binary in args.encoder.items():
                path = binary.resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                digest = port.file_sha256(path)
                identities.append((path, digest))
                args.encoder[label] = path
                report["encoders"][label] = dict(path=str(path), sha256=digest)
            args.decoder = args.decoder.resolve(strict=True)
            if not args.decoder.is_file():
                raise ValueError(f"Not an executable file: {args.decoder}")
            digest = port.file_sha256(args.decoder)
            identities.append((args.decoder, digest))
            report["decoder"] = dict(path=str(args.decoder), sha256=digest)
            if args.manifest is not None:
                digest = port.file_sha256(args.manifest)
                identities.append((args.manifest, digest))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=digest)
            labels = list(args.encoder)
            with tempfile.TemporaryDirectory(prefix="source-patch-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                external = set(sources)
                if args.synthetic:
                    sources += qlic.fixtures(temporary / "synthetic")
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    if source in external:
                        identities.append((source, digest))
                    payload, expected, preparation = prepare_png(source, original, temporary)
                    _, info = png_framing(original)
                    info["mode"] = preparation
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(payload)
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=info, snapshot_sha256=port.sha256(payload),
                                variants={label: [] for label in labels}, summaries={},
                                diagnostics={label: [] for label in labels}, execution_order=[])
                    report["cases"].append(case)
                    for repetition in range(-args.warmups, args.repeats):
                        for label in trial_order(labels, repetition, index):
                            sample = dict(repetition=repetition, warmup=repetition < 0, verified=False)
                            case["variants"][label].append(sample)
                            case["execution_order"].append(dict(label=label, repetition=repetition))
                            run_trial(args, label, snapshot, expected, temporary, env, sample)
                    case["summaries"] = {label: summarize(samples, pixels)
                                         for label, samples in case["variants"].items()}
                    if args.profile:
                        for repetition in range(args.profile_repeats):
                            for label in trial_order(labels, args.repeats + repetition, index):
                                sample = dict(profile=True, repetition=repetition,
                                              included_in_timing=False, verified=False)
                                case["diagnostics"][label].append(sample)
                                run_trial(args, label, snapshot, expected, temporary, env, sample, profile=True)
                    print(f"{source.name}: all versions verified", flush=True)
                report["aggregate"] = aggregate(report["cases"], labels)
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if port.file_sha256(path) == digest:
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
    try:
        report = benchmark(arguments(argv))
    except (OSError, ValueError) as error:
        print(f"Benchmark/report failure: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
