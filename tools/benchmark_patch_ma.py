"""Bounded e9 full-MA patch_lab experiments, using prebuilt binaries only.

--variants selects routes, not a cartesian product. Holdout: off off-prev1
off-prev2 off-gather. Positive cohort: source source-gather
source-prev2-residual source-prev2-both source-gather-prev2-residual
source-gather-prev2-both. Optional --profile runs separate diagnostics, excluded
from timing. selected_mode is an enabled route, NEVER proof of emitted patches.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from benchmark_fast_context import load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
from benchmark_palette_runs import prepare_png
import benchmark_qlic_candidates as qlic
import benchmark_source_patch_detector as source_bench
import compare_cpp_port as port
from corpus_inputs import png_framing


# Include explicit zero/one controls and reference-only controls without forcing
# them onto either of the parent's bounded cohorts.
VARIANTS = {"off": ("baseline", False, None, None),
            "off-gather": ("baseline", True, None, None)}
for previous in range(3):
    VARIANTS[f"off-prev{previous}"] = ("baseline", False, previous, "residual")
for gather in (False, True):
    stem = "source-gather" if gather else "source"
    VARIANTS[stem] = ("source", gather, None, None)
    for previous in range(3):
        for target in ("residual", "reference", "both"):
            VARIANTS[f"{stem}-prev{previous}-{target}"] = ("source", gather, previous, target)
for variant, config in list(VARIANTS.items()):
    if "gather" in variant:
        VARIANTS[variant.replace("gather", "waves")] = config
DEFAULT_VARIANTS = ["off", "off-prev1", "off-prev2", "off-gather"]


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--decoder", "--official-djxl", "--djxl", type=Path, required=True,
                   help="Caller-attested official djxl")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS.copy())
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--profile", action="store_true", help="One separate diagnostic per case/variant")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if (not 1 <= args.threads <= 256 or not 1 <= args.repeats <= 100
            or not 0 <= args.warmups <= 100 or not positive_number(args.timeout)):
        p.error("Invalid threads/repeats/warmups/timeout")
    if len(set(args.variants)) != len(args.variants):
        p.error("Variants must be distinct")
    if not args.case and args.manifest is None:
        p.error("Provide --case or --manifest")
    if args.out.exists():
        p.error("Report already exists; never overwritten")
    return args


def command(args, snapshot, output, variant, profile=False):
    mode, gather, previous, target = VARIANTS[variant]
    result = qlic.command(args, snapshot, output, mode)
    if gather:
        result.append("--parallel-gather")
    if "waves" in variant:
        result.append("--gather-waves")
    if previous is not None:
        result += ["--previous-channels", str(previous), "--channel-target", target]
    return result + ["--profile"] if profile else result


def validate_measurement(m, variant, size, width, height):
    mode, gather, previous, target = VARIANTS[variant]
    qlic.validate_measurement(m, mode + "-parallel" if gather else mode, size, width, height)
    if m.get("parallel_gather") is not gather:
        raise ValueError("Mismatched gather activation")
    expected_gather_mode = "waves" if "waves" in variant else "channels"
    if m.get("gather_mode", "channels") != expected_gather_mode:
        raise ValueError("Mismatched gather mode")
    channels = m.get("channels")
    if (not isinstance(channels, dict) or channels.get("requested") != previous
            or (previous is not None and type(channels.get("requested")) is not int)
            or channels.get("target") != target):
        raise ValueError("Missing/mismatched channel activation receipt")
    reference_available = mode == "source" and m["source_detector_enabled"]
    for part in ("main", "reference"):
        work = channels.get(part)
        available = part == "main" or reference_available
        configured = available and previous is not None and target in (
            "residual" if part == "main" else "reference", "both")
        if not isinstance(work, dict) or work.get("configured") is not configured:
            raise ValueError(f"Mismatched {part} channel configuration")
        if part == "reference" and work.get("available") is not available:
            raise ValueError("Mismatched reference availability")
        effective = work.get("effective_previous_channels")
        if not available:
            if effective is not None or work.get("configured_previous_channels") is not None:
                raise ValueError("Unavailable reference must have null budgets")
            continue
        if type(effective) is not int or not 0 <= effective <= 2:
            raise ValueError("Invalid effective previous channels")
        budget = work.get("configured_previous_channels")
        if configured and (type(budget) is not int or budget != previous or effective > previous):
            raise ValueError("Mismatched configured previous channels")
        if part == "reference" and (type(budget) is not int or not 0 <= budget <= 2):
            raise ValueError("Invalid reference raw budget")
    if "source_patch_selected" not in m or m["source_patch_selected"] is not (
            False if mode == "baseline" else None):
        raise ValueError("Ordinary receipt must not claim actual source Patch emission")


def patch_profile(stderr, enabled):
    """One fresh patch_lab process, one encode: existing scopes prove emission."""
    if "inclusive process totals" not in stderr:
        raise ValueError("Missing enabled profile report")
    stages = {}
    for line in stderr.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if fields[0] not in ("encode.patch_reference", "encode.patch_residual"):
            continue
        if len(fields) != 4 or fields[0] in stages:
            raise ValueError("Invalid/duplicate patch profile stage")
        calls, seconds = int(fields[1]), float(fields[2])
        if calls != 1 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid patch profile calls/time")
        stages[fields[0]] = seconds
    selected = bool(stages)
    if selected and (len(stages) != 2 or not enabled):
        raise ValueError("Inconsistent reference/residual profile emission")
    return dict(source_patch_selected=selected, evidence="separate diagnostic profile scopes",
                reference_seconds=stages.get("encode.patch_reference"),
                residual_seconds=stages.get("encode.patch_residual"))


def run_trial(args, variant, snapshot, expected, temporary, env, sample, profile=False):
    with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as folder:
        jxl, png = Path(folder) / "out.jxl", Path(folder) / "out.png"
        encoder_env = dict(env)
        if profile:
            encoder_env["RS_JXL_PROFILE"] = "1"
        sample["encoder"] = port.run_process(
            command(args, snapshot, jxl, variant, profile), encoder_env, args.timeout)
        port.check_process(sample["encoder"])
        sample["decoder"] = port.run_process(
            port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
        port.check_process(sample["decoder"])
        port.compare_pixels(expected, port.png_samples(png.read_bytes()))
        measurement = json.loads(sample["encoder"]["stdout"])
        validate_measurement(measurement, variant, jxl.stat().st_size, expected.width, expected.height)
        if profile:
            sample["patch_application"] = patch_profile(
                sample["encoder"]["stderr"], measurement["source_detector_enabled"] is True)
        sample.update(verified=True, measurement=measurement)


def benchmark(args):
    report = dict(schema_version=1, status="running", config=vars(args).copy(), cases=[], errors=[],
                  timing="Native seconds; per-case medians exclude warmups and separate diagnostics",
                  correctness="Official djxl vs original samples including hidden RGB",
                  official_decoder_provenance="Caller-attested; path and SHA256 recorded",
                  patch_contract="source/selected_mode means enabled route, not emitted dictionary; main may be whole frame, not residual",
                  aggregate_contract="All cases/trials required per variant; no partial-success aggregates; sums of per-case medians")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            for name in ("encoder", "decoder", "manifest"):
                binary = getattr(args, name)
                if binary is None:
                    continue
                path = binary.resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not a file: {path}")
                digest = port.file_sha256(path)
                identities.append((path, digest))
                setattr(args, name, path)
                report[name] = dict(path=str(path), sha256=digest)
            with tempfile.TemporaryDirectory(prefix="patch-ma-", dir=args.out.parent) as folder:
                temporary = Path(folder)
                sources = load_cases(args)
                if not sources:
                    raise ValueError("Empty corpus")
                for index, source in enumerate(sources):
                    case = dict(source=str(source), variants={v: [] for v in args.variants},
                                diagnostics={v: [] for v in args.variants}, summaries={}, execution_order=[])
                    report["cases"].append(case)
                    try:
                        original = source.read_bytes()
                        digest = port.sha256(original)
                        identities.append((source, digest))
                        case["sha256"] = digest
                        payload, expected, preparation = prepare_png(source, original, temporary)
                        _, info = png_framing(original)
                        info["mode"] = preparation
                        snapshot = temporary / "input.png"
                        snapshot.write_bytes(payload)
                        pixels = expected.width * expected.height
                        case.update(pixels=pixels, width=expected.width, height=expected.height,
                                    depth=expected.depth, preparation=info, snapshot_sha256=port.sha256(payload))
                    except Exception as error:
                        case["error"] = f"{type(error).__name__}: {error}"
                        report["errors"].append(f"{source}: {case['error']}")
                        continue
                    rounds = [(r, False) for r in range(-args.warmups, args.repeats)]
                    if args.profile:
                        rounds.append((args.repeats, True))
                    for repetition, diagnostic in rounds:
                        for variant in trial_order(args.variants, repetition, index):
                            sample = dict(repetition=repetition, warmup=repetition < 0,
                                          included_in_timing=not diagnostic and repetition >= 0, verified=False)
                            case["diagnostics" if diagnostic else "variants"][variant].append(sample)
                            case["execution_order"].append(dict(variant=variant, repetition=repetition,
                                                                 diagnostic=diagnostic))
                            try:
                                run_trial(args, variant, snapshot, expected, temporary, env, sample, diagnostic)
                            except Exception as error:
                                sample["error"] = f"{type(error).__name__}: {error}"
                                report["errors"].append(f"{source} {variant} r={repetition}: {sample['error']}")
                    for variant, samples in case["variants"].items():
                        if all(s["verified"] for s in samples):
                            case["summaries"][variant] = summarize(samples, pixels)
                    print(f"{source.name}: {len(case['summaries'])}/{len(args.variants)} routes complete", flush=True)
                    report["completed"] = len(report["cases"])
                    output.seek(0)
                    json.dump(report, output, indent=2, default=str, allow_nan=False)
                    output.truncate()
                    output.flush()
                report["aggregate"] = {}
                for variant in args.variants:
                    if all(variant in c["summaries"] for c in report["cases"]):
                        report["aggregate"].update(source_bench.aggregate(report["cases"], [variant]))
                    else:
                        report["aggregate"][variant] = dict(status="incomplete", reason="Failed/missing case or trial; no partial aggregate")
            report["status"] = "passed" if not report["errors"] else "failed"
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
            output.seek(0)
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
            output.truncate()
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
