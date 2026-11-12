"""Bounded direct static-context experiments, verified by caller-supplied official djxl.

Prebuilt binaries only: no builds, discovery, resizing or output-size selection.
Native time includes projection, analysis and encoding, excluding startup and IO.
Optional model costs measure emitted tree/header bits, never local score estimates.
Activity refine/repair modes retain the effort's native MA predictor/property policy.
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
from benchmark_ma_rct import fixtures, native_json
from benchmark_palette_runs import prepare_png
from benchmark_tile_patches import add_comparisons, aggregate
from corpus_inputs import png_framing
import compare_cpp_port as port


DEFAULT_MODES = ("baseline", "auto12", "activity25", "activity81")
SEED_MODES = ("activity25-refine", "activity81-refine", "activity81-repair")
MODES = (*DEFAULT_MODES, *SEED_MODES)
COUNTERS = ("trees", "leaves", "pixels", "offset_leaves", "horizontal", "vertical",
            "single", "joint", "cross_channel_plans")
SEED_COUNTERS = ("learners", "prepared_records", "seed_splits", "refined_leaves",
                 "repaired_branches", "merged_branches", "predictor_changes", "histogram_reuses")
# Schema 1's plotting contract; precise native scope is recorded separately.
TIMING = "Native discovery+admission+encode only; excludes startup and input/output IO"
NATIVE_TIMING = "native projection+analysis+encode; excludes startup and input/output IO"
MODEL_COST_CONTRACT = "emitted MA tree and entropy-header bits only; not estimated scores"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True, help="Prebuilt static_context_lab")
    p.add_argument("--decoder", "--official-djxl", "--djxl", dest="decoder", type=Path,
                   required=True, help="Caller-attested official djxl executable")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true", help="Six fixed RGB/RGBA 8/16-bit controls")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(DEFAULT_MODES))
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model-costs", action="store_true", help="Record emitted MA tree/header bits")
    p.add_argument("--max-cases", type=int, default=32, help="Hard admission cap, at most 32")
    p.add_argument("--max-pixels", type=int, default=1048576, help="Per-case admission cap; never resize")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300, help="Harness process guard only")
    p.add_argument("--out", type=Path, required=True, help="New schema-1 report, never overwritten")
    a = p.parse_args(argv)
    if (not 1 <= a.threads <= 256 or not 1 <= a.repeats <= 100 or not 0 <= a.warmups <= 100
            or not positive_number(a.timeout) or not 1 <= a.max_cases <= 32 or a.max_pixels < 1):
        p.error("Invalid threads/repeats/warmups/timeout/case/pixel bounds")
    if len(set(a.modes)) != len(a.modes) or "baseline" not in a.modes:
        p.error("Use distinct modes including baseline")
    if not a.case and a.manifest is None and not a.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if a.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return a


def command(args, source, output, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown static-context mode: {mode}")
    cmd = [str(args.encoder), str(source), "--output", str(output),
           "--effort", str(args.effort), "--threads", str(args.threads), "--mode", mode]
    if args.model_costs:
        cmd.append("--model-costs")
    return cmd


def seed_parameters(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown static-context mode: {mode}")
    if mode not in SEED_MODES:
        return None
    topology, seed_mode = mode.split("-")
    # Depth is reported for reproducibility, but fixed activity grids ignore it.
    return dict(topology=topology, mode=seed_mode, depth=3)


def validate_measurement(m, mode, size, width, height, args, wall_seconds=None):
    if not isinstance(m, dict) or not positive_number(m.get("seconds")):
        raise ValueError("Invalid native duration")
    if wall_seconds is not None and (not positive_number(wall_seconds)
            or m["seconds"] > wall_seconds + max(0.001, wall_seconds * 0.01)):
        raise ValueError("Native duration exceeds observed process wall time")
    if type(m.get("bytes")) is not int or m["bytes"] != size or size <= 0:
        raise ValueError("Reported bytes differ from actual nonempty output")
    if (any(type(m.get(k)) is not int for k in ("width", "height"))
            or (m["width"], m["height"]) != (width, height) or width * height <= 0):
        raise ValueError("Invalid measurement dimensions")
    for key, expected in (("bpp", size * 8 / (width * height)),
                          ("mpix_per_second", width * height / 1e6 / m["seconds"])):
        if not positive_number(m.get(key)) or not math.isclose(m[key], expected, rel_tol=1e-5):
            raise ValueError(f"Invalid actual-output metric: {key}")
    if (mode not in MODES or m.get("mode") != mode or m.get("search") is not False
            or type(m.get("native_encode_calls")) is not int or m["native_encode_calls"] != 1
            or m.get("timing") != NATIVE_TIMING):
        raise ValueError("Mode/timing/raw single-encode receipt mismatch")
    for key, expected in (("effort", args.effort), ("threads", args.threads), ("previous_channels", 0)):
        if type(m.get(key)) is not int or m[key] != expected:
            raise ValueError(f"Mismatched direct source-projected policy: {key}")
    if m.get("patches") != "Off" or m.get("patches_cleared") is not True:
        raise ValueError("Static-context experiment requires patches Off and all detectors cleared")
    if m.get("model_cost_contract") != MODEL_COST_CONTRACT or "model_costs" not in m:
        raise ValueError("Missing emitted-model-only cost contract/receipt")
    costs = m["model_costs"]
    if args.model_costs:
        if not isinstance(costs, dict) or any(
                type(costs.get(k)) is not int or costs[k] < 0
                for k in ("tree_bits", "entropy_header_bits", "total_bits")):
            raise ValueError("Invalid emitted model cost receipt")
        if (costs["total_bits"] != costs["tree_bits"] + costs["entropy_header_bits"]
                or costs["total_bits"] > size * 8):
            raise ValueError("Model costs do not sum or exceed actual codestream bits")
    elif costs is not None:
        raise ValueError("Unrequested model cost receipt")
    if "fast_context" not in m:
        raise ValueError("Missing nullable fast-context receipt")
    work = m["fast_context"]
    settings = seed_parameters(mode)
    if settings is not None:
        if work is not None:
            raise ValueError("Seed refinement must not enable the restricted fast-context learner")
        actual = m.get("seed_options")
        if (not isinstance(actual, dict) or type(actual.get("depth")) is not int
                or any(actual.get(k) != v for k, v in settings.items())):
            raise ValueError("Activity seed settings differ from requested experiment")
        seed = m.get("seed")
        if not isinstance(seed, dict) or any(type(seed.get(k)) is not int or seed[k] < 0 for k in SEED_COUNTERS):
            raise ValueError("Invalid activity seed work counters")
        return
    # Original static receipts may predate the nullable seed fields.
    if m.get("seed") is not None or m.get("seed_options") is not None:
        raise ValueError("Static/baseline mode unexpectedly enabled seed refinement")
    if mode == "baseline":
        if work is not None:
            raise ValueError("Baseline unexpectedly enabled fast context")
        return
    if (not isinstance(work, dict) or work.get("mode") != mode or work.get("bias") != "off"
            or work.get("scan") != "auto" or work.get("predictors") != "basic"
            or work.get("cross_channel") is not False):
        raise ValueError("Fast-context settings differ from requested default policy")
    if any(type(work.get(k)) is not int or work[k] < 0 for k in COUNTERS):
        raise ValueError("Invalid fast-context work counters")
    if work["offset_leaves"] != 0 or work["cross_channel_plans"] != 0:
        raise ValueError("Disabled offset/cross-channel tools reported work")
    if mode == "auto12" and work["joint"] != 0:
        raise ValueError("Auto12 unexpectedly used a joint activity plan")
    # Constant/palette shortcuts can legitimately bypass the learner entirely.
    # Direction counters count channel plans, not trees or original image pixels.
    if work["leaves"] < work["trees"]:
        raise ValueError("Fewer leaves than emitted fast-context trees")


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(), cases=[], errors=[],
                  experiment="static_activity_context", timing=TIMING, native_timing=NATIVE_TIMING,
                  model_cost_contract=MODEL_COST_CONTRACT,
                  correctness="Official djxl vs original PNG samples including hidden RGB; "
                              "packed grayscale/palette losslessly expanded to 8-bit; 16-bit preserved",
                  official_decoder_provenance="Caller-attested; executable path and SHA256 recorded",
                  baseline_policy="direct source backend; previous_channels=0; patches explicitly Off",
                  selection_policy="raw independent modes; one native encode per invocation; no fallback",
                  ratio_contract="mode / baseline; aggregate uses sums of per-case medians",
                  regression_policy="Size/time regressions reported, not failures",
                  timeout_contract="Harness process guard, not an encoder heuristic")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            for name in ("encoder", "decoder"):
                path = getattr(args, name).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Not an executable file: {path}")
                setattr(args, name, path)
                digest = port.file_sha256(path)
                identities.append((path, digest))
                report[name] = dict(path=str(path), sha256=digest)
            if args.manifest is not None:
                digest = port.file_sha256(args.manifest)
                identities.append((args.manifest, digest))
                report["manifest"] = dict(path=str(args.manifest.resolve()), sha256=digest)
            with tempfile.TemporaryDirectory(prefix="static-context-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                external = set(sources)
                if args.synthetic:
                    sources += fixtures(temporary / "synthetic")
                if not sources or len(sources) > args.max_cases:
                    raise ValueError("Case count exceeds --max-cases (no cases silently skipped)")
                # Admit the entire sample before pixel allocation or native invocations.
                input_hashes = {}
                for source in sources:
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    input_hashes[source] = digest
                    if source in external:
                        identities.append((source, digest))
                    _, info = png_framing(original)
                    if info["width"] * info["height"] > args.max_pixels:
                        raise ValueError(f"Input exceeds --max-pixels; no resizing: {source}")
                    # Framing validates legal depth/color combinations. prepare_png
                    # expands packed oracle samples only; native 16-bit stays intact.
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    if digest != input_hashes[source]:
                        raise ValueError(f"Input changed after sample admission: {source}")
                    payload, expected, preparation = prepare_png(source, original, temporary)
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(payload)
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=preparation, snapshot_sha256=port.sha256(payload),
                                variants={mode: [] for mode in args.modes}, summaries={})
                    report["cases"].append(case)
                    for repeat in range(-args.warmups, args.repeats):
                        for mode in trial_order(args.modes, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][mode].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                sample["encoder"] = port.run_process(command(args, snapshot, jxl, mode), env, args.timeout)
                                port.check_process(sample["encoder"])
                                # Check every successful output even if its JSON receipt is malformed.
                                sample["decoder"] = port.run_process(
                                    port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                port.check_process(sample["decoder"])
                                port.compare_pixels(expected, port.png_samples(png.read_bytes()))
                                measurement = native_json(sample["encoder"]["stdout"])
                                validate_measurement(measurement, mode, jxl.stat().st_size,
                                                     expected.width, expected.height, args,
                                                     sample["encoder"]["wall_seconds"])
                                sample.update(verified=True, measurement=measurement)
                            print(f"{source.name} {mode} repeat={repeat}: "
                                  f"{measurement['bytes']} bytes {measurement['seconds']:.6f}s verified", flush=True)
                    case["summaries"] = {mode: summarize(samples, pixels) for mode, samples in case["variants"].items()}
                    add_comparisons(case["summaries"])
                report["aggregate"] = aggregate(report["cases"], args.modes)
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
    args = arguments(argv)
    try:
        report = benchmark(args)
    except (OSError, ValueError) as error:
        print(f"Benchmark/report failure: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
