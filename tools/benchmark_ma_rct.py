"""Bounded raw MA lookahead experiments, independently verified by official djxl.

Prebuilt native labs only: no builds, downloads, automatic corpus discovery,
crop encodes, min-byte fallback, or hidden greedy selection. baseline means raw
greedy (a naming alias required by plot_patch_tradeoffs schema 1). Local MA score
receipts are estimates, not emitted model bits. Native timing excludes all IO.
Optional RCT joint is timed at TOTAL native search cost, including losing trials.
Only its exported winner is pixel-verified; candidate rows are unverified native
statistics, never separate verified plot points or decoder acceptance claims.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

from benchmark_fast_context import load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
from benchmark_palette_runs import prepare_png
from benchmark_tile_patches import add_comparisons, aggregate
from corpus_inputs import png_framing
import benchmark_qlic_candidates as qlic
import compare_cpp_port as port


MA_MODES = ("baseline", "lookahead2", "lookahead4")
MODES = (*MA_MODES, "rct-joint")
COUNTERS = ("roots", "eligible", "probed_candidates", "child_searches", "changed")
TIMING = "Native discovery+admission+encode only; excludes startup and input/output IO"
SCORE_CONTRACT = "estimated local scoring only; not emitted wire bits"
SYNTHETIC_NAMES = (
    "tiles-8-2.png", "tiles-8-6.png", "tiles-16-2.png", "tiles-16-6.png",
    "mixed-direction-8-rgba.png", "mixed-direction-16-rgba.png",
)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True, help="Prebuilt ma_lookahead_lab")
    p.add_argument("--rct-encoder", type=Path, help="Optional prebuilt rct_joint_lab")
    p.add_argument("--decoder", "--official-djxl", "--djxl", dest="decoder", type=Path,
                   required=True, help="Caller-attested official djxl executable")
    p.add_argument("--case", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--synthetic", action="store_true", help="Six fixed RGB/RGBA 8/16-bit controls")
    p.add_argument("--modes", nargs="+", choices=(*MODES, "greedy"))
    p.add_argument("--effort", type=int, choices=range(4, 11), default=9)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tree-learning", choices=("legacy", "cost"), default="legacy")
    p.add_argument("--min-records", type=int, default=8192)
    p.add_argument("--relative-window", type=float, default=0.01)
    p.add_argument("--shortlist", type=int, choices=range(1, 9), default=3)
    p.add_argument("--sample-pixels", type=int, choices=range(9, 65537), default=8192)
    p.add_argument("--max-cases", type=int, default=16, help="Hard admission cap; never silently truncate")
    p.add_argument("--max-pixels", type=int, default=1048576, help="Hard per-case cap; no resizing")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300, help="Harness process guard only")
    p.add_argument("--out", type=Path, required=True, help="New schema-1 report, never overwritten")
    a = p.parse_args(argv)
    if a.modes is None:
        a.modes = list(MODES if a.rct_encoder is not None else MA_MODES)
    a.modes = ["baseline" if m == "greedy" else m for m in a.modes]
    if (not 1 <= a.threads <= 256 or not 1 <= a.repeats <= 100 or not 0 <= a.warmups <= 100
            or not positive_number(a.timeout) or a.min_records < 4
            or not math.isfinite(a.relative_window) or not 0 <= a.relative_window <= 1
            or a.max_cases < 1 or a.max_pixels < 1):
        p.error("Invalid threads/repeats/warmups/timeout/lookahead/sample bounds")
    if len(set(a.modes)) != len(a.modes) or "baseline" not in a.modes:
        p.error("Use distinct modes including baseline (greedy is its alias)")
    if "rct-joint" in a.modes and (a.rct_encoder is None or a.effort != 9 or a.tree_learning != "legacy"):
        p.error("rct-joint requires --rct-encoder, effort 9 and legacy learning for a matched baseline")
    if not a.case and a.manifest is None and not a.synthetic:
        p.error("Provide --case, --manifest or --synthetic")
    if a.out.exists():
        p.error("Report already exists; it will not be overwritten")
    return a


def native_mode(mode):
    if mode not in MA_MODES:
        raise ValueError(f"Unknown MA variant: {mode}")
    return "greedy" if mode == "baseline" else mode


def command(args, source, output, mode):
    if mode == "rct-joint":
        return [str(args.rct_encoder), str(source), "--output", str(output),
                "--threads", str(args.threads), "--shortlist", str(args.shortlist),
                "--sample-pixels", str(args.sample_pixels)]
    cmd = [str(args.encoder), str(source), "--output", str(output),
           "--effort", str(args.effort), "--threads", str(args.threads),
           "--mode", native_mode(mode), "--tree-learning", args.tree_learning]
    if mode != "baseline":
        cmd += ["--min-records", str(args.min_records), "--relative-window", str(args.relative_window)]
    return cmd


def fixtures(folder):
    paths = {p.name: p for p in qlic.fixtures(folder)}
    return [paths[name] for name in SYNTHETIC_NAMES]


def validate_measurement(m, mode, size, width, height, args):
    if not isinstance(m, dict) or not positive_number(m.get("seconds")):
        raise ValueError("Invalid native duration")
    if type(m.get("bytes")) is not int or m["bytes"] != size or size <= 0:
        raise ValueError("Reported bytes differ from actual nonempty output")
    if (any(type(m.get(k)) is not int for k in ("width", "height"))
            or (m["width"], m["height"]) != (width, height) or width * height <= 0):
        raise ValueError("Invalid measurement dimensions")
    for key, expected in (("bpp", size * 8 / (width * height)),
                          ("mpix_per_second", width * height / 1e6 / m["seconds"])):
        if not positive_number(m.get(key)) or not math.isclose(m[key], expected, rel_tol=1e-5):
            raise ValueError(f"Invalid actual-output metric: {key}")
    if (m.get("mode") != native_mode(mode) or m.get("tree_learning") != args.tree_learning
            or type(m.get("native_encode_calls")) is not int or m["native_encode_calls"] != 1):
        raise ValueError("Mode/learning/raw single-encode receipt mismatch")
    if (type(m.get("effort")) is not int or m["effort"] != args.effort
            or type(m.get("threads")) is not int or m["threads"] != args.threads
            or type(m.get("previous_channels")) is not int or m["previous_channels"] != 0):
        raise ValueError("Mismatched common source-projected baseline policy")
    if m.get("patches") != "Off" or m.get("patches_cleared") is not True:
        raise ValueError("MA experiment requires patches Off and all detectors cleared")
    if m.get("lookahead_score_contract") != SCORE_CONTRACT:
        raise ValueError("Missing local-estimate-only scoring contract")
    if "lookahead" not in m or "lookahead_options" not in m:
        raise ValueError("Missing nullable lookahead receipt/options")
    work, settings = m["lookahead"], m["lookahead_options"]
    if mode == "baseline":
        if work is not None or settings is not None:
            raise ValueError("Greedy baseline unexpectedly enabled lookahead")
        return
    width = int(mode[-1])
    if (not isinstance(settings, dict) or type(settings.get("width")) is not int
            or settings["width"] != width or type(settings.get("min_records")) is not int
            or settings["min_records"] != args.min_records
            or not positive_or_zero(settings.get("relative_window"))
            or not math.isclose(settings["relative_window"], args.relative_window,
                                rel_tol=1e-6, abs_tol=1e-45)):
        raise ValueError("Lookahead settings differ from requested experiment")
    if not isinstance(work, dict) or any(type(work.get(k)) is not int or work[k] < 0 for k in COUNTERS):
        raise ValueError("Invalid lookahead work counters")
    if not positive_or_zero(work.get("estimated_saved_bits")):
        raise ValueError("Invalid local estimated saved bits")
    if (work["eligible"] > work["roots"] or work["changed"] > work["eligible"]
            or work["probed_candidates"] > width * work["eligible"]
            or work["child_searches"] > 2 * work["probed_candidates"]):
        raise ValueError("Inconsistent bounded root-lookahead counters")
    # Palette/constant shortcuts or min_records can legitimately do zero work.


def positive_or_zero(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def rct_measurement(receipt, size, width, height, args):
    """Normalize only the exported winner; retain all unverified native rows."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "rs-jxl.rct-joint-lab.v1":
        raise ValueError("Unsupported RCT receipt schema")
    for key, expected in (("width", width), ("height", height), ("effort", 9),
                          ("threads", args.threads), ("shortlist_limit", args.shortlist),
                          ("sample_pixel_limit", args.sample_pixels), ("baseline_trial", 0)):
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            raise ValueError(f"Invalid RCT configuration: {key}")
    if (receipt.get("selection") != "complete_standard_jxl_bytes_first_minimum"
            or receipt.get("decoder_policy") != "unchanged"
            or receipt.get("patch_policy") != "disabled"
            or receipt.get("baseline_policy") != "native_e9_patches_off"
            or receipt.get("auto_rct_control_trials") != [1, 2, 3]
            or type(receipt.get("winner_bytes")) is not int
            or receipt["winner_bytes"] != size or size <= 0):
        raise ValueError("Invalid RCT selection/actual output size")
    screen, timing, trials = (receipt.get(k) for k in ("screen", "timing", "trials"))
    if not isinstance(screen, dict) or screen.get("real_ma_training") is not False:
        raise ValueError("Missing RCT proxy screen receipt")
    shortlisted, scores = screen.get("shortlisted_rcts"), screen.get("scores")
    if (not isinstance(shortlisted, list) or not 1 <= len(shortlisted) <= args.shortlist
            or any(type(r) is not int or not 0 <= r < 42 for r in shortlisted)
            or len(set(shortlisted)) != len(shortlisted)
            or not isinstance(scores, list) or len(scores) != 42
            or type(screen.get("sampled_pixels")) is not int
            or not 1 <= screen["sampled_pixels"] <= args.sample_pixels):
        raise ValueError("Incomplete/unbounded all42 RCT screen")
    for index, score in enumerate(scores):
        if (not isinstance(score, dict) or type(score.get("rct")) is not int or score["rct"] != index
                or not positive_or_zero(score.get("native_bits"))
                or not positive_or_zero(score.get("seconds"))
                or not isinstance(score.get("proxy_bits"), list) or len(score["proxy_bits"]) != 3
                or not all(positive_or_zero(n) for n in score["proxy_bits"])):
            raise ValueError("Invalid RCT screen candidate statistics")
    if not isinstance(trials, list) or len(trials) != 4 + 3 * len(shortlisted):
        raise ValueError("Missing RCT control/trial rows")
    actual, seconds, successful = 0, 0.0, []
    for index, trial in enumerate(trials):
        kind = "native_e9_baseline" if index == 0 else "auto_rct_control" if index < 4 else "shortlisted_rct"
        previous = 0 if index == 0 else (index - 1) % 3
        rct = None if index < 4 else shortlisted[(index - 4) // 3]
        if (not isinstance(trial, dict) or type(trial.get("index")) is not int
                or trial["index"] != index or trial.get("kind") != kind):
            raise ValueError("Invalid RCT trial index/kind")
        params, effective = trial.get("params"), trial.get("effective")
        if (not isinstance(params, dict) or params.get("max_properties") != previous
                or params.get("rct") != rct or params.get("speed_tier") != 1
                or params.get("patches") != "Off"
                or not isinstance(effective, dict) or effective.get("previous_channels") != previous
                or effective.get("training_previous_channels") != previous
                or effective.get("patches_disabled") is not True):
            raise ValueError("RCT trial does not match joint previous_channels=0/1/2 plan")
        error, count, elapsed, reused = (trial.get(k) for k in ("error", "bytes", "seconds", "reused_from"))
        if not positive_or_zero(elapsed):
            raise ValueError("Invalid RCT candidate native timing")
        if error is None:
            if type(count) is not int or count <= 0:
                raise ValueError("Invalid successful RCT trial byte count")
            successful.append((count, index))
        elif not isinstance(error, str) or not error or count is not None:
            raise ValueError("Invalid RCT candidate error statistics")
        if reused is None:
            actual += 1
            if not positive_number(elapsed):
                raise ValueError("Actual RCT encodes require positive native time")
        elif (type(reused) is not int or not 0 <= reused < index or elapsed != 0
              or trials[reused]["bytes"] != count or trials[reused]["error"] != error
              or trials[reused]["params"] != params):
            raise ValueError("Invalid RCT reuse alias")
        seconds += elapsed
    winner = receipt.get("winner_trial")
    if (not successful or type(winner) is not int or winner != min(successful)[1]
            or trials[winner]["bytes"] != size
            or type(receipt.get("actual_encodes")) is not int or receipt["actual_encodes"] != actual):
        raise ValueError("RCT winner/encode accounting mismatch")
    if not isinstance(timing, dict) or any(not positive_or_zero(timing.get(k)) for k in
            ("total_seconds", "screen_seconds", "encode_seconds", "overhead_seconds", "pool_setup_seconds")):
        raise ValueError("Invalid total native RCT search timing")
    total = timing["total_seconds"]
    if (not positive_number(total) or not math.isclose(timing["encode_seconds"], seconds, rel_tol=1e-6)
            or not math.isclose(total, sum(timing[k] for k in
                ("screen_seconds", "encode_seconds", "overhead_seconds")), rel_tol=1e-6, abs_tol=1e-8)
            or not positive_or_zero(screen.get("seconds"))
            or not math.isclose(screen["seconds"], timing["screen_seconds"], rel_tol=1e-6)
            or timing["pool_setup_seconds"] > timing["overhead_seconds"] + 1e-8):
        raise ValueError("RCT search omitted or double-counted native work")
    return dict(bytes=size, seconds=total, bpp=8 * size / (width * height),
                mpix_per_second=width * height / 1e6 / total, width=width, height=height,
                mode="rct-joint", winner_trial=winner, native_encode_calls=actual,
                rct_joint=receipt, all_candidates_verified=False,
                sample_pixel_contract="Sample records, not distinct pixels; spatial tiles may overlap; bounded by sample_pixel_limit")


def native_json(value):
    def invalid(value):
        raise ValueError(f"Nonfinite native JSON number: {value}")

    def finite(value):
        number = float(value)
        return number if math.isfinite(number) else invalid(value)

    return json.loads(value, parse_constant=invalid, parse_float=finite)


def summarize_rct(samples):
    """Diagnostic candidate medians, NOT independently pixel-verified points."""
    groups, winners = {}, []
    for sample in samples:
        if sample["warmup"]:
            continue
        receipt = sample["measurement"]["rct_joint"]
        winners.append(receipt["winner_trial"])
        for trial in receipt["trials"]:
            params = trial["params"]
            key = f"{trial['kind']}:rct={params['rct']}:previous_channels={params['max_properties']}"
            group = groups.setdefault(key, dict(kind=trial["kind"], rct=params["rct"],
                                               previous_channels=params["max_properties"], rows=[]))
            group["rows"].append(trial)
    summaries = []
    for group in groups.values():
        rows = group.pop("rows")
        sizes = [row["bytes"] for row in rows if row["error"] is None]
        summaries.append(dict(group, samples=len(rows),
                              bytes_median=statistics.median(sizes) if sizes else None,
                              native_encode_seconds_median=statistics.median(row["seconds"] for row in rows),
                              errors=[row["error"] for row in rows if row["error"] is not None],
                              reused_rows=sum(row["reused_from"] is not None for row in rows),
                              independently_verified=False))
    return dict(winner_trials=winners, controls_and_trials=summaries,
                all_candidates_verified=False,
                timing="Candidate encode-only diagnostics; chart uses TOTAL native search seconds")


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(), cases=[], errors=[],
                  experiment="ma_root_lookahead_and_optional_rct_joint", mode_aliases={"baseline": "greedy"},
                  timing=TIMING, score_contract=SCORE_CONTRACT,
                  correctness="Official djxl vs original 8/16-bit samples including hidden RGB",
                  official_decoder_provenance="Caller-attested; executable path and SHA256 recorded",
                  regression_policy="Size/time regressions reported, not failures",
                  ratio_contract="mode / baseline; aggregate uses sums of per-case medians",
                  baseline_policy="source e9, previous_channels=0, patches explicitly Off in both labs",
                  selection_policy="MA: raw independent variants; RCT: explicit minimum with all search cost charged",
                  chart_time_label="Total native time (s); RCT includes all screening, trials and overhead",
                  candidate_verification="Only exported outputs verified; RCT losing/control rows are unverified statistics")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        try:
            env, report["environment"] = port.clean_environment(os.environ)
            for name in ("encoder", "decoder", *(("rct_encoder",) if "rct-joint" in args.modes else ())):
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
            with tempfile.TemporaryDirectory(prefix="ma-rct-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                sources = load_cases(args) if args.case or args.manifest else []
                external = set(sources)
                if args.synthetic:
                    sources += fixtures(temporary / "synthetic")
                if not sources or len(sources) > args.max_cases:
                    raise ValueError("Case count exceeds --max-cases (no cases silently skipped)")
                # Preflight the whole sample before any native timing or pixel allocation.
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
                    if info["bit_depth"] not in (8, 16):
                        raise ValueError(f"This experiment requires original 8/16-bit PNG: {source}")
                    if "rct-joint" in args.modes and info["color_type"] not in (2, 6):
                        raise ValueError(f"Matched RCT cases must be original RGB/RGBA: {source}")
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    if digest != input_hashes[source]:
                        raise ValueError(f"Input changed after sample admission: {source}")
                    payload, expected, preparation = prepare_png(source, original, temporary)
                    _, info = png_framing(original)
                    info["mode"] = preparation
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(payload)
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                preparation=info, snapshot_sha256=port.sha256(payload),
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
                                receipt = native_json(sample["encoder"]["stdout"])
                                if mode == "rct-joint":
                                    measurement = rct_measurement(receipt, jxl.stat().st_size,
                                                                  expected.width, expected.height, args)
                                else:
                                    measurement = receipt
                                    validate_measurement(measurement, mode, jxl.stat().st_size,
                                                         expected.width, expected.height, args)
                                sample["measurement"] = measurement
                                sample["decoder"] = port.run_process(
                                    port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                port.check_process(sample["decoder"])
                                port.compare_pixels(expected, port.png_samples(png.read_bytes()))
                                sample["verified"] = True
                                if mode == "rct-joint":
                                    sample.update(verification_scope="exported_winner_only",
                                                  verified_candidate_indices=[receipt["winner_trial"]],
                                                  all_candidates_verified=False)
                            print(f"{source.name} {mode} repeat={repeat}: "
                                  f"{measurement['bytes']} bytes {measurement['seconds']:.6f}s verified", flush=True)
                    case["summaries"] = {mode: summarize(samples, pixels) for mode, samples in case["variants"].items()}
                    add_comparisons(case["summaries"])
                    if "rct-joint" in args.modes:
                        case["rct_diagnostics"] = summarize_rct(case["variants"]["rct-joint"])
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
