"""Serial, rotated LZ77 heuristic comparison using an existing encoder_lab.

Native JSON encode seconds exclude process startup and disk I/O. Every trial,
including warmups, is checked against original PNG samples by official djxl.
No builds, discovery, size fallback, or metadata acceptance.
Manifest format: {"cases": [{"original": "path/to/input.png", "notes": "..."}]}
(paths relative to manifest). This measures the complete native fixed-plan encode,
not production scheduler/search performance. All modes use the same --lz value
(default 0: source effort-dependent route, not disabled matching).
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from benchmark_fast_context import add_ratios, load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process, sha256,
)


DEFAULT_MODES = ("baseline", "observe", "route", "sparse", "double-fast", "double-fast-sparse")
MODES = (*DEFAULT_MODES, "double-fast-cost")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", "--official-djxl", dest="decoder", type=Path, required=True,
                        help="Caller-provided official djxl; provenance recorded, not inferred")
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=MODES, action="append",
                        help="Repeat to select modes; include baseline; default original six; "
                             "double-fast-cost is explicit-only")
    parser.add_argument("--effort", type=int, choices=range(1, 11), default=9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lz", type=int, choices=(0, 1, 2), default=0,
                        help="Shared source route: 0 effort-dependent default, 1 fast, 2 Optc8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.mode = list(DEFAULT_MODES) if args.mode is None else args.mode
    if not args.case and args.manifest is None:
        parser.error("Provide --case or --manifest")
    if "baseline" not in args.mode or len(set(args.mode)) != len(args.mode):
        parser.error("Select distinct modes including baseline")
    if (not 1 <= args.threads <= 256 or args.repeats < 1 or args.warmups < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Invalid threads/repeats/warmups/timeout")
    if args.out.exists():
        parser.error("Report exists; refusing overwrite")
    return args


def command(args, source, output, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    result = [str(args.encoder), str(source), "--effort", str(args.effort),
              "--threads", str(args.threads), "--lz", str(args.lz), "--output", str(output)]
    if mode in ("observe", "route"):
        result += ["--lz-scout", mode]
    elif mode != "baseline":
        result += ["--lz-quick", mode]
    return result


def validate_measurement(measurement, encoded_size, expected, lz=0):
    if not isinstance(measurement, dict):
        raise ValueError("Encoder stdout must be a JSON object")
    seconds, size = measurement.get("seconds"), measurement.get("bytes")
    if not positive_number(seconds):
        raise ValueError("Missing/invalid native encode seconds; wall time is never substituted")
    if type(size) is not int or size <= 0 or size != encoded_size:
        raise ValueError("Reported bytes differ from actual nonempty output")
    if measurement.get("search") is not False:
        raise ValueError("Expected direct candidate search=false")
    if type(measurement.get("lz")) is not int or measurement["lz"] != lz:
        raise ValueError("Native LZ route differs from shared requested route")
    for key, value in (("width", expected.width), ("height", expected.height)):
        if type(measurement.get(key)) is not int or measurement[key] != value:
            raise ValueError(f"Invalid native {key}")
    pixels = expected.width * expected.height
    for key, value in (("bpp", size * 8 / pixels),
                       ("mpix_per_second", pixels / 1e6 / seconds)):
        actual = measurement.get(key)
        if not positive_number(actual) or not math.isclose(actual, value, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"Missing/inconsistent {key}")


def validate_activation(measurement, mode):
    """Validate work receipts, distinguishing intended careful bypass from activation."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    for key in ("lz_scout", "lz_quick"):
        if key not in measurement:
            raise ValueError(f"Missing {key} receipt; silent no-op modes are not accepted")
    scout, quick = measurement["lz_scout"], measurement["lz_quick"]
    if mode == "baseline":
        if scout is not None or quick is not None:
            raise ValueError("Baseline unexpectedly enabled an LZ experiment")
        return "baseline"
    if mode in ("observe", "route"):
        if quick is not None or not isinstance(scout, dict) or scout.get("mode") != mode:
            raise ValueError("Scout receipt differs from requested mode")
        calls = scout.get("calls")
        if type(calls) is not int or calls <= 0:
            raise ValueError("Scout mode did no observable work on nonempty input")
        return "scout-active"
    if scout is not None or not isinstance(quick, dict) or quick.get("mode") != mode:
        raise ValueError("Quick receipt differs from requested mode")
    for key in ("calls", "queries", "query_misses", "useful_matches", "skipped_positions"):
        value = quick.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid quick counter: {key}")
    if mode == "double-fast-cost":
        for key in ("cost_ranked_calls", "preserved_careful_calls"):
            value = quick.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid quick counter: {key}")
        if quick["cost_ranked_calls"] != quick["calls"]:
            raise ValueError("Cost-ranked calls differ from actual quick calls")
        if quick["calls"] == 0 and quick["preserved_careful_calls"] > 0:
            return "preserved-careful-bypass"
    if quick["calls"] == 0:
        raise ValueError("Quick mode did no work on nonempty input")
    return "quick-active"


def aggregate(cases, modes):
    """Sum per-file medians, never choose a smaller baseline artifact."""
    pixels = sum(case["pixels"] for case in cases)
    result = {}
    for mode in modes:
        rows = [case["summaries"][mode] for case in cases]
        size = sum(row["bytes"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        deltas = [case["summaries"][mode]["bytes"] - case["summaries"]["baseline"]["bytes"]
                  for case in cases]
        result[mode] = dict(files=len(cases), pixels=pixels, bytes=size,
                            weighted_bpp=8 * size / pixels, seconds=seconds,
                            mpix=pixels / 1e6, mpix_per_second=pixels / 1e6 / seconds,
                            size_regressions=sum(delta > 0 for delta in deltas),
                            size_improvements=sum(delta < 0 for delta in deltas),
                            size_ties=sum(delta == 0 for delta in deltas),
                            byte_delta=sum(deltas))
    return result


def case_notes(args):
    if args.manifest is None:
        return {}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {}
    for case in manifest["cases"]:
        path = Path(case["original"])
        path = path if path.is_absolute() else args.manifest.resolve().parent / path
        if "notes" in case:
            result[path.resolve()] = case["notes"]
    return result


def benchmark(args):
    report = dict(schema_version=1, status="failed", config=vars(args).copy(),
                  created_utc=datetime.now(timezone.utc).isoformat(), cases=[], errors=[],
                  timing="Serial rotated order; per-file median native JSON seconds; warmups excluded",
                  correctness="Official djxl vs original decoded PNG, exact depth and hidden RGBA",
                  metadata="Original PNG snapshot unchanged; metadata preservation is not tested",
                  decoder_provenance="Caller-attested official djxl; path and SHA256 recorded",
                  diagnostics="Requested mode and inactive counterpart null validated; calls>0 except "
                              "double-fast-cost all-careful receipts (calls=cost_ranked_calls=0, "
                              "preserved_careful_calls>0): intended bypass, NOT quick activation; "
                              "full snapshots and per-trial experiment_status retained",
                  scope="Complete native fixed-plan encode at requested effort/shared --lz; NOT production scheduler/search",
                  size_policy="No fallback, pruning, or omission of regressions",
                  aggregate_policy="All cases required; sums of per-file medians; pixel-weighted bpp")
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
                identities.append((args.manifest, file_sha256(args.manifest)))
            sources = load_cases(args)
            notes = case_notes(args)
            with tempfile.TemporaryDirectory(prefix="lz-scout-") as scratch:
                temporary = Path(scratch)
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = sha256(original)
                    identities.append((source, digest))
                    expected = png_samples(original)
                    pixels = expected.width * expected.height
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(original)
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                notes=notes.get(source), plan=f"fixed candidate, --lz {args.lz}, no search",
                                width=expected.width, height=expected.height, depth=expected.depth,
                                mpix=pixels / 1e6, modes={mode: [] for mode in args.mode}, summaries={})
                    report["cases"].append(case)
                    for repetition in range(-args.warmups, args.repeats):
                        for mode in trial_order(args.mode, repetition, index):
                            sample = dict(repetition=repetition, warmup=repetition < 0,
                                          pixels_verified=False, verified=False)
                            case["modes"][mode].append(sample)
                            try:
                                with tempfile.TemporaryDirectory(prefix="trial-", dir=scratch) as trial:
                                    jxl, png = Path(trial) / "out.jxl", Path(trial) / "out.png"
                                    sample["encoder"] = run_process(command(args, snapshot, jxl, mode), env, args.timeout)
                                    check_process(sample["encoder"])
                                    sample["decoder"] = run_process(decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
                                    check_process(sample["decoder"])
                                    compare_pixels(expected, png_samples(png.read_bytes()))
                                    sample["pixels_verified"] = True
                                    measurement = json.loads(sample["encoder"]["stdout"])
                                    sample["measurement"] = measurement
                                    validate_measurement(measurement, jxl.stat().st_size, expected, args.lz)
                                    sample["experiment_status"] = validate_activation(measurement, mode)
                                    sample["verified"] = True
                            except Exception as error:
                                sample["error"] = f"{type(error).__name__}: {error}"
                                report["errors"].append(f"{source} {mode} repeat={repetition}: {sample['error']}")
                            print(f"{source.name} {mode} repeat={repetition}: "
                                  f"{'verified' if sample['verified'] else sample['error']}", flush=True)
                    for mode, samples in case["modes"].items():
                        if all(sample["verified"] for sample in samples):
                            case["summaries"][mode] = summarize(samples, pixels)
                            case["summaries"][mode]["mpix"] = pixels / 1e6
                    if len(case["summaries"]) == len(args.mode):
                        add_ratios(case["summaries"])
                        for summary in case["summaries"].values():
                            summary["byte_delta"] = summary["bytes"] - case["summaries"]["baseline"]["bytes"]
                            summary["size_regression"] = summary["byte_delta"] > 0
            if not report["errors"]:
                report["aggregate"] = aggregate(report["cases"], args.mode)
                report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if file_sha256(path) != digest:
                        report["errors"].append(f"Input/binary changed during run: {path}")
                except OSError as error:
                    report["errors"].append(str(error))
            if report["errors"] or not report["temporary_outputs_cleaned"]:
                report["status"] = "failed"
            json.dump(report, output, indent=2, default=str, allow_nan=False)
            output.write("\n")
    return report


def main(argv=None):
    try:
        report = benchmark(arguments(argv))
    except (OSError, ValueError) as error:
        print(f"Report failure: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(error, file=sys.stderr)
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
