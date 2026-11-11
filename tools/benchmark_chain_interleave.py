"""Focused immutable-chain interleaving comparison using prebuilt encoder_lab.

Example: python tools/benchmark_chain_interleave.py --encoder lab.exe
--official-djxl djxl.exe --case selected.png --profile --out report.json

Use --family literal or --family cross6 for a fixed alternative candidate.

No builds or corpus discovery. Sources must be explicitly selected; official djxl
must reproduce every original PNG sample, including invisible RGB. --profile
is required and proves every mode's route before serial, order-rotated timings.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from benchmark_fast_context import load_cases, positive_number, summarize
from benchmark_ma_models import trial_order
import compare_cpp_port as port


VARIANTS = ("baseline", "batch1", "batch2", "batch4", "batch8")
NOTE_NAMES = ("lz77.source.route", "lz77.source.chain_batch", "lz77.source.chain_batch_work")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True, help="Prebuilt encoder_lab; never built here")
    parser.add_argument("--decoder", "--official-djxl", dest="decoder", type=Path, required=True,
                        help="Caller-attested official djxl; binary path and SHA256 recorded")
    parser.add_argument("--case", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, help="Selected cases with original PNG paths")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--effort", type=int, choices=(9, 10), default=9)
    parser.add_argument("--family", choices=("automatic", "literal", "cross6"), default="automatic",
                        help="Same fixed candidate plan in every mode; never outer search")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--warmups", type=int, choices=range(4), default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--profile", action="store_true", required=True,
                        help="Required route proof: one separate diagnostic encode per case/mode")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.case and args.manifest is None:
        parser.error("Provide --case or --manifest")
    if not 1 <= args.threads <= 256 or not positive_number(args.timeout):
        parser.error("Invalid threads/timeout")
    if (len(set(args.variants)) != len(args.variants)
            or not {"baseline", "batch1"}.issubset(args.variants)
            or not set(args.variants).intersection(("batch2", "batch4", "batch8"))):
        parser.error("Use distinct variants including baseline, batch1 and an interleaved mode")
    if args.out.exists():
        parser.error("Report already exists; it will not be overwritten")
    return args


def lanes_for(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return 1 if variant == "baseline" else int(variant.removeprefix("batch"))


def command(args, source, output, variant):
    lanes = lanes_for(variant)
    result = [str(args.encoder), str(source), "--effort", str(args.effort),
              "--threads", str(args.threads), "--family", args.family, "--output", str(output)]
    if variant != "baseline":
        # Isolate lanes from the later scheduling/lookahead improvements.
        result += ["--chain-batch", "--chain-batch-lanes", str(lanes),
                   "--chain-batch-legacy-schedule", "--chain-batch-fixed-window"]
    return result


def validate_measurement(measurement, encoded_size, expected):
    """Check native metrics only; route evidence belongs to profile notes, not JSON."""
    if not isinstance(measurement, dict) or not positive_number(measurement.get("seconds")):
        raise ValueError("Missing/invalid native encoder time")
    size, pixels = measurement.get("bytes"), expected.width * expected.height
    if type(size) is not int or size <= 0 or size != encoded_size:
        raise ValueError("Reported size differs from actual nonempty codestream")
    if (measurement.get("search") is not False
            or measurement.get("width") != expected.width
            or measurement.get("height") != expected.height):
        raise ValueError("Wrong dimensions or not a direct candidate encode")
    for key, value in (("bpp", size * 8 / pixels),
                       ("mpix_per_second", pixels / 1e6 / measurement["seconds"])):
        if not positive_number(measurement.get(key)) or not math.isclose(
                measurement[key], value, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"Missing/inconsistent native {key}")


def profile_notes(stderr, names=None):
    names = NOTE_NAMES if names is None else names
    notes = {name: [] for name in names}
    for line in stderr.splitlines():
        for name in names:
            prefix = f"[rs-jxl profile] {name}: "
            if not line.startswith(prefix):
                continue
            fields = {}
            for item in line[len(prefix):].split():
                key, separator, value = item.partition("=")
                if not separator or not value or key in fields:
                    raise ValueError(f"Malformed {name} profile note: {line}")
                fields[key] = value
            notes[name].append(fields)
    return notes


def counter(note, key):
    value = note.get(key, "")
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"Missing/invalid profile counter {key}: {value!r}")
    return int(value)


def validate_route(notes, variant):
    """Require observed work, not merely a requested flag or an inclusive timer."""
    lanes = lanes_for(variant)
    routes, batches, work = (notes[name] for name in NOTE_NAMES)
    chains = [row for row in routes if row.get("method") == "Optc256"
              and row.get("mode") == "SymbolId" and row.get("indexed") == "false"
              and row.get("bucket") == "false" and counter(row, "tokens") > 0]
    if not chains:
        raise ValueError("No ordinary Optc256 SymbolId chain route observed")
    if variant == "baseline":
        if batches or work or any(row.get("chain_batch") != "false" for row in chains):
            raise ValueError("Ordinary-chain baseline unexpectedly used chain batch")
        return dict(verified=True, lanes=1, batch_streams=0, computed=0, consumed=0,
                    overflow_requeries=0, demand_only_streams=0, interleaved_positions=0)
    active = [row for row in chains if row.get("chain_batch") == "true"]
    if not batches or not work or not active:
        raise ValueError("Requested chain batch did no observed work; choose an eligible original")
    if Counter(counter(row, "tokens") for row in active) != Counter(
            counter(row, "tokens") for row in batches):
        raise ValueError("Batch notes disagree with active chain routes")
    if Counter(counter(row, "tokens") for row in batches) != Counter(
            counter(row, "tokens") for row in work):
        raise ValueError("Incomplete chain batch/work profile notes")
    for row in batches + work:
        if counter(row, "lanes") != lanes:
            raise ValueError("Observed chain batch lanes differ from requested mode")
    computed = consumed = overflows = demand_only_streams = interleaved = 0
    for row in work:
        count, positions = counter(row, "computed"), counter(row, "interleaved_positions")
        tokens, queries = counter(row, "tokens"), counter(row, "consumed")
        overflow = counter(row, "overflow_requeries")
        if count > tokens or not 0 < queries <= tokens or overflow > queries:
            raise ValueError("Inconsistent chain batch work counters")
        if row.get("demand_only") not in ("true", "false"):
            raise ValueError("Missing/invalid chain batch demand_only flag")
        if positions != (0 if lanes == 1 else count):
            raise ValueError("Interleaved positions disagree with computed fill positions")
        computed += count
        consumed += queries
        overflows += overflow
        demand_only_streams += row["demand_only"] == "true"
        interleaved += positions
    if computed == 0:
        raise ValueError("No requested lane work observed")
    return dict(verified=True, lanes=lanes, batch_streams=len(batches),
                computed=computed, consumed=consumed, overflow_requeries=overflows,
                demand_only_streams=demand_only_streams, interleaved_positions=interleaved)


def encode_and_verify(args, source, expected, variant, folder, env, receipt, *, make_command=None):
    make_command = command if make_command is None else make_command
    jxl, png = folder / "output.jxl", folder / "decoded.png"
    receipt["encoder"] = port.run_process(make_command(args, source, jxl, variant), env, args.timeout)
    port.check_process(receipt["encoder"])
    receipt["decoder"] = port.run_process(
        port.decoder_command(args.decoder, jxl, png, expected.depth), env, args.timeout)
    port.check_process(receipt["decoder"])
    port.compare_pixels(expected, port.png_samples(png.read_bytes()))
    measurement = json.loads(receipt["encoder"]["stdout"])
    validate_measurement(measurement, jxl.stat().st_size, expected)
    receipt.update(verified=True, measurement=measurement)


def benchmark(args, *, make_command=None, read_notes=None, check_route=None,
              baseline="baseline", comparison="batch1", finalize_report=None):
    make_command = command if make_command is None else make_command
    read_notes = profile_notes if read_notes is None else read_notes
    check_route = validate_route if check_route is None else check_route
    report = dict(schema_version=1, status="failed", config=vars(args).copy(), cases=[], errors=[],
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  timing="Serial rotated native encoder call; median after warmups; no profiling",
                  profiles="Separate route-proof runs before timings; excluded from timing summaries; inclusive times overlap",
                  correctness="Original PNG samples including depth and alpha=0 RGB; official djxl",
                  decoder_provenance="Caller-attested official djxl; path and SHA256 recorded",
                  scope="Same fixed selected family, not outer search; baseline is ordinary chain",
                  interleaved_positions="Positions submitted to batched queries, including no-match/tail; not active concurrency",
                  regressions="All measured sizes/times retained; no speed or compression filtering",
                  timeout="Process guard only; not an encoder deadline")
    report["ratio_references"] = dict(baseline=baseline, comparison=comparison)
    identities, temporary = [], None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        try:
            if baseline not in args.variants or comparison not in args.variants:
                raise ValueError("Ratio references must be selected variants")
            env, report["environment"] = port.clean_environment(os.environ)
            profile_env = dict(env, RS_JXL_PROFILE="1")
            report["profile_environment"] = {"explicit": {"RS_JXL_PROFILE": "1"}}
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
            sources = load_cases(args)
            with tempfile.TemporaryDirectory(prefix="chain-interleave-", dir=args.out.parent) as scratch:
                temporary = Path(scratch)
                for index, source in enumerate(sources):
                    original = source.read_bytes()
                    digest = port.sha256(original)
                    identities.append((source, digest))
                    expected = port.png_samples(original)
                    snapshot = temporary / "input.png"
                    snapshot.write_bytes(original)
                    pixels = expected.width * expected.height
                    case = dict(source=str(source), sha256=digest, pixels=pixels,
                                width=expected.width, height=expected.height, depth=expected.depth,
                                variants={v: [] for v in args.variants}, profiles={}, summaries={})
                    report["cases"].append(case)
                    for variant in trial_order(args.variants, 0, index):
                        diagnostic = dict(verified=False, included_in_timings=False)
                        case["profiles"][variant] = diagnostic
                        with tempfile.TemporaryDirectory(prefix="profile-", dir=temporary) as trial:
                            encode_and_verify(args, snapshot, expected, variant, Path(trial),
                                              profile_env, diagnostic, make_command=make_command)
                        notes = read_notes(diagnostic["encoder"]["stderr"])
                        diagnostic["notes"] = notes
                        diagnostic["route"] = check_route(notes, variant)
                    for repeat in range(-args.warmups, args.repeats):
                        for variant in trial_order(args.variants, repeat, index):
                            sample = dict(repetition=repeat, warmup=repeat < 0, verified=False)
                            case["variants"][variant].append(sample)
                            with tempfile.TemporaryDirectory(prefix="trial-", dir=temporary) as trial:
                                encode_and_verify(args, snapshot, expected, variant, Path(trial), env, sample,
                                                  make_command=make_command)
                            measurement = sample["measurement"]
                            print(f"{source.name} {variant} repeat={repeat}: {measurement['bytes']} bytes "
                                  f"{measurement['seconds']:.6f}s verified", flush=True)
                    case["summaries"] = {v: summarize(samples, pixels)
                                         for v, samples in case["variants"].items()}
                    for reference in (baseline, comparison):
                        denominator = case["summaries"][reference]
                        for summary in case["summaries"].values():
                            summary[f"ratios_to_{reference}"] = {
                                key: summary[key] / denominator[key]
                                for key in ("bytes", "seconds", "bpp", "mpix_per_second")}
            report["status"] = "passed"
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            report["temporary_outputs_cleaned"] = temporary is None or not temporary.exists()
            for path, digest in identities:
                try:
                    if port.file_sha256(path) == digest:
                        continue
                    message = f"Input or binary changed during benchmark: {path}"
                except OSError as error:
                    message = str(error)
                report["errors"].append(message)
            if finalize_report is not None:
                try:
                    finalize_report(report)
                except (Exception, KeyboardInterrupt) as error:
                    report["errors"].append(f"{type(error).__name__}: {error}")
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
    print(report["status"], flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
