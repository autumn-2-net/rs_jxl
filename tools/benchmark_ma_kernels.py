"""Paired native kernel experiments on an explicit image manifest (default e9).

Diagnostics and timed trials are separate runs. Every output is checked with
an independent djxl against original samples, including invisible RGB.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from statistics import median
import tempfile


def parse_ema(values, variants, distances=(), polish=(), final_assignment=()):
    ema = {}
    for value in values:
        match = re.fullmatch(r"([a-zA-Z0-9_-]+)=([0-9]+):(random|farthest)", value)
        if match is None:
            raise ValueError("Require --ema NAME=PASSES:random|farthest")
        name, passes, initialization = match.groups()
        passes = int(passes)
        if name not in variants or name in ema or not 1 <= passes <= 255:
            raise ValueError("EMA requires a known, unique variant and passes in 1..255")
        ema[name] = dict(passes=passes, initialization=initialization)
    for flag, settings, field, pattern in (
            ("--ema-distance", distances, "distance", r"cross-entropy|hellinger|merge-entropy"),
            ("--ema-polish", polish, "polish", r"[0-9]+")):
        seen = set()
        for value in settings:
            match = re.fullmatch(r"([a-zA-Z0-9_-]+)=(" + pattern + r")", value)
            if match is None:
                raise ValueError(f"Require {flag} NAME=" + (
                    "cross-entropy|hellinger|merge-entropy" if field == "distance" else "PASSES (0..255)"))
            name, setting = match.groups()
            if name not in ema or name in seen:
                raise ValueError(f"{flag} requires a known, unique --ema variant")
            if field == "polish":
                setting = int(setting)
                if not 0 <= setting <= 255:
                    raise ValueError("EMA polish passes must be in 0..255")
            ema[name][field] = setting
            seen.add(name)
    seen = set()
    for name in final_assignment:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name not in ema or name in seen:
            raise ValueError("--ema-final-assignment requires a safe, unique --ema variant name")
        ema[name]["final_assignment"] = True
        seen.add(name)
    return ema


def ema_arguments(options):
    if options is None:
        return []
    result = ["--ema-passes", str(options["passes"])]
    if options["initialization"] == "farthest":
        result.append("--ema-farthest")
    if "distance" in options:
        result.extend(["--ema-distance", options["distance"]])
    if "polish" in options:
        result.extend(["--ema-polish", str(options["polish"])])
    if options.get("final_assignment", False):
        result.append("--ema-final-assignment")
    return result


def validate_ema_receipt(row, options):
    # Older baseline binaries predate all EMA receipt fields.
    if options is None and not any(field in row for field in (
            "ema_passes", "ema_farthest", "ema_distance", "ema_polish", "ema_final_assignment")):
        return
    passes = options["passes"] if options is not None else None
    farthest = options is not None and options["initialization"] == "farthest"
    if ("ema_passes" not in row or "ema_farthest" not in row
            or type(row["ema_passes"]) is not type(passes)
            or row["ema_passes"] != passes or row["ema_farthest"] is not farthest):
        raise ValueError("EMA receipt does not match requested passes/initialization")
    distance = options.get("distance", "cross-entropy") if options is not None else None
    polish = options.get("polish", 0) if options is not None else 0
    # Legacy EMA binaries only implement cross-entropy without polishing.
    actual_distance = row.get("ema_distance", "cross-entropy" if options is not None else None)
    actual_polish = row.get("ema_polish", 0)
    if (type(actual_distance) is not type(distance) or actual_distance != distance
            or type(actual_polish) is not int or actual_polish != polish):
        raise ValueError("EMA receipt does not match requested distance/polish")
    final_assignment = options.get("final_assignment", False) if options is not None else False
    actual_final_assignment = row.get("ema_final_assignment", False)
    if type(actual_final_assignment) is not bool or actual_final_assignment is not final_assignment:
        raise ValueError("EMA receipt does not match requested final assignment")


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"], row["variant"], row["threads"]].append(row)
    totals = defaultdict(list)
    for trials in grouped.values():
        first = trials[0]
        totals[first["variant"], first["threads"]].append(dict(
            source=first["source"], pixels=first["width"] * first["height"],
            bytes=median(r["bytes"] for r in trials),
            seconds=median(r["seconds"] for r in trials),
        ))
    result = []
    for (variant, threads), items in totals.items():
        size = sum(r["bytes"] for r in items)
        seconds = sum(r["seconds"] for r in items)
        pixels = sum(r["pixels"] for r in items)
        result.append(dict(variant=variant, threads=threads, images=len(items),
                           bytes=size, seconds=seconds,
                           mean_bpp=sum(8*r["bytes"]/r["pixels"] for r in items)/len(items),
                           weighted_bpp=8*size/pixels, mpix_s=pixels/1e6/seconds,
                           per_image=items))
    return result


def parse_stages(stderr):
    stages = {}
    in_table = False
    for line in stderr.splitlines():
        line = line.strip()
        if line == "stage | calls | total seconds | longest seconds":
            in_table = True
        elif line.startswith("[rs-jxl profile]"):
            in_table = False
        elif in_table and "|" in line:
            name, calls, total, longest = (field.strip() for field in line.split("|"))
            stages[name] = dict(calls=int(calls), seconds=float(total), longest=float(longest))
        elif line and in_table:
            in_table = False
    if not stages:
        raise ValueError("Missing native stage diagnostics")
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=EXECUTABLE")
    parser.add_argument("--avx512", nargs="*", default=[], help="Variant names opting into existing AVX512 scoring")
    parser.add_argument("--ema", action="append", default=[], metavar="NAME=PASSES:random|farthest",
                        help="Opt a variant into EMA clustering with 1..255 passes; repeat per variant")
    parser.add_argument("--ema-distance", action="append", default=[], metavar="NAME=MODE",
                        help="Set cross-entropy, hellinger or merge-entropy for a --ema variant")
    parser.add_argument("--ema-polish", action="append", default=[], metavar="NAME=PASSES",
                        help="Set 0..255 polishing passes for a --ema variant")
    parser.add_argument("--ema-final-assignment", action="append", default=[], metavar="NAME",
                        help="Assign against frozen final centers for a --ema variant")
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--effort", type=int, choices=range(1, 12), default=9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", choices=("off", "stages", "detail"), default="off")
    parser.add_argument("--profile-ma-nodes", action="store_true",
                        help="Include MA node-size timing buckets (requires profiling)")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.profile_ma_nodes and args.profile == "off":
        parser.error("--profile-ma-nodes requires --profile stages or detail")
    if (args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
            or not all(1 <= t <= 256 for t in args.threads)
            or len(args.threads) != len(set(args.threads))):
        parser.error("Require positive repeats/timeout and distinct thread counts in 1..256")
    variants = {}
    for value in args.variant:
        name, separator, path = value.partition("=")
        if not separator or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in variants:
            parser.error("Require distinct safe NAME=EXECUTABLE variants")
        variants[name] = Path(path).resolve(strict=True)
    if not set(args.avx512) <= variants.keys():
        parser.error("Unknown AVX512 variant")
    try:
        ema = parse_ema(args.ema, variants, args.ema_distance, args.ema_polish,
                        args.ema_final_assignment)
    except ValueError as error:
        parser.error(str(error))

    from compare_cpp_port import (
        check_process, clean_environment, compare_pixels, decoder_command,
        file_sha256, png_samples, run_process,
    )
    from corpus_inputs import prepare

    cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    if not cases or len({str(Path(c["source"]).resolve()) for c in cases}) != len(cases):
        parser.error("Require a nonempty manifest of distinct sources")
    args.out = args.out.resolve()
    args.decoder = args.decoder.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = {}
    for name, path in variants.items():
        target = args.out / (name + path.suffix)
        shutil.copy2(path, target)
        frozen[name] = target
    manifest = dict(cases=cases, threads=args.threads, repeats=args.repeats,
                    profile=args.profile, profile_ma_nodes=args.profile_ma_nodes,
                    avx512=args.avx512, ema=ema, effort=args.effort,
                    binaries={name: dict(source=str(variants[name]), sha256=file_sha256(path))
                              for name, path in frozen.items()},
                    decoder=dict(path=str(args.decoder), sha256=file_sha256(args.decoder)),
                    timing="sum of per-image median native calls; excludes CLI/IO/verification",
                    stage_semantics="inclusive scopes; nested/parallel totals overlap, not wall shares",
                    included_in_timing=args.profile == "off")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env, _ = clean_environment(os.environ)
    rows = []
    order = list(frozen)
    random.Random(20261114).shuffle(order)
    with (args.out/"trials.jsonl").open("w", encoding="utf-8") as journal:
        for index, case in enumerate(cases):
            with tempfile.TemporaryDirectory(prefix="image-", dir=args.out) as tmp:
                directory = Path(tmp)
                info, snapshot, expected = prepare(Path(case["source"]), directory)
                for key, actual in (("source_sha256", "input_sha256"), ("payload_sha256", "payload_sha256")):
                    if key in case and case[key] != info[actual]:
                        raise ValueError(f"Input changed: {case['source']} {key}")
                for threads in args.threads:
                    for repeat in range(args.repeats):
                        shift = (index+repeat) % len(order)
                        for position, name in enumerate(order[shift:]+order[:shift]):
                            stem = f"{index:03}-t{threads}-r{repeat}-{name}"
                            output, decoded = directory/(stem+".jxl"), directory/(stem+".png")
                            active_env = dict(env, RS_JXL_MA_AVX512="1" if name in args.avx512 else "0")
                            if args.profile != "off":
                                active_env["RS_JXL_PROFILE"] = "1"
                                active_env["RS_JXL_PROFILE_DETAIL"] = "1" if args.profile == "detail" else "0"
                                if args.profile_ma_nodes:
                                    active_env["RS_JXL_PROFILE_MA_NODES"] = "1"
                            process = run_process([str(frozen[name]), str(snapshot), "--output", str(output),
                                                   "--effort", str(args.effort), "--threads", str(threads),
                                                   "--direct-candidate", "baseline"] + ema_arguments(ema.get(name)),
                                                  active_env, args.timeout)
                            check_process(process)
                            row = json.loads(process["stdout"])
                            if (not row["direct"] or row["encode_calls"] != 1
                                    or row["effort"] != args.effort or row["threads"] != threads
                                    or row["bytes"] != output.stat().st_size
                                    or not math.isfinite(row["seconds"]) or row["seconds"] <= 0
                                    or (row["width"], row["height"]) != (expected.width, expected.height)):
                                raise ValueError("Invalid native encoder receipt")
                            validate_ema_receipt(row, ema.get(name))
                            dec = run_process(decoder_command(args.decoder, output, decoded, expected.depth), env, args.timeout)
                            check_process(dec)
                            compare_pixels(expected, png_samples(decoded.read_bytes()))
                            row.update(source=case["source"], variant=name, threads=threads,
                                       repeat=repeat, order=position, verified=True,
                                       included_in_timing=args.profile == "off")
                            if args.profile != "off":
                                (args.out/(stem+".profile.txt")).write_text(process["stderr"], encoding="utf-8")
                                row["stages"] = parse_stages(process["stderr"])
                            journal.write(json.dumps(row)+"\n")
                            journal.flush()
                            rows.append(row)
                            output.unlink()
                            decoded.unlink()
                            print(f"[{index+1}/{len(cases)}] t{threads} r{repeat+1} {name} "
                                  f"{row['bytes']} B {row['seconds']:.4f}s {Path(case['source']).name}", flush=True)
            if args.profile == "off":
                (args.out/"summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
