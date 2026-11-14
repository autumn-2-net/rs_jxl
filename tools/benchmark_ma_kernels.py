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

from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process,
)
from corpus_inputs import prepare


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
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--effort", type=int, choices=range(1, 12), default=9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", choices=("off", "stages", "detail"), default="off")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
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
                    profile=args.profile, avx512=args.avx512, effort=args.effort,
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
                            process = run_process([str(frozen[name]), str(snapshot), "--output", str(output),
                                                   "--effort", str(args.effort), "--threads", str(threads),
                                                   "--direct-candidate", "baseline"], active_env, args.timeout)
                            check_process(process)
                            row = json.loads(process["stdout"])
                            if (not row["direct"] or row["encode_calls"] != 1
                                    or row["effort"] != args.effort or row["threads"] != threads
                                    or row["bytes"] != output.stat().st_size
                                    or not math.isfinite(row["seconds"]) or row["seconds"] <= 0
                                    or (row["width"], row["height"]) != (expected.width, expected.height)):
                                raise ValueError("Invalid native encoder receipt")
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
