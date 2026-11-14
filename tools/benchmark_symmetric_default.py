"""Bounded e9 tree-learning comparison against the Half + single-pass default.

Uses an explicit manifest only. Every trial is independently decoded by djxl
and checked against original samples, including invisible RGB. No corpus scan.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import median
import tempfile

from benchmark_offset_single_pass import summarize as summarize_profiles
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process,
)
from corpus_inputs import prepare


# Freeze the preceding e9 profile explicitly; None exercises current defaults.
PROFILES = {
    "baseline": ("half-single-pass", "legacy"),
    "default": (None, "symmetric"),
    "symmetric-single-pass": ("single-pass", "symmetric"),
    "symmetric-half-single-pass": ("half-single-pass", "symmetric"),
    "cost-half-single-pass": ("half-single-pass", "cost"),
}


def select_cases(cases, subset):
    if subset:
        chosen, seen = [], set()
        for case in cases:
            if (case.get("cohort") == "stratified" and int(case["pixels"]) >= 100000
                    and case["dataset"] not in seen):
                chosen.append(case)
                seen.add(case["dataset"])
        chosen.extend(c for c in cases if c.get("cohort") in
                      ("adverse-description-half", "adverse-learning-symmetric"))
        cases = chosen
    # Deduplicate overlapping adverse controls while preserving manifest order.
    unique = {}
    for case in cases:
        unique.setdefault(str(Path(case["source"]).resolve()), case)
    return list(unique.values())


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"]["source"], row["threads"], row["profile"])].append(row)
    images = []
    for trials in grouped.values():
        first = trials[0]
        if any((r["width"], r["height"]) !=
               (first["width"], first["height"]) for r in trials):
            raise ValueError("Repeated encodes differ in image dimensions")
        size = median(r["bytes"] for r in trials)
        images.append(dict(first, bytes=size, bpp=8*size/(first["width"]*first["height"]),
                           seconds=median(r["seconds"] for r in trials)))
    result = []
    for threads in sorted({r["threads"] for r in images}):
        # Compare with the explicit preceding profile even after defaults change.
        selected = [dict(r, profile="legacy" if r["profile"] == "baseline"
                         else r["profile"]) for r in images if r["threads"] == threads]
        for summary in summarize_profiles(selected):
            if summary["profile"] == "legacy":
                summary["profile"] = "baseline"
            result.append(dict(summary, threads=threads, time_aggregation="per-image median"))
    return result


def verify_receipt(row, output, profile, threads):
    ma, learning = PROFILES[profile]
    if (not row["direct"] or row["encode_calls"] != 1 or row["effort"] != 9
            or row["threads"] != threads or row["bytes"] != output.stat().st_size):
        raise ValueError("Invalid direct encoder receipt")
    if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
        raise ValueError("Invalid native time")
    if (row["direct_ma_profile"] != (ma or "half-single-pass")
            or row["direct_tree_learning"] != learning):
        raise ValueError("Requested MA/tree-learning settings were not applied")
    stats = row["refinement"]
    if (stats["domains"] < 1 or stats["models"] != stats["domains"]
            or stats["offset_single_pass_domains"] != stats["domains"]
            or stats["offset_wins"] or stats["saved_bits"]):
        raise ValueError("Single-pass path performed hidden comparison or misreported savings")
    counts = row["tree_learning"]
    if any(not isinstance(counts[k], int) or counts[k] < 0 for k in
           ("learners", "records", "cost_splits", "symmetric_splits")):
        raise ValueError("Invalid tree-learning counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, nargs="+", choices=(1, 4, 16), default=[1, 4, 16])
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--thread-subset", action=argparse.BooleanOptionalAction, default=True,
                        help="Same first >=100kpix stratified case per dataset and adverse controls at every thread count")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use repeats >= 1")
    if (len(set(args.threads)) != len(args.threads)
            or len(set(args.profiles)) != len(args.profiles) or "baseline" not in args.profiles):
        parser.error("Use distinct threads and distinct profiles including baseline")
    cases = select_cases(json.loads(args.manifest.read_text(encoding="utf-8"))["cases"],
                         args.thread_subset)
    if not cases:
        parser.error("The explicit manifest selects no cases")
    args.out = args.out.resolve()
    args.decoder = args.decoder.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.out / args.encoder.name
    shutil.copy2(args.encoder, binary)
    env, _ = clean_environment(os.environ)
    manifest = dict(
        cases=cases, profiles=args.profiles, profile_options=PROFILES,
        threads=args.threads, thread_subset=args.thread_subset, effort=9, repeats=args.repeats,
        order="seeded profile order rotated by case index and repetition; same across threads",
        summary_timing="sum of per-image/profile median native seconds; images counted once",
        source_manifest=str(args.manifest.resolve()),
        encoder_sha256=file_sha256(binary), decoder_sha256=file_sha256(args.decoder),
        timing="one native encode including recipe projection/pool; excludes image IO and djxl",
        correctness="independent djxl exact original samples including invisible RGB",
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records = []
    order = args.profiles.copy()
    random.Random(20261114).shuffle(order)
    with (args.out / "trials.jsonl").open("w", encoding="utf-8") as journal:
        for index, case in enumerate(cases):
            with tempfile.TemporaryDirectory(prefix="case-", dir=args.out) as tmp:
                folder = Path(tmp)
                info, snapshot, expected = prepare(Path(case["source"]), folder)
                for key, got in (("source_sha256", "input_sha256"),
                                 ("payload_sha256", "payload_sha256")):
                    if key in case and case[key] != info[got]:
                        raise ValueError(f"Input changed: {case['source']} {key}")
                for threads in args.threads:
                    for repetition in range(1, args.repeats + 1):
                        shift = (index + repetition - 1) % len(order)
                        for position, profile in enumerate(order[shift:] + order[:shift], 1):
                            stem = f"t{threads}-r{repetition}-{profile}"
                            output, decoded = folder / (stem + ".jxl"), folder / (stem + ".png")
                            command = [str(binary), str(snapshot), "--output", str(output),
                                       "--effort", "9", "--threads", str(threads),
                                       "--direct-candidate", "baseline"]
                            ma, learning = PROFILES[profile]
                            if ma is not None:
                                command += ["--direct-ma-profile", ma,
                                            "--direct-tree-learning", learning]
                            process = run_process(command, env, 600)
                            check_process(process)
                            row = json.loads(process["stdout"])
                            verify_receipt(row, output, profile, threads)
                            dec = run_process(decoder_command(args.decoder, output, decoded, expected.depth),
                                              env, 120)
                            check_process(dec)
                            compare_pixels(expected, png_samples(decoded.read_bytes()))
                            if (row["width"], row["height"]) != (expected.width, expected.height):
                                raise ValueError("Encoder receipt dimensions differ from original")
                            row.update(case=case, profile=profile, repetition=repetition,
                                       order_position=position, verified=True)
                            records.append(row)
                            journal.write(json.dumps(row) + "\n")
                            journal.flush()
                            print(f"[{index+1}/{len(cases)}] t{threads} r{repetition} {profile:28} "
                                  f"{row['bytes']:10} {row['seconds']:.3f}s {Path(case['source']).name}",
                                  flush=True)
                    (args.out / "summary.json").write_text(
                        json.dumps(summarize(records), indent=2), encoding="utf-8")
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
