"""Explicit, paired MA-profile benchmark; no discovery or automatic large run.

Input is a manifest with a cases list containing source and cohort fields.
Original samples, including hidden RGB, are compared with an official decoder.
Encoded images are temporary; timings are native-call times, not CLI or IO.
The current default may include additional tools beyond explicit MA profiles;
compare their measurements, not equality of their compressed outputs.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

from benchmark_refinement import fixtures
from compare_cpp_port import (
    check_process, clean_environment, compare_pixels, decoder_command,
    file_sha256, png_samples, run_process,
)
from corpus_inputs import prepare

PROFILES = ("legacy", "half", "checked", "half-checked", "single-pass",
            "half-single-pass", "default")


def summarize(rows):
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[row["case"].get("cohort", "unspecified")].append(row)
        if row["case"].get("cohort") != "synthetic":
            cohorts["all-real"].append(row)
    result = []
    for cohort, items in cohorts.items():
        bases = {r["case"]["source"]: r for r in items if r["profile"] == "legacy"}
        for profile in sorted({r["profile"] for r in items}):
            selected = [r for r in items if r["profile"] == profile
                        and r["case"]["source"] in bases]
            if not selected:
                continue
            baseline = [bases[r["case"]["source"]] for r in selected]
            size = sum(r["bytes"] for r in selected)
            seconds = sum(r["seconds"] for r in selected)
            pixels = sum(r["width"] * r["height"] for r in selected)
            result.append(dict(
                cohort=cohort, profile=profile, images=len(selected), bytes=size,
                seconds=seconds, mean_bpp=sum(r["bpp"] for r in selected)/len(selected),
                weighted_bpp=8*size/pixels, mpix_per_second=pixels/1e6/seconds,
                size_change_pct=100*(size/sum(r["bytes"] for r in baseline)-1),
                time_ratio=seconds/sum(r["seconds"] for r in baseline),
                smaller=sum(r["bytes"] < b["bytes"] for r, b in zip(selected, baseline)),
                larger=sum(r["bytes"] > b["bytes"] for r, b in zip(selected, baseline)),
                worst_growth_pct=max(100*(r["bytes"]/b["bytes"]-1)
                                     for r, b in zip(selected, baseline)),
            ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES,
                        default=["legacy", "half", "checked", "single-pass", "half-single-pass", "default"])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--thread-subset", action="store_true",
                        help="First stratified >=100kpix original per dataset plus Half/Symmetric adverse controls")
    args = parser.parse_args()
    if not 1 <= args.threads <= 256 or "legacy" not in args.profiles or len(set(args.profiles)) != len(args.profiles):
        parser.error("Use threads 1..256 and distinct profiles including legacy")
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    if args.thread_subset:
        chosen, seen = [], set()
        for case in cases:
            if (case.get("cohort") == "stratified" and int(case["pixels"]) >= 100000
                    and case["dataset"] not in seen):
                chosen.append(case)
                seen.add(case["dataset"])
        chosen.extend(c for c in cases if c.get("cohort") in
                      ("adverse-description-half", "adverse-learning-symmetric"))
        cases = chosen
    if not cases:
        parser.error("No selected cases")
    args.out.mkdir(parents=True, exist_ok=False)
    binary = (args.out / args.encoder.name).resolve()
    shutil.copy2(args.encoder, binary)
    env, _ = clean_environment(os.environ)
    manifest = dict(cases=cases, profiles=args.profiles, threads=args.threads, effort=9,
                    encoder_sha256=file_sha256(binary), decoder_sha256=file_sha256(args.decoder),
                    timing="one native encode including profile projection/pool; excludes image IO",
                    correctness="official decoder samples; includes hidden RGB; no JXL byte identity requirement")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records = []
    with tempfile.TemporaryDirectory(prefix="fixtures-", dir=args.out) as synthetic:
        if args.synthetic:
            cases = cases + [dict(source=str(p.resolve()), cohort="synthetic")
                             for p in fixtures(Path(synthetic))]
        with (args.out/"trials.jsonl").open("w", encoding="utf-8") as journal:
            for index, case in enumerate(cases):
                with tempfile.TemporaryDirectory(prefix="case-", dir=args.out) as tmp:
                    folder = Path(tmp).resolve()
                    info, snapshot, expected = prepare(Path(case["source"]), folder)
                    for key, got in (("source_sha256", "input_sha256"), ("payload_sha256", "payload_sha256")):
                        if key in case and case[key] != info[got]:
                            raise ValueError(f"Input changed: {case['source']} {key}")
                    order = args.profiles.copy()
                    random.Random(20261114 + index).shuffle(order)
                    for profile in order:
                        output, decoded = folder/(profile+".jxl"), folder/(profile+".png")
                        command = [str(binary), str(snapshot), "--output", str(output),
                                   "--effort", "9", "--threads", str(args.threads),
                                   "--direct-candidate", "baseline"]
                        if profile != "default":
                            command += ["--direct-ma-profile", profile]
                        process = run_process(command, env, 600)
                        check_process(process)
                        row = json.loads(process["stdout"])
                        if not row["direct"] or row["encode_calls"] != 1 or row["bytes"] != output.stat().st_size:
                            raise ValueError("Invalid direct encoder receipt")
                        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                            raise ValueError("Invalid native time")
                        expected_profile = "half-single-pass" if profile == "default" else profile
                        if row["direct_ma_profile"] != expected_profile:
                            raise ValueError("Requested MA profile was not applied")
                        if "single-pass" in expected_profile:
                            stats = row["refinement"]
                            if (stats["domains"] < 1 or stats["models"] != stats["domains"]
                                    or stats["offset_single_pass_domains"] != stats["domains"]
                                    or stats["offset_wins"] or stats["saved_bits"]):
                                raise ValueError("Single-pass path performed hidden comparison or misreported savings")
                        dec = run_process(decoder_command(args.decoder, output, decoded, expected.depth), env, 120)
                        check_process(dec)
                        compare_pixels(expected, png_samples(decoded.read_bytes()))
                        row.update(case=case, profile=profile, verified=True)
                        journal.write(json.dumps(row)+"\n")
                        journal.flush()
                        records.append(row)
                        print(f"[{index+1}/{len(cases)}] {profile:17} {row['bytes']:10} {row['seconds']:.3f}s {Path(case['source']).name}", flush=True)
                (args.out/"summary.json").write_text(json.dumps(summarize(records), indent=2), encoding="utf-8")
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
