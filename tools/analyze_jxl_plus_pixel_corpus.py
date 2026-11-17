"""Validate and summarize a complete fixed PNG pixel-lossless experiment.

Reads source files only for SHA256/PNG header identity. Does not decode pixels,
run codecs, benchmark, or use a compressed entropy subsection as a size result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ("plus_static", "plus_fitted_zero_gate", "plus_adaptive_zero_gate", "standard_jxl_default")
ADAPTIVE = RECIPES[2]
EXPERIMENT = "pixel_lossless_zero_gate"
ANALYSIS_SCHEMA = "rs-jxl-plus.pixel-corpus-analysis.v1"
BYTE_FIELDS = ("tree_bytes", "wp_bytes", "gate_bytes", "table_bytes",
               "entropy_header_bytes", "entropy_data_bytes", "framing_bytes")
BREAKDOWN_FIELDS = ("groups",) + BYTE_FIELDS
EXPERIMENT_SOURCES = (
    "crates/rs-jxl-plus/src/pixel_model.rs",
    "crates/rs-jxl-plus/src/pixel_tests.rs",
    "crates/rs-jxl/src/modular/plus_bridge.rs",
    "crates/rs-jxl/src/entropy/experimental_decode.rs",
    "vendor/jxl-rs/jxl/src/api/experimental_entropy.rs",
    "crates/rs-jxl-plus/src/arith.rs",
    "Cargo.toml",
    "tools/analyze_jxl_plus_pixel_corpus.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, description, minimum=0):
    require(type(value) is int and value >= minimum, f"{description}: expected integer >= {minimum}")
    return value


def seconds(value, description):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{description}: expected finite nonnegative seconds")
    return float(value)


def normalized(path):
    return str(path).replace("\\", "/").casefold()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant {value} in {path}")
    data = path.read_bytes()
    return json.loads(data, parse_constant=invalid_constant), hashlib.sha256(data).hexdigest()


def source_snapshot():
    records = []
    for relative in EXPERIMENT_SOURCES:
        path = ROOT / relative
        data = path.read_bytes()
        records.append(dict(source=relative, sha256=hashlib.sha256(data).hexdigest(), original_bytes=len(data)))
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    release = re.search(r"(?ms)^\[profile\.release\]\s*\n(.*?)(?=^\[|\Z)", cargo)
    require(release is not None
            and re.search(r'(?m)^lto\s*=\s*"thin"\s*$', release.group(1)) is not None
            and re.search(r"(?m)^codegen-units\s*=\s*1\s*$", release.group(1)) is not None,
            "Current Cargo release profile differs from the measured thin-LTO/codegen-units=1 environment")
    return dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), files=records,
                claim="Individual files frozen at analysis time, not an assertion of a clean repository HEAD or a binary/source build attestation.")


def source_identity(case):
    """Verify actual source bytes after the measured run, without image decoding."""
    path = Path(case["source"])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(digest == case["sha256"], f"Source SHA256 changed: {path}")
    require(len(data) == case["original_bytes"], f"Source size changed: {path}")
    require(len(data) >= 33 and data[:16] == b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR", f"Bad PNG IHDR: {path}")
    width, height, bits, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", data[16:29])
    for key, actual in (("width", width), ("height", height), ("bit_depth", bits), ("png_color_type", color)):
        require(case[key] == actual, f"Manifest/source {key} differs: {path}")
    require(width > 0 and height > 0 and compression == filtering == 0 and interlace <= 1,
            f"Bad PNG header parameters: {path}")
    # The native PNG input uses EXPAND: packed gray/palettes expand to 8 bits;
    # tRNS adds an alpha channel. Header chunks suffice to check that identity.
    transparency = False
    position = 8
    while position + 12 <= len(data):
        size = int.from_bytes(data[position:position + 4], "big")
        end = position + 12 + size
        require(end <= len(data), f"Truncated PNG chunk: {path}")
        kind = data[position + 4:position + 8]
        if kind == b"tRNS":
            transparency = True
        if kind == b"IDAT":
            break
        position = end
    require(color in (0, 2, 3, 4, 6), f"Unknown PNG color type: {path}")
    channels = {0: 1 + int(transparency), 2: 3 + int(transparency),
                3: 3 + int(transparency), 4: 2, 6: 4}[color]
    return dict(source=path.as_posix(), sha256=digest, original_bytes=len(data),
                width=width, height=height, bits=max(8, bits), channels=channels)


def recipe_map(values, description):
    require(isinstance(values, list) and len(values) == len(RECIPES), f"{description}: expected {len(RECIPES)} recipes")
    result = {}
    for value in values:
        require(isinstance(value, dict), f"{description}: expected recipe object")
        recipe = value.get("recipe")
        require(recipe in RECIPES and recipe not in result, f"{description}: missing/duplicate/unknown recipe")
        result[recipe] = value
    return result


def validate_case(case, measured):
    label = case["label"]
    for key in ("label", "sha256", "group", "previously_used"):
        require(measured.get(key) == case[key], f"{label}: receipt/manifest {key} differs")
    require(type(case["previously_used"]) is bool and type(measured["previously_used"]) is bool,
            f"{label}: prior-use classification must be Boolean")
    require(normalized(measured.get("source", "")) == normalized(case["source"]), f"{label}: source path differs")
    source = source_identity(case)
    require(measured.get("source_type") == "PNG", f"{label}: source type differs")
    for key in ("width", "height", "bits", "channels"):
        require(measured.get(key) == source[key], f"{label}: decoded {key} differs from PNG input contract")
    pixels = source["width"] * source["height"]
    require(measured.get("pixels") == pixels and case.get("pixels", pixels) == pixels, f"{label}: pixel count differs")
    require(measured.get("original_png_bytes") == source["original_bytes"], f"{label}: receipt source size differs")
    require(measured.get("sample_storage") == ("u16" if source["bits"] == 16 else "u8"), f"{label}: sample storage differs")
    variants = recipe_map(measured.get("variants"), f"{label} variants")
    repeats = measured.get("repeats")
    require(isinstance(repeats, list) and len(repeats) == 3 * len(RECIPES), f"{label}: expected all {3 * len(RECIPES)} measured outputs")
    slots, measurements = set(), {recipe: [] for recipe in RECIPES}
    for run in repeats:
        round_id = integer(run.get("round"), f"{label} round")
        position = integer(run.get("position"), f"{label} position")
        require(round_id < 3 and position < len(RECIPES) and (round_id, position) not in slots,
                f"{label}: duplicate/out-of-range measurement slot")
        slots.add((round_id, position))
        recipe = run.get("recipe")
        require(recipe == RECIPES[(position + round_id + 1) % len(RECIPES)], f"{label}: recipe rotation differs")
        require(run.get("all_samples_verified") is True, f"{label}/{recipe}: samples not verified exactly")
        size = integer(run.get("complete_bytes"), f"{label}/{recipe} complete bytes", 1)
        measurements[recipe].append(dict(
            complete_bytes=size,
            encode_seconds=seconds(run.get("encode_seconds"), f"{label}/{recipe} encode"),
            decode_seconds=seconds(run.get("decode_seconds"), f"{label}/{recipe} decode")))
    result = dict(label=label, group=case["group"], previously_used=case["previously_used"],
                  source=case["source"], sha256=source["sha256"], pixels=pixels,
                  original_png_bytes=source["original_bytes"], variants={})
    for recipe in RECIPES:
        runs = measurements[recipe]
        require(len(runs) == 3, f"{label}/{recipe}: expected three measured repeats")
        sizes = {run["complete_bytes"] for run in runs}
        require(len(sizes) == 1, f"{label}/{recipe}: nondeterministic complete sizes")
        size = next(iter(sizes))
        variant = variants[recipe]
        require(integer(variant.get("complete_bytes"), f"{label}/{recipe} summary bytes", 1) == size,
                f"{label}/{recipe}: repeat/summary bytes differ")
        encode = statistics.median(run["encode_seconds"] for run in runs)
        decode = statistics.median(run["decode_seconds"] for run in runs)
        require(close(seconds(variant.get("median_encode_seconds"), f"{label}/{recipe} median encode"), encode),
                f"{label}/{recipe}: encode median differs")
        require(close(seconds(variant.get("median_decode_seconds"), f"{label}/{recipe} median decode"), decode),
                f"{label}/{recipe}: decode median differs")
        breakdown = variant.get("byte_breakdown")
        if recipe == "standard_jxl_default":
            require(breakdown is None, f"{label}: unexpected standard breakdown schema")
        else:
            require(isinstance(breakdown, dict) and set(breakdown) == set(BREAKDOWN_FIELDS),
                    f"{label}/{recipe}: incomplete/unknown cost breakdown")
            breakdown = {key: integer(breakdown[key], f"{label}/{recipe} {key}") for key in BREAKDOWN_FIELDS}
            require(sum(breakdown[key] for key in BYTE_FIELDS) == size,
                    f"{label}/{recipe}: cost sections do not sum to complete bytes")
        result["variants"][recipe] = dict(complete_bytes=size, median_encode_seconds=encode,
                                         median_decode_seconds=decode, byte_breakdown=breakdown)
    return result


def summarize(cases):
    pixels = sum(case["pixels"] for case in cases)
    result = dict(images=len(cases), pixels=pixels,
                  original_png_bytes=sum(case["original_png_bytes"] for case in cases), variants={})
    for recipe in RECIPES:
        variants = [case["variants"][recipe] for case in cases]
        size = sum(row["complete_bytes"] for row in variants)
        encode = math.fsum(row["median_encode_seconds"] for row in variants)
        decode = math.fsum(row["median_decode_seconds"] for row in variants)
        breakdown = None if recipe == "standard_jxl_default" else {
            key: sum(row["byte_breakdown"][key] for row in variants) for key in BREAKDOWN_FIELDS}
        result["variants"][recipe] = dict(
            complete_bytes=size, bits_per_pixel=ratio(8 * size, pixels),
            sum_per_image_median_encode_seconds=encode, sum_per_image_median_decode_seconds=decode,
            encode_megapixels_per_second=ratio(pixels / 1_000_000, encode),
            decode_megapixels_per_second=ratio(pixels / 1_000_000, decode),
            summed_byte_breakdown=breakdown)
    adaptive_bytes = result["variants"][ADAPTIVE]["complete_bytes"]
    result["adaptive_vs_baselines"] = {}
    for baseline in RECIPES:
        if baseline == ADAPTIVE:
            continue
        baseline_bytes = result["variants"][baseline]["complete_bytes"]
        differences = [case["variants"][baseline]["complete_bytes"] - case["variants"][ADAPTIVE]["complete_bytes"]
                       for case in cases]
        savings = baseline_bytes - adaptive_bytes
        result["adaptive_vs_baselines"][baseline] = dict(
            baseline_complete_bytes=baseline_bytes, adaptive_complete_bytes=adaptive_bytes,
            saved_bytes=savings, savings_percent=ratio(100 * savings, baseline_bytes),
            wins=sum(value > 0 for value in differences), ties=sum(value == 0 for value in differences),
            losses=sum(value < 0 for value in differences))
        comparison = result["adaptive_vs_baselines"][baseline]
        if baseline == "standard_jxl_default":
            comparison["saved_bytes_by_section"] = None
        else:
            baseline_sections = result["variants"][baseline]["summed_byte_breakdown"]
            adaptive_sections = result["variants"][ADAPTIVE]["summed_byte_breakdown"]
            # Positive values are bytes saved by adaptive; negative values are
            # additional overhead. Group counts are not byte contributions.
            comparison["saved_bytes_by_section"] = {
                key: baseline_sections[key] - adaptive_sections[key] for key in BYTE_FIELDS}
            comparison["group_count_difference_baseline_minus_adaptive"] = baseline_sections["groups"] - adaptive_sections["groups"]
            require(sum(comparison["saved_bytes_by_section"].values()) == savings,
                    f"{baseline}: section deltas do not explain complete-byte savings")
    return result


def analyze(manifest_path, receipt_path):
    manifest, manifest_hash = read_json(manifest_path)
    receipt, receipt_hash = read_json(receipt_path)
    require(manifest.get("schema") == "rs-jxl-plus.pixel-corpus.v1", "Unknown manifest schema")
    require(receipt.get("schema") == 1 and receipt.get("experiment") == EXPERIMENT, "Unknown receipt schema")
    require(receipt.get("warmup_rounds") == 1 and receipt.get("measured_rounds") == 3,
            "Receipt must have one warmup and three measured rounds")
    require(receipt.get("strip_metadata") is True and receipt.get("color_conversion") is False,
            "Unexpected input metadata/color handling")
    require(receipt.get("all_samples_include_invisible_rgb") is True, "Invisible RGB samples are not covered")
    require(receipt.get("threads") == 1 and receipt.get("plus_tile_size") == 256, "Unexpected experiment settings")
    expected, measured = manifest.get("cases"), receipt.get("cases")
    require(isinstance(expected, list) and len(expected) > 0 and isinstance(measured, list), "Missing image lists")
    require(len(expected) == len(measured), "Receipt is incomplete or has unexpected images")
    require(all(isinstance(row, dict) and isinstance(row.get("label"), str) for row in expected + measured),
            "Invalid image identity record")
    require(len({row["label"] for row in expected}) == len(expected), "Duplicate manifest image labels")
    require(len({normalized(row["source"]) for row in expected}) == len(expected), "Duplicate manifest sources")
    require(len({row["sha256"] for row in expected}) == len(expected), "Duplicate manifest source bytes")
    require([row["label"] for row in expected] == [row["label"] for row in measured], "Image order/identity differs")
    cases = [validate_case(case, row) for case, row in zip(expected, measured)]
    overall = summarize(cases)
    if "summary" in manifest:
        require(manifest["summary"]["cases"] == len(cases) and manifest["summary"]["pixels"] == overall["pixels"],
                "Manifest summary count/pixels differs")
    totals = recipe_map(receipt.get("totals"), "receipt totals")
    for recipe in RECIPES:
        actual, declared = overall["variants"][recipe], totals[recipe]
        require(declared.get("complete_bytes") == actual["complete_bytes"], f"{recipe}: root complete-byte total differs")
        for key in ("sum_per_image_median_encode_seconds", "sum_per_image_median_decode_seconds"):
            require(close(seconds(declared.get(key), f"{recipe} {key}"), actual[key]), f"{recipe}: root timing total differs")
    groups = sorted({case["group"] for case in cases})
    return dict(
        schema=ANALYSIS_SCHEMA,
        manifest=dict(source=manifest_path.resolve().as_posix(), sha256=manifest_hash),
        receipt=dict(source=receipt_path.resolve().as_posix(), sha256=receipt_hash),
        verification=dict(source_sha256_verified_after_run=True, images=len(cases),
                          measured_outputs=len(cases) * 3 * len(RECIPES), all_measured_outputs_sample_exact=True,
                          complete_sizes_deterministic=True, per_image_medians_recomputed=True,
                          complete_cost_breakdowns_reconciled=True, receipt_totals_reconciled=True),
        measurement_environment=dict(cpu="AMD Ryzen 9 7945HX", cpu_evidence="Registry verified by the parent agent",
                                     operating_system="Windows", architecture="x86_64", build_profile="release",
                                     lto="thin", codegen_units=1, threads=receipt["threads"],
                                     caveat="Environment supplied by the experiment owner; diagnostic local timings, not cross-machine performance claims."),
        source_snapshot=source_snapshot(),
        scope=dict(acceptance="Original decoded PNG samples including invisible RGB; not original PNG file bytes.",
                   timing=receipt.get("timing"), prior_use=manifest.get("previously_used_scope"),
                   zero_denominators="Ratios and throughput with zero denominators are null.",
                   comparison="Fixed recipes on every image; no best-of-all-tools selection.",
                   sizes="Aggregate complete encoded bytes, including every reported tool/table/header/framing byte.",
                   standard_breakdown="Standard JXL has a complete size measurement but no subsection breakdown.",
                   sample_verification="Exact flags come from native full-sample comparisons in the receipt; this analyzer does not rerun decoders."),
        summaries=dict(all=overall,
                       previously_used=summarize([case for case in cases if case["previously_used"]]),
                       absent_from_prior_records=summarize([case for case in cases if not case["previously_used"]]),
                       content_groups={group: summarize([case for case in cases if case["group"] == group]) for group in groups}),
        cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "out/pja-port-20261117/pixel-manifest.json")
    parser.add_argument("--receipt", type=Path, default=ROOT / "out/pja-port-20261117/pixel-corpus.json")
    parser.add_argument("--output", type=Path, default=ROOT / "out/pja-port-20261117/pixel-corpus-analysis.json")
    args = parser.parse_args()
    result = analyze(args.manifest, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["verification"], ensure_ascii=False))
    print(json.dumps(result["summaries"]["all"]["adaptive_vs_baselines"], ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Pixel receipt validation failed: {error}", file=sys.stderr)
        sys.exit(1)
