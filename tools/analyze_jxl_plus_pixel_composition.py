"""Validate and summarize eight-recipe pixel composition receipts; runs no codec."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import analyze_jxl_plus_pixel_corpus as core

ROOT = core.ROOT
RECIPES = ("plus_static", "plus_fitted_zero_gate", "plus_adaptive_zero_gate",
           "plus_static_lz", "plus_fitted_zero_gate_lz", "plus_adaptive_zero_gate_lz",
           "plus_selected", "standard_jxl_default")
ADAPTIVE_LZ = RECIPES[5]
SELECTED = RECIPES[6]
_validate_case = core.validate_case
_summarize = core.summarize
_sources = core.EXPERIMENT_SOURCES


def validate_case(case, measured):
    # Counts are deliberately outside the byte breakdown: neither a group count
    # nor a selection histogram contributes bytes to the complete-cost sum.
    cleaned = copy.deepcopy(measured)
    counts = {}
    for variant in cleaned.get("variants", []):
        recipe = variant.get("recipe")
        if recipe == "standard_jxl_default":
            continue
        breakdown = variant.get("byte_breakdown")
        core.require(isinstance(breakdown, dict), f"{case['label']}/{recipe}: missing byte breakdown")
        selected = variant.get("selected_groups")
        lz = variant.get("lz_groups")
        core.require(isinstance(selected, list) and len(selected) == 6,
                     f"{case['label']}/{recipe}: expected six selection counts")
        selected = [core.integer(x, f"{recipe} selected_groups") for x in selected]
        lz = core.integer(lz, f"{recipe} lz_groups")
        groups = core.integer(breakdown.get("groups"), f"{recipe} groups")
        core.require(sum(selected) == groups and sum(selected[3:]) == lz,
                     f"{case['label']}/{recipe}: selection/group/LZ counts disagree")
        if recipe in RECIPES[:6]:
            index = RECIPES.index(recipe)
            # An LZ-enabled recipe may emit no LZ commands and therefore signal
            # the matching non-LZ mode. Histogram bins reflect actual stream
            # flags, not merely the requested encoder setting.
            allowed = (index,) if index < 3 else (index % 3, index)
            core.require(sum(selected[i] for i in allowed) == groups,
                         f"{case['label']}/{recipe}: fixed recipe selected a different mode")
        elif recipe == SELECTED:
            core.require(selected[1] == 0 and selected[4] == 0,
                         f"{case['label']}: Selected used a fitted mode outside its declared search")
        counts[recipe] = dict(lz_groups=lz, selected_groups=selected)
    result = _validate_case(case, cleaned)
    chosen = result["variants"][SELECTED]["complete_bytes"]
    for baseline in (RECIPES[0], RECIPES[2], RECIPES[3], RECIPES[5]):
        core.require(chosen <= result["variants"][baseline]["complete_bytes"],
                     f"{case['label']}: complete group selection exceeded candidate {baseline}")
    for recipe, distribution in counts.items():
        result["variants"][recipe]["group_selection"] = distribution
    return result


def comparison(summary, cases, candidate, baseline):
    a, b = summary["variants"][candidate], summary["variants"][baseline]
    savings = b["complete_bytes"] - a["complete_bytes"]
    differences = [c["variants"][baseline]["complete_bytes"]
                   - c["variants"][candidate]["complete_bytes"] for c in cases]
    result = dict(
        baseline_complete_bytes=b["complete_bytes"], candidate_complete_bytes=a["complete_bytes"],
        saved_bytes=savings, savings_percent=core.ratio(100 * savings, b["complete_bytes"]),
        candidate_over_baseline_size_ratio=core.ratio(a["complete_bytes"], b["complete_bytes"]),
        wins=sum(d > 0 for d in differences), ties=sum(d == 0 for d in differences),
        losses=sum(d < 0 for d in differences))
    for direction in ("encode", "decode"):
        key = f"sum_per_image_median_{direction}_seconds"
        result[f"{direction}_seconds_candidate_minus_baseline"] = a[key] - b[key]
        result[f"{direction}_time_candidate_over_baseline"] = core.ratio(a[key], b[key])
        result[f"{direction}_throughput_candidate_over_baseline"] = core.ratio(b[key], a[key])
    if baseline == "standard_jxl_default":
        result["saved_bytes_by_section"] = None
    else:
        sa, sb = a["summed_byte_breakdown"], b["summed_byte_breakdown"]
        result["saved_bytes_by_section"] = {key: sb[key] - sa[key] for key in core.BYTE_FIELDS}
        core.require(sum(result["saved_bytes_by_section"].values()) == savings,
                     f"{candidate}/{baseline}: section deltas do not reconcile")
    return result


def summarize(cases):
    result = _summarize(cases)
    # Keep the original adaptive comparison as an additional non-LZ control.
    for recipe in RECIPES[:-1]:
        values = [c["variants"][recipe]["group_selection"] for c in cases]
        result["variants"][recipe]["summed_group_selection"] = dict(
            lz_groups=sum(v["lz_groups"] for v in values),
            selected_groups=[sum(v["selected_groups"][i] for v in values) for i in range(6)])
    result["comparisons"] = {
        candidate: {baseline: comparison(result, cases, candidate, baseline)
                    for baseline in RECIPES if baseline != candidate}
        for candidate in (ADAPTIVE_LZ, SELECTED)}
    return result


def analyze(manifest_path, receipt_path):
    settings = dict(RECIPES=RECIPES, EXPERIMENT="pixel_lossless_composition",
                    ANALYSIS_SCHEMA="rs-jxl-plus.pixel-composition-analysis.v1",
                    EXPERIMENT_SOURCES=_sources + (
                        "crates/rs-jxl-plus/src/pixel_compose_tests.rs",
                        "crates/rs-jxl-plus/examples/pixel_lab.rs",
                        "vendor/jxl-rs/jxl/src/entropy_coding/decode.rs",
                        "tools/analyze_jxl_plus_pixel_composition.py",),
                    validate_case=validate_case, summarize=summarize)
    previous = {key: getattr(core, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(core, key, value)
        result = core.analyze(manifest_path, receipt_path)
    finally:
        for key, value in previous.items():
            setattr(core, key, value)
    result["scope"]["comparison"] = (
        "Fixed Static/Fitted/Adaptive recipes with and without LZ, and actual encoded Selected. "
        "Selected encode timing includes all candidate searches; its stream signals each chosen group mode. "
        "Selected is not an offline best-per-image oracle and its complete descriptor/framing costs are counted.")
    result["scope"]["selection_order"] = [RECIPES[i] for i in (0, 2, 3, 5)]
    result["scope"]["selection_histogram_order"] = list(RECIPES[:6])
    result["scope"]["counts_are_not_byte_cost"] = "selected_groups and lz_groups are group counts, outside byte-cost sums."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "out/pja-port-20261117/pixel-manifest.json")
    parser.add_argument("--receipt", type=Path, default=ROOT / "out/pja-port-20261117/pixel-composition.json")
    parser.add_argument("--output", type=Path, default=ROOT / "out/pja-port-20261117/pixel-composition-analysis.json")
    args = parser.parse_args()
    result = analyze(args.manifest, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["verification"], ensure_ascii=False))
    print(json.dumps(result["summaries"]["all"]["comparisons"], ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Pixel composition receipt validation failed: {error}", file=sys.stderr)
        sys.exit(1)
