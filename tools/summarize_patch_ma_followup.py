"""Read-only summaries of complete patch-MA/refinement benchmark reports.

Usage: --report input.json --baseline off --out out/prefix [--manifest cases.json]
[--plot]. Outputs: prefix.summary.json, prefix.summary.md, and optionally
prefix.bpp-v-time.png and prefix.totalMiB-v-time.png. Repeated --report options
produce separate prefix.REPORT_STEM.* files; reports are NEVER pooled.

Bytes must be stable across timed repetitions: totals represent actual outputs,
not fractional median sizes. Native time is the sum of per-image medians.
Offset counters use only each image's first verified timed sample (lowest
repetition), never multiplied by repeats. Missing counters are not zero.
Patch activation uses separate diagnostic patch_application evidence only.
Manifest cases use original (or source) and optional stratum; extra metadata
cases do not enlarge the report's cohort. Relative manifest paths are resolved
against the manifest directory. No input images or executables are opened.
"""

import argparse
import json
import math
import ntpath
from pathlib import Path, PureWindowsPath
import statistics
import sys


OFFSET_COUNTERS = (
    "offset_proposals", "offset_wins", "offset_leaves", "offset_samples",
    "offset_population", "offset_candidates", "offset_token_fits",
    "offset_reused_tokens", "offset_raw_reused_tokens",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, where, minimum=0, integer=False):
    require(type(value) is int if integer else type(value) in (int, float),
            f"{where}: invalid number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and value >= minimum, f"{where}: invalid number")
    return value


def close(actual, expected, where, rel_tol=1e-9):
    number(actual, where)
    require(math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=1e-10),
            f"{where}: inconsistent value (expected {expected})")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def finite_float(value):
    return number(float(value), "JSON float", minimum=-math.inf)


def load_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=unique_object,
                         parse_constant=reject_constant, parse_float=finite_float)


def source_key(source, directory=None):
    require(isinstance(source, str) and bool(source.strip()), "Missing source path")
    if PureWindowsPath(source).is_absolute():
        return ntpath.normcase(ntpath.normpath(source))
    path = Path(source)
    resolved = str(((directory or Path.cwd()) / path).resolve())
    return ntpath.normcase(resolved) if PureWindowsPath(resolved).is_absolute() else resolved


def metadata(manifest, directory):
    if manifest is None:
        return None
    require(isinstance(manifest, dict), "Invalid manifest")
    cases = manifest.get("cases")
    require(isinstance(cases, list) and bool(cases), "Empty manifest cases")
    result = {}
    for case in cases:
        require(isinstance(case, dict), "Invalid manifest case")
        key = source_key(case.get("original", case.get("source")), directory)
        require(key not in result, "Duplicate manifest source")
        stratum = case.get("stratum", "unstratified")
        require(isinstance(stratum, str) and bool(stratum.strip()), "Invalid stratum")
        result[key] = dict(case, stratum=stratum)
    return result


def check_sample(sample, where):
    require(isinstance(sample, dict) and sample.get("verified") is True
            and not sample.get("error"), f"{where}: failed/unverified sample")
    for process in ("encoder", "decoder"):
        if process in sample:
            receipt = sample[process]
            require(isinstance(receipt, dict) and type(receipt.get("returncode")) is int
                    and receipt["returncode"] == 0, f"{where}: failed {process}")


def check_measurement(measurement, case, where):
    require(isinstance(measurement, dict), f"{where}: missing measurement")
    size = number(measurement.get("bytes"), where + " bytes", 1, integer=True)
    seconds = number(measurement.get("seconds"), where + " seconds")
    require(seconds > 0, f"{where}: nonpositive seconds")
    for key, expected in (("bpp", 8 * size / case["pixels"]),
                          ("mpix_per_second", case["pixels"] / 1e6 / seconds)):
        if key in measurement:
            close(measurement[key], expected, where + " " + key, rel_tol=1e-5)
    for key in ("width", "height", "pixels"):
        if key in case and key in measurement:
            require(type(measurement[key]) is int and measurement[key] == case[key],
                    f"{where}: inconsistent {key}")
    refinement = measurement.get("refinement")
    require(refinement is None or isinstance(refinement, dict),
            f"{where}: invalid refinement receipt")
    counters = {}
    if refinement is not None:
        for key in OFFSET_COUNTERS:
            if key in refinement:
                counters[key] = number(refinement[key], where + " " + key, integer=True)
    return size, seconds, counters


def timed_case(case, variant, config):
    where = f"{case['source']}/{variant}"
    samples = case["variants"][variant]
    require(isinstance(samples, list) and bool(samples), f"{where}: missing trials")
    schedule, timed = set(), []
    for sample in samples:
        check_sample(sample, where)
        repetition, warmup = sample.get("repetition"), sample.get("warmup")
        require(type(repetition) is int and type(warmup) is bool
                and warmup == (repetition < 0), f"{where}: invalid repetition/warmup")
        require(repetition not in schedule, f"{where}: duplicate repetition")
        schedule.add(repetition)
        require("included_in_timing" not in sample
                or sample["included_in_timing"] is (not warmup),
                f"{where}: inconsistent timing inclusion")
        size, seconds, counters = check_measurement(sample.get("measurement"), case, where)
        if not warmup:
            timed.append((repetition, size, seconds, counters))
    require(bool(timed), f"{where}: no timed trials")
    repeats = config.get("repeats", len(timed))
    warmups = config.get("warmups", len(samples) - len(timed))
    number(repeats, "repeats", 1, integer=True)
    number(warmups, "warmups", integer=True)
    require(schedule == set(range(-warmups, repeats)), f"{where}: incomplete trials")
    timed.sort()
    require(len({trial[1] for trial in timed}) == 1,
            f"{where}: varying timed bytes; no single actual output size")
    size = timed[0][1]
    seconds = statistics.median(trial[2] for trial in timed)
    summary = case["summaries"][variant]
    require(isinstance(summary, dict), f"{where}: missing summary")
    number(summary.get("bytes"), where + " summary bytes", minimum=1)
    require(summary["bytes"] == size, f"{where}: inconsistent summary bytes")
    for key, expected in (("seconds", seconds),
                          ("bpp", 8 * size / case["pixels"])):
        close(summary.get(key), expected, where + " summary " + key)
    if "mpix_per_second" in summary:
        close(summary["mpix_per_second"], case["pixels"] / 1e6 / seconds,
              where + " summary MPix/s", rel_tol=1e-5)
    if "sizes" in summary:
        require(summary["sizes"] == [size], f"{where}: inconsistent summary sizes")
    for key, expected in (("min_seconds", min(t[2] for t in timed)),
                          ("max_seconds", max(t[2] for t in timed))):
        if key in summary:
            close(summary[key], expected, where + " summary " + key)
    return dict(bytes=size, seconds=seconds, bpp=8 * size / case["pixels"],
                offset_counters=timed[0][3]), schedule


def diagnostic_case(samples, where):
    require(isinstance(samples, list), f"{where}: invalid diagnostics")
    selections, reference, residual = [], [], []
    for sample in samples:
        check_sample(sample, where)
        require("included_in_timing" not in sample or sample["included_in_timing"] is False,
                f"{where}: diagnostic included in timing")
        patch = sample.get("patch_application")
        require(isinstance(patch, dict) and type(patch.get("source_patch_selected")) is bool,
                f"{where}: missing actual Patch activation evidence")
        selected = patch["source_patch_selected"]
        selections.append(selected)
        for key, target in (("reference_seconds", reference), ("residual_seconds", residual)):
            if selected:
                target.append(number(patch.get(key), where + " " + key))
            else:
                require(patch.get(key) is None, f"{where}: inactive Patch has stage timing")
                target.append(0)
    require(len(set(selections)) <= 1, f"{where}: inconsistent diagnostic activation")
    return dict(source_patch_selected=selections[0] if selections else None,
                diagnostic_samples=len(samples),
                reference_seconds=statistics.median(reference) if reference else None,
                residual_seconds=statistics.median(residual) if residual else None)


def totals(cases, variants, baseline):
    pixels = sum(case["pixels"] for case in cases)
    output = {}
    for variant in variants:
        rows = [case["variants"][variant] for case in cases]
        size = sum(row["bytes"] for row in rows)
        seconds = math.fsum(row["seconds"] for row in rows)
        counter_keys = rows[0]["offset_counters"].keys()
        require(all(row["offset_counters"].keys() == counter_keys for row in rows),
                f"{variant}: inconsistent Offset counter coverage")
        comparisons = []
        for case, row in zip(cases, rows):
            base = case["variants"][baseline]
            comparisons.append(dict(
                source=case["source"], stratum=case["stratum"],
                baseline_bytes=base["bytes"], bytes=row["bytes"],
                byte_delta=row["bytes"] - base["bytes"],
                size_change_pct=100 * (row["bytes"] / base["bytes"] - 1),
                baseline_seconds=base["seconds"], seconds=row["seconds"],
                seconds_delta=row["seconds"] - base["seconds"],
                time_change_pct=100 * (row["seconds"] / base["seconds"] - 1)))
        regressions = {}
        for name, field in (("largest_byte_increases", "byte_delta"),
                            ("largest_size_regressions_pct", "size_change_pct"),
                            ("largest_time_regressions_seconds", "seconds_delta"),
                            ("largest_time_regressions_pct", "time_change_pct")):
            regressions[name] = sorted((r for r in comparisons if r[field] > 0),
                                       key=lambda r: (-r[field], r["source"]))[:10]
        evidence = [row["patch_diagnostics"] for row in rows]
        coverage = sum(row["source_patch_selected"] is not None for row in evidence)
        output[variant] = dict(
            images=len(cases), pixels=pixels, total_bytes=size, total_mib=size / 2**20,
            mean_image_bpp=statistics.mean(row["bpp"] for row in rows),
            weighted_bpp=8 * size / pixels, native_seconds=seconds,
            mpix_per_second=pixels / 1e6 / seconds,
            byte_increases=sum(r["byte_delta"] > 0 for r in comparisons),
            byte_decreases=sum(r["byte_delta"] < 0 for r in comparisons),
            byte_ties=sum(r["byte_delta"] == 0 for r in comparisons),
            offset_counters={key: sum(row["offset_counters"][key] for row in rows)
                             for key in counter_keys},
            offset_counter_images=len(rows) if counter_keys else 0,
            source_patch_activation_count=(sum(row["source_patch_selected"] is True
                                               for row in evidence) if coverage else None),
            source_patch_diagnostic_images=coverage,
            diagnostic_samples=sum(row["diagnostic_samples"] for row in evidence),
            regressions=regressions)
    base = output[baseline]
    for value in output.values():
        value["total_bytes_ratio_to_baseline"] = value["total_bytes"] / base["total_bytes"]
        value["native_seconds_ratio_to_baseline"] = value["native_seconds"] / base["native_seconds"]
    return output


def summarize(report, baseline, manifest=None, manifest_dir=None):
    require(isinstance(report, dict) and report.get("status") == "passed"
            and not report.get("errors"), "Report failed/incomplete or contains errors")
    require(report.get("temporary_outputs_cleaned", True) is True,
            "Report cleanup/integrity incomplete")
    cases = report.get("cases")
    require(isinstance(cases, list) and bool(cases), "Empty/missing cases")
    config = report.get("config", {})
    require(isinstance(config, dict), "Invalid config")
    require(isinstance(cases[0], dict) and isinstance(cases[0].get("summaries"), dict),
            "Missing case summaries")
    variants = config.get("variants", config.get("modes", list(cases[0]["summaries"])))
    require(isinstance(variants, list) and bool(variants)
            and all(isinstance(v, str) and v.strip() for v in variants)
            and len(set(variants)) == len(variants), "Invalid variants")
    require(baseline in variants, f"Missing baseline variant: {baseline}")
    if "completed" in report:
        require(type(report["completed"]) is int and report["completed"] == len(cases),
                "Incomplete completed case count")
    strata = metadata(manifest, manifest_dir)
    seen, normalized = set(), []
    expected_schedule, expected_diagnostics = None, None
    for case in cases:
        require(isinstance(case, dict) and not case.get("error"), "Failed/invalid case")
        key = source_key(case.get("source"))
        require(key not in seen, "Duplicate case source")
        seen.add(key)
        pixels = number(case.get("pixels"), "pixels", 1, integer=True)
        if "width" in case or "height" in case:
            width = number(case.get("width"), "width", 1, integer=True)
            height = number(case.get("height"), "height", 1, integer=True)
            require(pixels == width * height, "Inconsistent case pixels/dimensions")
        for field in ("variants", "summaries"):
            require(isinstance(case.get(field), dict) and set(case[field]) == set(variants),
                    f"{case['source']}: missing/extra {field} variants")
        diagnostics = case.get("diagnostics", {v: [] for v in variants})
        require(isinstance(diagnostics, dict) and set(diagnostics) == set(variants),
                "Missing/extra diagnostic variants")
        stratum = case.get("stratum", "unstratified")
        if strata is not None:
            require(key in strata, f"Missing manifest metadata: {case['source']}")
            info = strata[key]
            stratum = info["stratum"]
            for field in ("sha256", "pixels", "width", "height"):
                if field in info and field in case:
                    require(info[field] == case[field], f"Manifest {field} mismatch")
            if "stratum" in case:
                require(case["stratum"] == stratum, "Manifest stratum mismatch")
        require(isinstance(stratum, str) and bool(stratum.strip()), "Invalid stratum")
        item = dict(source=case["source"], pixels=pixels, stratum=stratum, variants={})
        for variant in variants:
            value, schedule = timed_case(case, variant, config)
            require(expected_schedule is None or schedule == expected_schedule,
                    "Inconsistent case/variant trial schedules")
            expected_schedule = schedule
            diagnostic = diagnostic_case(diagnostics[variant], case["source"] + "/" + variant)
            count = diagnostic["diagnostic_samples"]
            require(expected_diagnostics is None or count == expected_diagnostics,
                    "Incomplete/inconsistent diagnostic coverage")
            expected_diagnostics = count
            if config.get("profile") is True:
                require(count == 1, "Missing requested profile diagnostic")
            value["patch_diagnostics"] = diagnostic
            item["variants"][variant] = value
        normalized.append(item)
    overall = totals(normalized, variants, baseline)
    if "aggregate" in report:
        aggregate = report["aggregate"]
        require(isinstance(aggregate, dict) and set(aggregate) == set(variants),
                "Incomplete aggregate variants")
        for variant, value in aggregate.items():
            require(isinstance(value, dict) and value.get("status", "passed") == "passed",
                    "Incomplete aggregate")
            fields = (("total_bytes", "total_bytes"), ("total_median_seconds", "native_seconds"),
                      ("mean_bpp", "mean_image_bpp"), ("weighted_bpp", "weighted_bpp"),
                      ("mpix_per_second", "mpix_per_second")) if "total_bytes" in value else (
                      ("bytes", "total_bytes"), ("seconds", "native_seconds"),
                      ("bpp", "weighted_bpp"), ("mpix_per_second", "mpix_per_second"))
            for key, field in fields:
                if field == "total_bytes":
                    number(value.get(key), f"{variant} aggregate bytes", minimum=1)
                    require(value[key] == overall[variant][field],
                            f"{variant} aggregate bytes: inconsistent value")
                else:
                    close(value.get(key), overall[variant][field], f"{variant} aggregate {key}")
    return dict(
        schema_version=1, status="passed", baseline=baseline, images=len(normalized),
        sources=[case["source"] for case in normalized],
        contracts=dict(
            cohort="This input report only; all variants use all cases; never pooled across reports.",
            bytes="One stable actual encoded size per image; sum once, not across repeats.",
            bpp="mean_image_bpp = arithmetic mean of image bpp; weighted_bpp = 8*total_bytes/pixels.",
            timing="native_seconds = sum of per-image median native seconds; warmups/diagnostics excluded.",
            throughput="MPix/s = total_pixels/1e6/native_seconds; MiB = bytes/2^20.",
            offset="Sum per-image first verified timed sample (lowest repetition); no multiplication by repeats. Missing counters unavailable, not zero.",
            patch="Count images with actual diagnostic source_patch_selected=true; enabled routes are not activation evidence. Missing diagnostics unavailable.",
            plots="Mean-only aggregate points; no error bars or statistical significance claim."),
        overall=overall,
        strata={s: totals([c for c in normalized if c["stratum"] == s], variants, baseline)
                for s in sorted({c["stratum"] for c in normalized})},
        cases=normalized)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown(summary):
    lines = ["# Patch-MA Followup Summary", "",
             f"Report: {cell(summary.get('input_report', '(in-memory)'))}",
             f"Images: {summary['images']}; baseline: {cell(summary['baseline'])}.", ""]
    lines.extend(summary["contracts"].values())
    lines.append("")
    groups = [("Overall", summary["overall"])]
    groups.extend(("Stratum: " + s, group) for s, group in summary["strata"].items())
    for name, group in groups:
        lines += ["## " + cell(name), "",
                  "| Variant | Images | Actual bytes | MiB | Mean image bpp | Weighted bpp | Native s | MPix/s | Bytes +/-/= | Size ratio | Time ratio | Patch active/diagnosed |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
        for variant, value in group.items():
            active = value["source_patch_activation_count"]
            patch = "unavailable" if active is None else f"{active}/{value['source_patch_diagnostic_images']}"
            lines.append(
                f"| {cell(variant)} | {value['images']} | {value['total_bytes']} | "
                f"{value['total_mib']:.6f} | {value['mean_image_bpp']:.6f} | "
                f"{value['weighted_bpp']:.6f} | {value['native_seconds']:.6f} | "
                f"{value['mpix_per_second']:.6f} | {value['byte_increases']}/"
                f"{value['byte_decreases']}/{value['byte_ties']} | "
                f"{value['total_bytes_ratio_to_baseline']:.6f} | "
                f"{value['native_seconds_ratio_to_baseline']:.6f} | {patch} |")
        lines += ["", "### Offset Counters", ""]
        for variant, value in group.items():
            counters = value["offset_counters"]
            text = ", ".join(f"{k}={v}" for k, v in counters.items()) if counters else "unavailable"
            lines.append(f"- {cell(variant)}: {text}")
        lines.append("")
    lines += ["## Largest Regressions Vs Baseline", ""]
    for variant, value in summary["overall"].items():
        if variant == summary["baseline"]:
            continue
        lines += ["### " + cell(variant), ""]
        for name, rows in value["regressions"].items():
            lines += [name.replace("_", " ") + ":", ""]
            if not rows:
                lines.append("- None.")
            for row in rows:
                lines.append(f"- {cell(row['source'])}: {row['byte_delta']:+d} bytes "
                             f"({row['size_change_pct']:+.3f}%), "
                             f"{row['seconds_delta']:+.6f} s ({row['time_change_pct']:+.3f}%).")
            lines.append("")
    return "\n".join(lines)


def plot_summary(summary, paths):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError("--plot requires matplotlib") from error
    for field, ylabel, path in zip(
            ("mean_image_bpp", "total_mib"),
            ("Arithmetic mean per-image bpp", "Total encoded MiB (2^20 bytes)"), paths):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        try:
            for variant, value in summary["overall"].items():
                x, y = value["native_seconds"], value[field]
                ax.scatter([x], [y], label=variant)
            ax.set(xlabel="Sum of per-image median native encode time (s)", ylabel=ylabel,
                   title=f"{summary['images']} images: {Path(summary['input_report']).name}\n"
                         "Mean-only aggregate points; no uncertainty estimate")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            ax.grid(alpha=0.2)
            fig.savefig(path, dpi=140)
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output filename prefix")
    parser.add_argument("--manifest", type=Path, help="Optional stratum metadata")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_json(args.manifest) if args.manifest else None
        inputs = {path.resolve() for path in args.report}
        require(len(inputs) == len(args.report), "Duplicate input report")
        if args.manifest:
            inputs.add(args.manifest.resolve())
        jobs, targets = [], set()
        # Validate every report and reserve distinct names before producing output.
        for path in args.report:
            summary = summarize(load_json(path), args.baseline, manifest,
                                args.manifest.resolve().parent if args.manifest else None)
            summary["input_report"] = str(path.resolve())
            prefix = str(args.out) + ("." + path.stem if len(args.report) > 1 else "")
            outputs = [Path(prefix + suffix) for suffix in
                       (".summary.json", ".summary.md", ".bpp-v-time.png", ".totalMiB-v-time.png")]
            selected = outputs if args.plot else outputs[:2]
            for target in selected:
                resolved = target.resolve()
                require(resolved not in inputs and resolved not in targets,
                        "Output collides with input/other output")
                require(not target.exists(), f"Output exists; never overwritten: {target}")
                targets.add(resolved)
            jobs.append((summary, outputs))
        for summary, outputs in jobs:
            outputs[0].parent.mkdir(parents=True, exist_ok=True)
            if args.plot:
                plot_summary(summary, outputs[2:])
            with outputs[0].open("x", encoding="ascii", newline="\n") as stream:
                json.dump(summary, stream, indent=2, ensure_ascii=True, allow_nan=False)
                stream.write("\n")
            with outputs[1].open("x", encoding="ascii", errors="backslashreplace", newline="\n") as stream:
                stream.write(markdown(summary))
            print(f"{summary['input_report']}: {summary['images']} images -> {outputs[0]}")
    except (OSError, ValueError, TypeError, OverflowError) as error:
        print(f"Summary failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
