"""Read-only, independent Source/Delta study summaries with explicit cohort coverage.

--report REPORT --manifest MANIFEST --out PREFIX [--plot] writes summary JSON
and per-case CSV. Repeated reports stay separate; no codecs, images, classifiers,
builds or corpus jobs are invoked. Baselines are Source first and Delta absolute.

Failed/interrupted reports are permitted but remain PARTIAL, even when every
recorded case verified. Supply the original declared manifest to expose cases
never reached. Native-grouping full-cohort trials and predeclared full-frame/cap
sensitivity cohorts need separate reports and their own manifests; never pool
them or relabel an interrupted prefix as a complete study. Cohort membership
must be declared independently of experiment outcomes.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import summarize_patch_ma_followup as common
from benchmark_palette_delta import MODES, SCOPE, activation, validate_attempt

require = common.require
number = common.number
SOURCE_COUNTERS = ("retained_groups", "occurrences", "searched_groups", "variable_sample_groups",
                   "changed_groups", "candidates", "sampled_occurrences", "sampled_pixels",
                   "emitted_plans", "atlas_pixels", "dictionary_positions", "float_fallback_groups")
DELTA_CSV_COUNTERS = ("attempts", "exact_delta_attempted", "exact_delta_produced", "exact_delta_accepted",
                      "exact_zero_attempted", "exact_zero_accepted", "absolute_fallback_accepted",
                      "kernel_declined", "cost_rejected", "delta_accepted", "delta_used", "delta_pixels")


def timed_measurements(case, variant, config):
    """Use raw complete verified schedules, including when a checkpoint lacks summaries."""
    declared = case.get("variants", {})
    require(isinstance(declared, dict), "Invalid case variants")
    samples = declared.get(variant)
    require(isinstance(samples, list) and samples, f"{variant}: missing trials")
    repeats, warmups = config.get("repeats", 1), config.get("warmups", 0)
    number(repeats, "repeats", 1, integer=True)
    number(warmups, "warmups", integer=True)
    seen, timed = set(), []
    for sample in samples:
        common.check_sample(sample, variant)
        require(all(key in sample for key in ("encoder", "decoder")), f"{variant}: missing process verification")
        r, warmup = sample.get("repetition"), sample.get("warmup")
        require(type(r) is int and type(warmup) is bool and warmup == (r < 0), "Invalid trial schedule")
        require(r not in seen, "Duplicate repetition")
        seen.add(r)
        require("included_in_timing" not in sample or sample["included_in_timing"] is (not warmup),
                "Diagnostic mixed into timed trials")
        size, seconds, _ = common.check_measurement(sample.get("measurement"), case, variant)
        if not warmup:
            timed.append((r, size, seconds, sample["measurement"]))
    require(seen == set(range(-warmups, repeats)), f"{variant}: incomplete trials")
    timed.sort()
    require(len({row[1] for row in timed}) == 1, f"{variant}: varying timed bytes")
    size, seconds = timed[0][1], statistics.median(row[2] for row in timed)
    value = dict(bytes=size, seconds=seconds, bpp=8 * size / case["pixels"],
                 mpix_per_second=case["pixels"] / 1e6 / seconds)
    saved_summaries = case.get("summaries", {})
    require(isinstance(saved_summaries, dict), "Invalid case summaries")
    saved = saved_summaries.get(variant)
    if saved is not None:
        require(isinstance(saved, dict), "Invalid saved summary")
        for key, expected in value.items():
            if key in saved:
                common.close(saved[key], expected, "Saved summary " + key)
    return value, timed


def source_evidence(measurement, variant):
    """Plan-formation counters on a successful, verified single-encode lab output.

    emitted_plans is recorded after successful subtract/replay and plan formation,
    BEFORE frame encode. This lab has no later encode fallback, so a successful
    output implies emission; this is not a generic bitstream-emission counter.
    """
    work = measurement.get("representative_work")
    if variant == "off":
        require(work is None, "Off variant claims representative work")
        return dict(available=True, counters=dict.fromkeys(SOURCE_COUNTERS, 0), selection_seconds=0)
    if work is None:
        return dict(available=False, reason="Missing representative_work; enabled route is not activation")
    require(isinstance(work, dict) and work.get("mode") == variant, "Mismatched representative mode")
    counters = {key: number(work.get(key), key, integer=True) for key in SOURCE_COUNTERS}
    seconds = number(work.get("selection_seconds"), "selection_seconds")
    require(counters["changed_groups"] <= counters["variable_sample_groups"] <= counters["searched_groups"]
            <= counters["retained_groups"], "Inconsistent representative search counts")
    require(counters["emitted_plans"] <= 1
            and bool(counters["emitted_plans"]) == bool(counters["atlas_pixels"])
            == bool(counters["dictionary_positions"]), "Inconsistent emitted representative plan")
    if variant == "first":
        require(not counters["searched_groups"] and not counters["changed_groups"] and seconds == 0,
                "First baseline claims representative search")
    return dict(available=True, counters=counters, selection_seconds=seconds)


def delta_evidence(measurement, variant):
    e = measurement.get("palette_evidence")
    if not isinstance(e, dict) or e.get("scope") != SCOPE:
        return dict(available=False, reason="Missing invocation-global/local telemetry; configuration is not activation")
    require(e.get("route") in ("full_image", "groupwise") and isinstance(e.get("attempts"), list),
            "Invalid delta route/attempts")
    for attempt in e["attempts"]:
        validate_attempt(attempt, variant)
    if e["route"] == "groupwise":
        require(all(a["scope"] == "local" for a in e["attempts"]), "Groupwise claims global work")
    scopes = activation(measurement)
    accepted = sum(s["delta_accepted"] for s in scopes.values())
    used = sum(s["delta_used"] for s in scopes.values())
    fallback = sum(s["absolute_fallback_accepted"] for s in scopes.values())
    category = ("delta_used" if used else "accepted_unused" if accepted else
                "fallback_only" if fallback else "inactive")
    return dict(available=True, route=e["route"], scopes=scopes, classification=category)


def evidence_for_trials(timed, variant, kind):
    read = source_evidence if kind == "source" else delta_evidence
    values = [read(row[3], variant) for row in timed]
    first = dict(values[0], repetition=timed[0][0])
    # Counts describe one encode, not repeat-multiplied totals. Expose variation.
    def signature(value):
        return {k: v for k, v in value.items() if k != "selection_seconds"}
    first["varies_across_timed_trials"] = any(signature(v) != signature(values[0]) for v in values[1:])
    if kind == "source" and all(v["available"] for v in values):
        first["selection_seconds"] = statistics.median(v["selection_seconds"] for v in values)
    if first["varies_across_timed_trials"]:
        first["trial_evidence"] = [dict(v, repetition=row[0]) for row, v in zip(timed, values)]
    return first


def metrics(rows, pixels):
    size = sum(row["bytes"] for row in rows)
    seconds = math.fsum(row["seconds"] for row in rows)
    return dict(images=len(rows), pixels=pixels, total_bytes=size, total_mib=size / 2**20,
                mean_image_bpp=statistics.mean(row["bpp"] for row in rows), weighted_bpp=8 * size / pixels,
                native_seconds=seconds, mpix_per_second=pixels / 1e6 / seconds)


def source_totals(evidence):
    known = [e for e in evidence if e["available"]]
    result = dict(evidence_images=len(known), unavailable_images=len(evidence) - len(known),
                  varying_evidence_images=sum(e["varies_across_timed_trials"] for e in evidence))
    if not known:
        return dict(result, counters=None, selection_seconds=None, formed_groups=None,
                    **dict.fromkeys(("formed_images", "emitted_images", "searched_images", "variable_images",
                                     "changed_images", "formed_not_emitted_images", "inactive_images")))
    result["counters"] = {key: sum(e["counters"][key] for e in known) for key in SOURCE_COUNTERS}
    # 'formed' means retained representative groups, not raw detector components.
    result["formed_groups"] = result["counters"]["retained_groups"]
    result["selection_seconds"] = math.fsum(e["selection_seconds"] for e in known)
    for label, field in (("formed", "retained_groups"), ("emitted", "emitted_plans"),
                         ("searched", "searched_groups"), ("variable", "variable_sample_groups"),
                         ("changed", "changed_groups")):
        result[label + "_images"] = sum(e["counters"][field] > 0 for e in known)
    result["formed_not_emitted_images"] = sum(e["counters"]["retained_groups"] > 0
                                              and not e["counters"]["emitted_plans"] for e in known)
    result["inactive_images"] = sum(not e["counters"]["emitted_plans"] for e in known)
    return result


def delta_totals(evidence):
    known = [e for e in evidence if e["available"]]
    result = dict(evidence_images=len(known), unavailable_images=len(evidence) - len(known),
                  varying_evidence_images=sum(e["varies_across_timed_trials"] for e in evidence))
    if not known:
        return dict(result, global_scope=None, local_scope=None, routes=None,
                    **dict.fromkeys(("delta_used_images", "accepted_unused_images", "fallback_only_images",
                                     "inactive_images", "exact_delta_accepted_images", "exact_zero_accepted_images",
                                     "exact_delta_used_images", "exact_zero_used_images")))
    for scope in ("global", "local"):
        counters = known[0]["scopes"][scope]
        totals = {key: sum(e["scopes"][scope][key] for e in known) for key in counters}
        for name in ("delta_accepted", "delta_used", "exact_delta_accepted", "exact_zero_accepted"):
            totals[name + "_images"] = sum(e["scopes"][scope][name] > 0 for e in known)
        # Used exact delta/zero counts need both acceptance and references.
        totals["exact_delta_used_images"] = sum(e["scopes"][scope]["exact_delta_accepted"] > 0
                                                 and e["scopes"][scope]["delta_used"] > 0 for e in known)
        totals["exact_zero_used_images"] = sum(e["scopes"][scope]["exact_zero_accepted"] > 0
                                                and e["scopes"][scope]["delta_used"] > 0 for e in known)
        result[scope + "_scope"] = totals
    for category in ("delta_used", "accepted_unused", "fallback_only", "inactive"):
        result[category + "_images"] = sum(e["classification"] == category for e in known)
    for name in ("exact_delta_accepted", "exact_zero_accepted"):
        result[name + "_images"] = sum(any(s[name] for s in e["scopes"].values()) for e in known)
    result["exact_delta_used_images"] = sum(any(s["exact_delta_accepted"] and s["delta_used"]
                                               for s in e["scopes"].values()) for e in known)
    result["exact_zero_used_images"] = sum(any(s["exact_zero_accepted"] and s["delta_used"]
                                              for s in e["scopes"].values()) for e in known)
    result["routes"] = dict(Counter(e["route"] for e in known))
    return result


def group_totals(cases, variants, baseline, kind):
    pixels = sum(c["pixels"] for c in cases)
    output = {}
    base_rows = [c["variants"][baseline] for c in cases]
    for variant in variants:
        rows = [c["variants"][variant] for c in cases]
        deltas = [row["bytes"] - base["bytes"] for row, base in zip(rows, base_rows)]
        value = metrics(rows, pixels)
        value.update(size_wins=sum(d < 0 for d in deltas), size_ties=sum(d == 0 for d in deltas),
                     size_losses=sum(d > 0 for d in deltas), bytes_saved_on_wins=-sum(d for d in deltas if d < 0),
                     bytes_added_on_losses=sum(d for d in deltas if d > 0), net_bytes_saved=-sum(deltas))
        value[kind] = (source_totals if kind == "source" else delta_totals)([r["evidence"] for r in rows])
        cross_tab = {label: Counter() for label in ("win", "tie", "loss")}
        for row, delta in zip(rows, deltas):
            evidence = row["evidence"]
            category = ("unavailable" if not evidence["available"] else evidence["classification"]
                        if kind == "delta" else "emitted" if evidence["counters"]["emitted_plans"] else "not_emitted")
            cross_tab["win" if delta < 0 else "loss" if delta > 0 else "tie"][category] += 1
        value["activation_by_size_result"] = {label: dict(counts) for label, counts in cross_tab.items()}
        output[variant] = value
    base = output[baseline]
    for value in output.values():
        value["total_bytes_ratio_to_baseline"] = value["total_bytes"] / base["total_bytes"]
        value["native_seconds_ratio_to_baseline"] = value["native_seconds"] / base["native_seconds"]
    oracle = metrics([c["oracle"] for c in cases], pixels)
    oracle.update(selection_counts=dict(Counter(c["oracle"]["winner"] for c in cases)),
                  size_wins=sum(c["oracle"]["bytes"] < c["variants"][baseline]["bytes"] for c in cases),
                  size_ties=sum(c["oracle"]["bytes"] == c["variants"][baseline]["bytes"] for c in cases),
                  size_losses=0, net_bytes_saved=base["total_bytes"] - oracle["total_bytes"],
                  timing_contract="Sum of ALL variant per-case median native times, not winning-variant time",
                  total_bytes_ratio_to_baseline=oracle["total_bytes"] / base["total_bytes"],
                  native_seconds_ratio_to_baseline=oracle["native_seconds"] / base["native_seconds"])
    return dict(variants=output, oracle=oracle)


def summarize(report, manifest=None, manifest_dir=None, kind=None):
    require(isinstance(report, dict) and report.get("status") in ("passed", "failed", "running"), "Unknown report status")
    require(report.get("temporary_outputs_cleaned", True) is True, "Input cleanup/integrity failed")
    require(isinstance(report.get("errors", []), list), "Invalid input errors")
    require(not any("changed during" in str(e).lower() for e in report.get("errors", [])), "Input/binary identity changed")
    config = report.get("config", {})
    require(isinstance(config, dict), "Missing configuration")
    variants = config.get("variants", config.get("modes"))
    require(isinstance(variants, list) and variants and all(isinstance(v, str) for v in variants)
            and len(set(variants)) == len(variants), "Invalid declared variants")
    inferred = "delta" if "absolute" in variants else "source" if "first" in variants else None
    require(inferred is not None and (kind is None or kind == inferred), "Study kind/baseline mismatch")
    kind = inferred
    baseline = "first" if kind == "source" else "absolute"
    allowed = ("off", "first", "medoid", "entropy") if kind == "source" else MODES
    require(set(variants) <= set(allowed), "Unknown independent study variant")
    raw_cases = report.get("cases")
    require(isinstance(raw_cases, list) and raw_cases, "No report cases")
    metadata = common.metadata(manifest, manifest_dir)
    included, excluded, seen = [], [], set()
    order = [baseline] + [v for v in variants if v != baseline]
    for case in raw_cases:
        require(isinstance(case, dict), "Invalid case")
        key = common.source_key(case.get("source"))
        require(key not in seen, "Duplicate report source")
        seen.add(key)
        info = metadata.get(key) if metadata is not None else None
        require(metadata is None or info is not None, "Report source absent from stratum manifest")
        if info:
            for field in ("sha256", "width", "height", "pixels"):
                if field in case and field in info:
                    require(case[field] == info[field], "Manifest " + field + " mismatch")
        stratum = info["stratum"] if info else case.get("stratum", "unstratified")
        require(isinstance(stratum, str) and stratum, "Invalid stratum")
        if info and "stratum" in case:
            require(case["stratum"] == stratum, "Conflicting manifest stratum")
        value = dict(source=case["source"], sha256=case.get("sha256"), stratum=stratum, variants={})
        errors, trials = {}, {}
        try:
            require(not case.get("error"), str(case.get("error")))
            value["pixels"] = number(case.get("pixels"), "pixels", 1, integer=True)
            if "width" in case or "height" in case:
                w = number(case.get("width"), "width", 1, integer=True)
                h = number(case.get("height"), "height", 1, integer=True)
                require(w * h == value["pixels"], "Inconsistent case dimensions")
            for variant in variants:
                try:
                    value["variants"][variant], trials[variant] = timed_measurements(case, variant, config)
                except (ValueError, TypeError, KeyError) as error:
                    errors[variant] = str(error)
        except (ValueError, TypeError, KeyError) as error:
            errors["case"] = str(error)
        if errors:
            excluded.append(dict(source=case["source"], stratum=stratum, reasons=errors))
            continue
        # Missing telemetry remains unavailable, never a reason to drop verified bytes.
        for variant in variants:
            value["variants"][variant]["evidence"] = evidence_for_trials(trials[variant], variant, kind)
        if kind == "delta":
            fingerprints = {row[3]["invariant_options"] for t in trials.values() for row in t
                            if "invariant_options" in row[3]}
            require(len(fingerprints) <= 1, "Non-mode configuration changed in Delta ablation")
        winner = min(order, key=lambda v: value["variants"][v]["bytes"])
        size = value["variants"][winner]["bytes"]
        value["oracle"] = dict(winner=winner, bytes=size, bpp=8 * size / value["pixels"],
                               seconds=math.fsum(v["seconds"] for v in value["variants"].values()))
        included.append(value)
    require(included, "No common complete verified cases")
    missing = sorted(set(metadata or {}) - seen)
    totals = group_totals(included, variants, baseline, kind)
    return dict(schema_version=1, study=kind, baseline=baseline, variants=variants,
                status="partial" if excluded or missing or report["status"] != "passed" or report.get("errors") else "passed",
                input_status=report["status"], input_errors=report.get("errors", []), config=config,
                images=len(included), cases=included, excluded_cases=excluded,
                coverage=dict(report_cases=len(raw_cases), common_verified_cases=len(included), excluded_cases=len(excluded),
                              manifest_cases=len(metadata) if metadata is not None else None, missing_manifest_sources=missing),
                contracts=dict(cohort="All declared variants on the same common complete verified cases; exclusions listed, reports never pooled. Failed/interrupted input stays partial; provide the original declared manifest, not a completion-filtered replacement. Predeclared sensitivity cohorts use their own manifest, separately from full-cohort native-grouping trials",
                               timing="Sum of per-case median native seconds, warmups excluded",
                               counters="First verified timed repetition per image, not multiplied by repeats; variability exposed; selection_seconds uses per-image median",
                               source="formed_groups aliases retained_groups. emitted_plans records successful subtract/replay and plan formation BEFORE frame encode; on successful single-encode patch_lab outputs there is no later fallback, so it implies emission, not a generic emission counter. Enabled routes and proxy scores are not byte wins",
                               delta="Accepted tables and actual delta references separate, global/local included; reference counts are not unique input pixels",
                               oracle="Baseline wins size ties; best bytes paired with cost of ALL variants, not a deployable router",
                               charts="Aggregate points only, no error bars or significance claim"),
                overall=totals["variants"], oracle=totals["oracle"],
                strata={s: group_totals([c for c in included if c["stratum"] == s], variants, baseline, kind)
                        for s in sorted({c["stratum"] for c in included})})


def csv_rows(summary):
    for case in summary["cases"]:
        base = case["variants"][summary["baseline"]]
        for variant, value in case["variants"].items():
            delta = value["bytes"] - base["bytes"]
            evidence = value["evidence"]
            row = dict(study_status=summary["status"], common_verified_cases=summary["images"],
                       manifest_cases=summary["coverage"]["manifest_cases"],
                       source=case["source"], sha256=case["sha256"], stratum=case["stratum"], pixels=case["pixels"],
                       variant=variant, baseline=summary["baseline"], bytes=value["bytes"], native_seconds=value["seconds"],
                       bpp=value["bpp"], mpix_per_second=value["mpix_per_second"], baseline_bytes=base["bytes"],
                       byte_delta=delta, size_result="win" if delta < 0 else "loss" if delta > 0 else "tie",
                       oracle_winner=case["oracle"]["winner"], oracle_bytes=case["oracle"]["bytes"],
                       oracle_all_variants_seconds=case["oracle"]["seconds"],
                       evidence_available=evidence["available"],
                       evidence_varies_across_trials=evidence["varies_across_timed_trials"],
                       evidence_repetition=evidence["repetition"])
            if summary["study"] == "source":
                for key in SOURCE_COUNTERS:
                    row["source_" + key] = evidence["counters"][key] if evidence["available"] else None
                row["source_formed_groups"] = row["source_retained_groups"]
                row["source_selection_seconds"] = evidence.get("selection_seconds")
            else:
                row["delta_route"] = evidence.get("route")
                row["delta_classification"] = evidence.get("classification")
                for scope in ("global", "local"):
                    counters = evidence["scopes"][scope] if evidence["available"] else None
                    for key in DELTA_CSV_COUNTERS:
                        row[f"delta_{scope}_{key}"] = counters[key] if counters is not None else None
                    for name in ("exact_delta", "exact_zero"):
                        row[f"delta_{scope}_{name}_used"] = (bool(counters[name + "_accepted"] and counters["delta_used"])
                                                             if counters is not None else None)
            row["evidence_json"] = json.dumps(evidence, sort_keys=True, allow_nan=False)
            yield row


def plot_summary(summary, paths):
    # Reuse the existing no-errorbar plotter. The oracle is a separate all-cost point.
    expected = summary["coverage"]["manifest_cases"]
    cohort = f"{summary['images']} of {expected} manifest cases" if expected is not None else f"{summary['images']} common cases; manifest unavailable"
    config = summary["config"]
    buffering = ("grouping=report defaults" if "full_frame" not in config else
                 "full-frame" if config["full_frame"] else "native grouping")
    cap = config.get("palette_colors")
    label = (f"{Path(summary.get('input_report', 'in-memory')).name}\n"
             f"[{summary['status'].upper()}; {cohort}]\n"
             f"{buffering}; cap={cap if cap is not None else 'projected'}")
    plotted = dict(summary, input_report=label,
                   overall=dict(summary["overall"], **{"oracle (all variant cost)": summary["oracle"]}))
    common.plot_summary(plotted, paths)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, action="append", required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--kind", choices=("source", "delta"))
    p.add_argument("--out", type=Path, required=True, help="Output prefix, not a directory")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)
    try:
        inputs = {v.resolve() for v in args.report}
        require(len(inputs) == len(args.report), "Duplicate report")
        if args.manifest:
            inputs.add(args.manifest.resolve())
        manifest = common.load_json(args.manifest) if args.manifest else None
        jobs, targets = [], set()
        for path in args.report:
            result = summarize(common.load_json(path), manifest,
                               args.manifest.resolve().parent if args.manifest else None, args.kind)
            result["input_report"] = str(path.resolve())
            result["input_manifest"] = str(args.manifest.resolve()) if args.manifest else None
            prefix = str(args.out) + ("." + path.stem if len(args.report) > 1 else "")
            outputs = [Path(prefix + suffix) for suffix in (".summary.json", ".cases.csv")]
            if args.plot:
                outputs += [Path(prefix + suffix) for suffix in (".bpp-v-time.png", ".totalMiB-v-time.png")]
            for target in outputs:
                resolved = target.resolve()
                require(resolved not in inputs and resolved not in targets and not target.exists(),
                        f"Output exists or collides; never overwritten: {target}")
                targets.add(resolved)
            jobs.append((result, outputs))
        # Reserve ALL outputs before writing, including plots; no overwrite races.
        with ExitStack() as stack:
            opened = []
            for result, outputs in jobs:
                outputs[0].parent.mkdir(parents=True, exist_ok=True)
                streams = [stack.enter_context(path.open("xb")) for path in outputs]
                opened.append((result, outputs, streams))
            for result, outputs, streams in opened:
                streams[0].write((json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii"))
                # csv.writer needs a text stream, but the exclusively reserved descriptor stays owned here.
                import io
                text = io.StringIO(newline="")
                rows = list(csv_rows(result))
                writer = csv.DictWriter(text, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
                streams[1].write(text.getvalue().encode("utf-8"))
                if args.plot:
                    plot_summary(result, streams[2:])
                print(f"{result['input_report']}: {result['images']} common verified cases ({result['status']}) -> {outputs[0]}")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        print(f"Summary failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
