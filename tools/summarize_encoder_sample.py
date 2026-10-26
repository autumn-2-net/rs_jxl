"""Compare encoder-tool reports without running codecs or reading source images.

Baselines are ordered, first priority. An overlapping case with a different
identity is an error, not a reason to try a more favorable cached measurement.
Output totals sum per-case medians; cached time is historical, not a controlled
speedup. Only a new output directory may be created.
"""
import argparse
import hashlib
import html
import json
import math
import ntpath
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import statistics


TIMING_NOTE = (
    "Cached time is historical, NOT a controlled speedup. Encoder-call time includes "
    "the requested search and excludes startup, input decode and output I/O. "
    "Totals sum per-case medians; throughput is total pixels / total call seconds, "
    "not the mean of individual rates. Matching effort/threads does not imply "
    "identical encoder policies or measurement conditions."
)
IDENTITY_FIELDS = ("original_sha256", "payload_sha256", "width", "height", "depth", "color_type")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, *, integer=False, positive=False):
    require(type(value) in ((int,) if integer else (int, float)), f"{label}: expected a number")
    require(math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"{label}: expected a finite {'positive' if positive else 'nonnegative'} value")
    return value


def digest(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdefABCDEF" for c in value), f"{label}: invalid SHA256")
    return value.lower()


def case_path(value):
    require(isinstance(value, str) and bool(value), "case path must be an absolute path")
    # Archived Windows reports can also be inspected on a POSIX host, without stat calls.
    if PureWindowsPath(value).is_absolute():
        return "windows:" + ntpath.normcase(ntpath.normpath(value))
    require(PurePosixPath(value).is_absolute(), f"case path must be absolute: {value!r}")
    return "posix:" + posixpath.normpath(value)


def records(data, key, context):
    value = data.get(key)
    require(isinstance(value, list) and all(isinstance(row, dict) for row in value),
            f"{context}: {key} must be an array of objects")
    return value


def metrics(size, seconds, pixels):
    return dict(bytes=size, call_seconds=seconds, bpp=8 * size / pixels,
                call_mpix_per_second=pixels / 1e6 / seconds)


def case_metrics(case, rows, summary, repeats, warmups, context):
    require(len(rows) == repeats + warmups, f"{context}: incomplete or duplicate rows")
    seen = set()
    measured = []
    for row in rows:
        require(row.get("status") == "passed", f"{context}: row is not passed")
        repeat = row.get("repeat")
        require(type(repeat) is int and -warmups <= repeat < repeats and repeat not in seen,
                f"{context}: invalid or duplicate repeat")
        require(type(row.get("warmup")) is bool and row["warmup"] == (repeat < 0),
                f"{context}: invalid warmup flag")
        seen.add(repeat)
        size = number(row.get("bytes"), f"{context}: row bytes", integer=True, positive=True)
        number(row.get("seconds"), f"{context}: row seconds", positive=True)
        bpp = number(row.get("bpp"), f"{context}: row bpp", positive=True)
        require(math.isclose(bpp, 8 * size / case["pixels"], rel_tol=1e-9, abs_tol=1e-12),
                f"{context}: row bpp disagrees with bytes/dimensions")
        if not row["warmup"]:
            measured.append(row)
    trials = number(summary.get("trials"), f"{context}: summary trials", integer=True, positive=True)
    require(trials == repeats, f"{context}: summary trials disagree with measured rows")
    result = metrics(statistics.median(row["bytes"] for row in measured),
                     statistics.median(row["seconds"] for row in measured), case["pixels"])
    for field, key in (("median_bytes", "bytes"), ("median_call_seconds", "call_seconds"),
                       ("median_bpp", "bpp")):
        value = number(summary.get(field), f"{context}: {field}", positive=True)
        matches = (value == result[key] if key == "bytes" else
                   math.isclose(value, result[key], rel_tol=1e-9, abs_tol=1e-12))
        require(matches,
                f"{context}: {field} disagrees with passed measured rows")
    return dict(result, trials=trials)


def load_report(path, variant):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    context = f"{path} [variant {variant!r}]"
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{context}: invalid JSON: {error}") from error
    require(isinstance(data, dict), f"{context}: report must be an object")
    require(bool(variant), f"{context}: variant cannot be empty")
    require(data.get("status") == "passed", f"{context}: report is not passed")
    require(data.get("primary_metric") == "call_mpix_per_second",
            f"{context}: expected encoder-call timing, not CLI wall timing")
    repeats = number(data.get("repeats"), f"{context}: repeats", integer=True, positive=True)
    warmups = number(data.get("warmups"), f"{context}: warmups", integer=True)
    origin = dict(report=str(path), report_sha256=hashlib.sha256(raw).hexdigest(), variant=variant,
                  encoder=data.get("encoder"),
                  encoder_sha256=digest(data.get("encoder_sha256"), f"{context}: encoder_sha256"),
                  effort=number(data.get("effort"), f"{context}: effort", integer=True, positive=True),
                  threads=number(data.get("threads"), f"{context}: threads", integer=True, positive=True),
                  repeats=repeats, warmups=warmups)
    for key in ("timing", "environment", "djxl", "djxl_sha256", "decoder_environment"):
        if key in data:
            origin[key] = data[key]
    cases, indices = {}, {}
    for source in records(data, "cases", context):
        case = dict(source)
        index = number(case.get("index"), f"{context}: case index", integer=True)
        key = case_path(case.get("path"))
        label = f"{context}: {case['path']}"
        require(key not in cases, f"{label}: duplicate case path")
        require(index not in indices, f"{label}: duplicate case index")
        for field in ("original_sha256", "payload_sha256"):
            case[field] = digest(case.get(field), f"{label}: {field}")
        for field in ("width", "height", "depth", "pixels"):
            number(case.get(field), f"{label}: {field}", integer=True, positive=True)
        number(case.get("color_type"), f"{label}: color_type", integer=True)
        require(case["pixels"] == case["width"] * case["height"],
                f"{label}: pixels disagree with dimensions")
        cases[key] = dict(case=case, rows=[], summary=None)
        indices[index] = cases[key]
    require(bool(cases), f"{context}: no cases")
    for row in records(data, "rows", context):
        if row.get("variant") == variant:
            index = row.get("case_index")
            require(type(index) is int and index in indices, f"{context}: row has unknown case_index")
            indices[index]["rows"].append(row)
    for summary in records(data, "summaries", context):
        if summary.get("variant") == variant:
            index = summary.get("case_index")
            require(type(index) is int and index in indices,
                    f"{context}: summary has unknown case_index")
            require(indices[index]["summary"] is None, f"{context}: duplicate summary for case {index}")
            indices[index]["summary"] = summary
    for item in cases.values():
        label = f"{context}: {item['case']['path']}"
        require(item["summary"] is not None, f"{label}: missing summary for selected variant")
        item["metrics"] = case_metrics(item["case"], item["rows"], item["summary"],
                                       repeats, warmups, label)
    return dict(origin=origin, cases=cases)


def deltas(current, baseline):
    result = {}
    for key in ("bytes", "call_seconds", "bpp"):
        result[key] = current[key] - baseline[key]
        result[key + "_pct"] = 100 * result[key] / baseline[key]
    return result


def totals(cases, side):
    pixels = sum(case["pixels"] for case in cases)
    size = sum(case[side]["bytes"] for case in cases)
    seconds = math.fsum(case[side]["call_seconds"] for case in cases)
    return dict(metrics(size, seconds, pixels), case_count=len(cases), total_pixels=pixels)


def summarize(current_path, variant, baseline_specs):
    require(bool(baseline_specs), "At least one baseline report/variant is required")
    current = load_report(current_path, variant)
    baselines = [load_report(path, name) for path, name in baseline_specs]
    for priority, baseline in enumerate(baselines, 1):
        baseline["origin"]["priority"] = priority
        for field in ("effort", "threads"):
            require(baseline["origin"][field] == current["origin"][field],
                    f"{baseline['origin']['report']}: {field} mismatch with current report "
                    f"({baseline['origin'][field]} != {current['origin'][field]})")
    cases = []
    for key, item in current["cases"].items():
        source = item["case"]
        selected = None
        for baseline in baselines:
            candidate = baseline["cases"].get(key)
            if candidate is None:
                continue
            differences = [field for field in IDENTITY_FIELDS
                           if source[field] != candidate["case"][field]]
            require(not differences, f"{baseline['origin']['report']}: baseline identity mismatch for "
                    f"{source['path']}: {', '.join(differences)}")
            # Validate all overlapping identities, but never replace the first matching result.
            if selected is None:
                selected = (baseline, candidate)
        require(selected is not None, f"No baseline for current case: {source['path']}")
        baseline, candidate = selected
        entry = {field: source[field] for field in ("path", "pixels", *IDENTITY_FIELDS)}
        entry.update(case_index=source["index"], current=item["metrics"],
                     baseline=dict(candidate["metrics"], case_index=candidate["case"]["index"],
                                   origin=baseline["origin"]),
                     delta=deltas(item["metrics"], candidate["metrics"]))
        cases.append(entry)
    current_totals, baseline_totals = totals(cases, "current"), totals(cases, "baseline")
    rankings = {}
    for name, metric in (("volume_regressions", "bytes"), ("worst_slowdowns", "call_seconds")):
        ranked = sorted((case for case in cases if case["delta"][metric] > 0),
                        key=lambda case: (-case["delta"][metric], case["path"]))
        rankings[name] = [case["case_index"] for case in ranked]
    return dict(status="passed", timing=TIMING_NOTE, timing_comparable=False,
                baseline_selection="First listed report containing the case; all overlapping identities must match.",
                current=current["origin"], baselines=[baseline["origin"] for baseline in baselines],
                totals=dict(current=current_totals, baseline=baseline_totals,
                            delta=deltas(current_totals, baseline_totals)),
                cases=cases, rankings=rankings)


def cell(value):
    return (html.escape(str(value), quote=False).replace("|", "&#124;")
            .replace("`", "&#96;").replace("\r", " ").replace("\n", " "))


def byte_count(value, *, signed=False):
    text = str(int(value)) if value == int(value) else str(value)
    return "+" + text if signed and value >= 0 else text


def markdown(report):
    lines = ["# Encoder Sample Comparison", "", report["timing"], "",
             "Positive deltas mean larger output or longer call time. All cases must have a verified pair.",
             "", "## Totals", "",
             "| Measurement | Cases | Pixels | Sum bytes | Sum call s | Weighted MPix/s | bpp |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for side, label in (("baseline", "Cached historical"), ("current", "Current")):
        item = report["totals"][side]
        lines.append(f"| {label} | {item['case_count']} | {item['total_pixels']} | {byte_count(item['bytes'])} | "
                     f"{item['call_seconds']:.6f} | {item['call_mpix_per_second']:.6f} | {item['bpp']:.6f} |")
    change = report["totals"]["delta"]
    lines += ["", f"Bytes delta: {byte_count(change['bytes'], signed=True)} ({change['bytes_pct']:+.3f}%). "
              f"Historical call-time delta: {change['call_seconds']:+.6f} s "
              f"({change['call_seconds_pct']:+.3f}%). bpp delta: {change['bpp']:+.6f}.",
              "", "## Origins", "", report["baseline_selection"], ""]
    for label, origin in [("Current", report["current"])] + [
            (f"Baseline {item['priority']}", item) for item in report["baselines"]]:
        lines += [f"- {label}: {cell(origin['report'])}; variant `{cell(origin['variant'])}`; "
                  f"effort {origin['effort']}, threads {origin['threads']}.",
                  f"  Report SHA256 `{origin['report_sha256']}`; encoder SHA256 `{origin['encoder_sha256']}`."]
    lookup = {case["case_index"]: case for case in report["cases"]}
    groups = [("Per Case", report["cases"]),
              ("Volume Regressions (Top 10 by Added Bytes)",
               [lookup[index] for index in report["rankings"]["volume_regressions"][:10]]),
              ("Worst Slowdowns (Top 10 by Historical Added Seconds)",
               [lookup[index] for index in report["rankings"]["worst_slowdowns"][:10]])]
    for title, cases in groups:
        lines += ["", "## " + title, ""]
        if not cases:
            lines.append("None.")
            continue
        lines += ["| Original path | Baseline priority | Old bytes | New bytes | Bytes delta | Bytes % | "
                  "Old call s | New call s | Time delta s | Time % | Old bpp | New bpp | bpp delta |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for case in cases:
            old, new, change = case["baseline"], case["current"], case["delta"]
            lines.append(f"| {cell(case['path'])} | {old['origin']['priority']} | "
                         f"{byte_count(old['bytes'])} | {byte_count(new['bytes'])} | "
                         f"{byte_count(change['bytes'], signed=True)} | "
                         f"{change['bytes_pct']:+.3f} | {old['call_seconds']:.6f} | "
                         f"{new['call_seconds']:.6f} | {change['call_seconds']:+.6f} | "
                         f"{change['call_seconds_pct']:+.3f} | {old['bpp']:.6f} | "
                         f"{new['bpp']:.6f} | {change['bpp']:+.6f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--current", required=True, type=Path)
    cli.add_argument("--variant", required=True)
    cli.add_argument("--baseline", required=True, action="append", nargs=2,
                     metavar=("REPORT", "VARIANT"), help="Repeat in fixed first-priority order")
    cli.add_argument("--out", required=True, type=Path, help="New directory for summary.json and REPORT.md")
    args = cli.parse_args(argv)
    try:
        require(not args.out.exists(), f"Output directory already exists: {args.out}")
        report = summarize(args.current, args.variant, args.baseline)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        rendered = markdown(report)
        args.out.mkdir(exist_ok=False)
        (args.out / "summary.json").write_text(encoded, encoding="utf-8")
        (args.out / "REPORT.md").write_text(rendered, encoding="utf-8")
    except (OSError, ValueError, OverflowError) as error:
        cli.error(str(error))
    print(f"Compared {len(report['cases'])} cases; cached time is historical, not a controlled speedup. "
          f"Reports: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
