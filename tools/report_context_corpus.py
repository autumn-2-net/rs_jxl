"""Render a completed context-corpus summary as Markdown; no codecs or source IO.

Use python -B while a frozen benchmark is running to avoid bytecode-cache writes.
Only a new output file is written; existing destinations are never overwritten.
"""

import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys

import benchmark_context_corpus as bench
import plot_context_corpus as plot


BASELINE = "rust-e9"


def analyze(report, resamples=10000, seed=20261112):
    images, totals = plot.validate_report(report)
    bench.require(BASELINE in images, "Report requires a verified Rust e9 baseline")
    baseline = images[BASELINE]
    experiments = [p for p in report["variants"] if p["family"] == "experiment"]
    bootstrap_images = dict(images)
    for point in experiments:
        name = point["id"]
        bootstrap_images["delta:" + name] = [dict(case_index=v["case_index"], bpp=v["bpp"] - b["bpp"])
                                             for v, b in zip(images[name], baseline)]
    # One shared resampling pass covers means and paired within-image differences.
    intervals = plot.bootstrap_ci(bootstrap_images, resamples, seed)
    comparisons, size_regressions, time_regressions = {}, [], []
    reference = totals[BASELINE]
    for point in experiments:
        name = point["id"]
        values, total = images[name], totals[name]
        wins = sum(v["bytes"] < b["bytes"] for v, b in zip(values, baseline))
        ties = sum(v["bytes"] == b["bytes"] for v, b in zip(values, baseline))
        mean_delta = statistics.mean(v["bpp"] for v in bootstrap_images["delta:" + name])
        byte_delta = total["summed_bytes"] - reference["summed_bytes"]
        comparisons[name] = dict(mean_bpp_delta=mean_delta,
            mean_bpp_delta_percent=100 * mean_delta / reference["mean_bpp"],
            bytes_delta=byte_delta, bytes_delta_percent=100 * byte_delta / reference["summed_bytes"],
            time_ratio=total["native_seconds"] / reference["native_seconds"],
            wins=wins, ties=ties, losses=len(values) - wins - ties,
            **intervals["delta:" + name])
        for value, base in zip(values, baseline):
            common = dict(variant_id=name, case_index=value["case_index"])
            if value["bytes"] > base["bytes"]:
                size_regressions.append(dict(common, ratio=value["bytes"] / base["bytes"],
                                             delta=value["bytes"] - base["bytes"]))
            if value["seconds"] > base["seconds"]:
                time_regressions.append(dict(common, ratio=value["seconds"] / base["seconds"],
                                             delta=value["seconds"] - base["seconds"]))
    def top_five(values):
        return sorted(values, key=lambda v: (-v["ratio"], -v["delta"], v["variant_id"], v["case_index"]))[:5]
    return dict(totals=totals, intervals={name: intervals[name] for name in images},
                comparisons=comparisons, size_regressions=top_five(size_regressions),
                time_regressions=top_five(time_regressions))


def cell(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(
        "|", "&#124;").replace("`", "&#96;").replace("\r", " ").replace("\n", " ")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report, analysis, report_path, resamples=10000, seed=20261112):
    identity = report["identity"]
    config, cases = identity["config"], identity["inputs"]
    reference = analysis["totals"][BASELINE]
    strata = Counter(str(c.get("stratum") or "unspecified") for c in cases)
    lines = ["# Context Corpus Results", "",
        f"Complete: {len(cases)} images, {len(report['variants'])} points, "
        f"{report['verified_trials']}/{report['expected_trials']} verified trials. "
        f"Threads: {config['threads']}; repeats: {config['repeats']}; patches AUTO.", "",
        f"Corpus: {reference['summed_pixels']:,} pixels ({reference['summed_mpix']:.6f} MPix). "
        "Recorded strata: " + ", ".join(f"{cell(k)}={v}" for k, v in sorted(strata.items())) + ".", "",
        "## All Points", "",
        "Mean bpp gives each image equal weight. Bytes and native time sum per-image medians. "
        "MPix/s is total pixels / total native seconds / 1e6.", ""]
    lines.append(table(["Point", "Mean bpp", "95% CI", "Weighted bpp", "Bytes", "Native s", "MPix/s"], [
        [p["id"], f"{analysis['totals'][p['id']]['mean_bpp']:.6f}",
         f"[{analysis['intervals'][p['id']]['ci_low']:.6f}, {analysis['intervals'][p['id']]['ci_high']:.6f}]",
         f"{analysis['totals'][p['id']]['pixel_weighted_bpp']:.6f}",
         f"{analysis['totals'][p['id']]['summed_bytes']:,.1f}".removesuffix(".0"),
         f"{analysis['totals'][p['id']]['native_seconds']:.6f}",
         f"{analysis['totals'][p['id']]['mpix_per_second']:.6f}"] for p in report["variants"]]))
    lines += ["", "## Experiments vs Rust e9", "",
        "Deltas are experiment minus baseline; negative size deltas are improvements. "
        "Time ratio is experiment / baseline (less than 1 is faster). "
        "Wins/ties/losses compare per-image median bytes exactly, not timing.", ""]
    lines.append(table(["Experiment", "Mean bpp delta (%)", "Paired 95% delta CI", "Bytes delta (%)",
                        "Time ratio", "Wins/ties/losses"], [
        [p["mode"], f"{v['mean_bpp_delta']:+.6f} ({v['mean_bpp_delta_percent']:+.3f}%)",
         f"[{v['ci_low']:+.6f}, {v['ci_high']:+.6f}]",
         f"{v['bytes_delta']:+,.1f}".removesuffix(".0") + f" ({v['bytes_delta_percent']:+.3f}%)",
         f"{v['time_ratio']:.4f}x", f"{v['wins']}/{v['ties']}/{v['losses']}"]
        for p in report["variants"] if p["family"] == "experiment"
        for v in [analysis["comparisons"][p["id"]]]]))
    for kind, title, unit in (("size", "Size", "bytes"), ("time", "Time", "native s")):
        lines += ["", f"## Top Five {title} Regressions", "",
                  "Experiment/image pairs versus Rust e9, ranked by relative increase; "
                  "absolute increase breaks ties. Only regressions are listed.", ""]
        rows = analysis[kind + "_regressions"]
        if rows:
            lines.append(table(["Experiment", "Image (1-based)", "Source", "Increase", f"Extra {unit}"], [
                [v["variant_id"], v["case_index"] + 1, cases[v["case_index"]]["path"],
                 f"{100 * (v['ratio'] - 1):+.3f}%", f"{v['delta']:+,.6f}" if kind == "time"
                 else f"{v['delta']:+,.1f}".removesuffix(".0")] for v in rows]))
        else:
            lines.append("None observed.")
    lines += ["", "## Interpretation", "",
        f"- Bootstrap: {resamples:,} resamples, seed {seed}; 95% percentile intervals. "
        "All points and deltas use the same resampled image indices. Delta CIs bootstrap paired "
        "per-image differences, not differences between separate CI endpoints.",
        "- This manifest-selected/stratified cohort is not necessarily an IID random sample. "
        "Image-bootstrap intervals are descriptive, not population guarantees; no stratified "
        "resampling or stratum weighting is applied.",
        "- Intervals describe image-sample variation, not quality loss, codec stochastic error, "
        "or runtime uncertainty. Multiple experiment comparisons are not multiplicity-adjusted.",
        "- Native timing excludes process startup, PNG decoding and file IO; process wall seconds "
        "are separate diagnostics. Trials are serial with rotated variant order. "
        "Runtime ratios are observational; no timing CI is estimated.",
        "- cjxl native seconds are reconstructed from the final three-decimal MP/s rate. "
        "Quantization bounds remain in summary.json and the plotter CSV, separate from bootstrap CIs. "
        "Printed rounded kB are never used for size.",
        "- Every output was checked by official djxl against original samples, including hidden RGB. "
        "Packed PNGs remain native encoder inputs; only the oracle expands them. 16-bit samples "
        "are preserved. No resizing or silent case filtering is allowed.",
        "- This is a whole-pipeline patches AUTO comparison, not the earlier patches-OFF MA isolation. "
        "Modes are direct native encodes without an outer portfolio, e10, or joint RCT search."]
    if config["repeats"] == 1:
        lines.append("- One repeat per image/point does not quantify runtime variability.")
    lines += ["", "## Audit Paths", "",
              f"- Summary: {cell(Path(report_path).resolve())}",
              f"- Run fingerprint: {cell(report['fingerprint'])}"]
    for name, item in [("Manifest", identity.get("manifest")), *identity.get("binaries", {}).items()]:
        if item:
            lines.append(f"- {cell(name)}: {cell(item['path'])}; SHA256 {cell(item.get('sha256', 'not recorded'))}")
    return "\n".join(lines) + "\n"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Complete summary.json")
    parser.add_argument("--out", type=Path, required=True, help="New Markdown destination; never overwritten")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261112)
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".md" or args.bootstrap_resamples < 2 or args.bootstrap_seed < 0:
        parser.error("Require .md output, at least two resamples and a nonnegative seed")
    return args


def main(argv=None):
    try:
        args = arguments(argv)
        bench.require(not args.out.exists() and args.out.resolve() != args.report.resolve(),
                      "Output already exists or aliases input; use a new Markdown path")
        report = bench.loads(args.report.read_text(encoding="utf-8-sig"))
        analysis = analyze(report, args.bootstrap_resamples, args.bootstrap_seed)
        markdown = render_markdown(report, analysis, args.report, args.bootstrap_resamples, args.bootstrap_seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(markdown)
        print(args.out)
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, OverflowError) as error:
        print(f"Context report failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
