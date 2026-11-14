"""Plot validated ordinary-effort refresh evidence; never run encoders.

Exports PNG, SVG and per-effort CSV point estimates without error bars.
"""

import argparse
from contextlib import ExitStack
import csv
import importlib
from pathlib import Path
import sys

import benchmark_context_corpus as bench
from plot_context_corpus import METRICS, place_labels


STYLES = (
    ("new", "New rs_jxl", "#087e68", "o", "-"),
    ("previous", "Previous rs_jxl (cached)", "#3766ad", "s", "--"),
    ("cjxl", "Official cjxl 0.12 (cached)", "#bd493b", "^", "-."),
)
POINT_FIELDS = ("effort", "n", "mean_bpp", "summed_bytes", "pixel_weighted_bpp",
                "summed_pixels", "native_seconds", "mpix_per_second",
                "native_seconds_lower", "native_seconds_upper")
TIMING_NOTE = ("Historical baselines are cached; one-run timings are non-simultaneous. "
               "Native time excludes I/O and process startup.")


def validate_report(report):
    # The runner imports plotting only after the complete verified run.
    series = importlib.import_module("benchmark_effort_refresh").validate_report(report)
    bench.require(report.get("schema") == "effort-refresh-v1", "Not an effort refresh report")
    bench.require(isinstance(series, dict) and set(series) == {s[0] for s in STYLES}
                  and report.get("series") == series, "Report series mismatch")
    return series


def draw_plot(report, series, metric="mean-bpp"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key, divisor, ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.subplots_adjust(left=.09, right=.72, bottom=.20, top=.85)
    ax.set_xscale("log")
    ax.set_xlabel("Total native encode time (seconds; log scale)", labelpad=12)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(True, which="major", color="#dedede", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=.16, y=.18)
    entries = []
    for name, label, color, marker, linestyle in STYLES:
        points = sorted(series[name]["points"], key=lambda p: p["effort"])
        ax.plot([p["native_seconds"] for p in points], [p[key] / divisor for p in points],
                color=color, marker=marker, linestyle=linestyle, lw=1.2, ms=6, label=label)
        entries.extend((p["native_seconds"], p[key] / divisor, f"e{p['effort']}", color)
                       for p in points)
    ax.legend(loc="upper left", bbox_to_anchor=(1.025, 1), borderaxespad=0,
              frameon=False, fontsize=10, labelspacing=1.2)
    config = report["identity"]["config"]
    fig.suptitle("Ordinary rs_jxl effort refresh / cached baselines", x=.09,
                 ha="left", y=.96, fontsize=17)
    fig.text(.09, .902, f"{len(report['identity']['inputs'])} matched images | "
             f"{config['threads']} threads | new run: {config['repeats']} repeat(s) | "
             + ", ".join(f"e{e}" for e in config["efforts"]),
             fontsize=11)
    statistic = ("Arithmetic mean per-image bpp: each image has equal weight."
                 if metric == "mean-bpp" else "Total compressed bytes / 1,048,576 (MiB).")
    fig.text(.09, .045, statistic + " Point estimates only; no error bars.\n"
             "Historical baselines are cached; one-run timings are non-simultaneous.\n"
             "Native time excludes I/O and process startup. Timing bounds in CSV are not statistical CIs.",
             fontsize=10, color="#444444", linespacing=1.6)
    try:
        place_labels(fig, ax, entries)
    except Exception:
        plt.close(fig)
        raise
    return fig


def write_csv(stream, report, series, metric="mean-bpp"):
    fields = ["series", "label", "cached", *POINT_FIELDS, "plot_metric", "plot_x", "plot_y",
              "threads", "new_run_repeats", "timing_note"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    key, divisor, _ = METRICS[metric]
    config = report["identity"]["config"]
    for name, _, _, _, _ in STYLES:
        for point in sorted(series[name]["points"], key=lambda p: p["effort"]):
            writer.writerow(dict({field: point[field] for field in POINT_FIELDS},
                series=name, label=series[name]["label"], cached=name != "new",
                plot_metric=metric, plot_x=point["native_seconds"], plot_y=point[key] / divisor,
                threads=config["threads"], new_run_repeats=config["repeats"],
                timing_note=TIMING_NOTE))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Validated effort-refresh-v1 JSON report")
    parser.add_argument("--out", type=Path, required=True, help="PNG destination; also writes SVG and CSV")
    parser.add_argument("--metric", choices=tuple(METRICS), default="mean-bpp")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".png":
        parser.error("--out must have a .png extension")
    return args


def output_paths(report_path, out, overwrite=False):
    paths = [out, out.with_suffix(".svg"), out.with_suffix(".csv")]
    resolved = [p.resolve() for p in paths]
    bench.require(len(set(resolved)) == 3 and report_path.resolve() not in resolved,
                  "Output paths alias each other or the report")
    for i, path in enumerate(paths):
        if path.exists():
            bench.require(overwrite and path.is_file(), "Output exists; use --overwrite")
            bench.require(not path.samefile(report_path)
                          and not any(p.exists() and path.samefile(p) for p in paths[:i]),
                          "Output aliases report or another output")
    return paths


def main(argv=None):
    fig = None
    try:
        args = arguments(argv)
        report = bench.loads(args.report.read_text(encoding="utf-8-sig"))
        series = validate_report(report)
        paths = output_paths(args.report, args.out, args.overwrite)
        fig = draw_plot(report, series, args.metric)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            png = stack.enter_context(paths[0].open("wb" if args.overwrite else "xb"))
            svg = stack.enter_context(paths[1].open("wb" if args.overwrite else "xb"))
            stream = stack.enter_context(paths[2].open("w" if args.overwrite else "x",
                                                    encoding="utf-8", newline=""))
            fig.savefig(png, format="png", dpi=200)
            fig.savefig(svg, format="svg")
            write_csv(stream, report, series, args.metric)
        for path in paths:
            print(path)
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, ImportError) as error:
        print(f"Effort refresh plot failure: {error}", file=sys.stderr)
        return 1
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
