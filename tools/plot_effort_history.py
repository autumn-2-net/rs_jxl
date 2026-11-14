"""Plot refreshed efforts alongside unchanged efforts from a verified older run.

Both inputs remain immutable measurement evidence. The composed object is only
a display view, not a new benchmark report. This tool never runs an encoder.
"""

import argparse
import copy
import csv
from io import StringIO
from pathlib import Path

import plot_effort_refresh as plot


def compose(current, cached):
    fresh = plot.validate_report(current)
    old = plot.validate_report(cached)
    a, b = current["identity"], cached["identity"]
    plot.bench.require(a["inputs"] == b["inputs"]
                       and a["previous_fingerprint"] == b["previous_fingerprint"]
                       and a["binaries"]["decoder"]["sha256"] == b["binaries"]["decoder"]["sha256"],
                       "Cannot combine different cohorts, baselines or decoders")
    for field in ("threads", "repeats", "serial", "patches", "search"):
        plot.bench.require(a["config"][field] == b["config"][field],
                           f"Cannot combine different {field} settings")
    fresh_efforts = set(a["config"]["efforts"])
    tail = sorted(set(b["config"]["efforts"]) - fresh_efforts)
    series = copy.deepcopy(fresh)
    for name in series:
        old_points = {p["effort"]: p for p in old[name]["points"]}
        if name != "new":
            for point in fresh[name]["points"]:
                if point["effort"] in old_points:
                    plot.bench.require(point == old_points[point["effort"]],
                                       "Cached baseline measurements disagree")
        series[name]["points"].extend(copy.deepcopy(old_points[e]) for e in tail)
        series[name]["points"].sort(key=lambda p: p["effort"])
    all_efforts = fresh_efforts | set(tail)
    previous_points = {p["effort"]: p for p in old["new"]["points"]}
    plot.bench.require(all_efforts <= previous_points.keys(), "Previous run lacks an effort")
    # Compare against the immediately preceding Rust run, not that run's own
    # older cached baseline. High-effort points intentionally coincide.
    series["previous"] = dict(label="rs_jxl before low-effort optimization (cached)",
                              points=[copy.deepcopy(previous_points[e]) for e in sorted(all_efforts)])
    view = dict(schema="effort-history-view-v1", fresh_efforts=sorted(fresh_efforts),
                cached_efforts=tail, identity=dict(inputs=a["inputs"],
                    config=dict(a["config"], efforts=sorted(fresh_efforts | set(tail)))))
    return view, series


def effort_label(values):
    if values and values == list(range(values[0], values[-1] + 1)):
        return f"e{values[0]}" if len(values) == 1 else f"e{values[0]}-e{values[-1]}"
    return ", ".join(f"e{e}" for e in values) or "none"


def draw(view, series, metric):
    fig = plot.draw_plot(view, series, metric)
    ax = fig.axes[0]
    points = series["new"]["points"]
    cached = [p for p in points if p["effort"] in view["cached_efforts"]]
    ax.lines[0].set_markevery([i for i, p in enumerate(points)
                              if p["effort"] in view["fresh_efforts"]])
    key, divisor, _ = plot.METRICS[metric]
    ax.plot([p["native_seconds"] for p in cached], [p[key] / divisor for p in cached],
            color=plot.STYLES[0][2], marker="o", markerfacecolor="white", linestyle="none", ms=6)
    tail = effort_label(view["cached_efforts"])
    ax.get_legend().get_texts()[0].set_text(f"Updated rs_jxl\n({tail} cached)")
    fig.suptitle("rs_jxl effort comparison / low-effort refresh",
                 x=.09, ha="left", y=.96, fontsize=17)
    fig.texts[1].set_text(f"{len(view['identity']['inputs'])} matched images | "
                         f"{view['identity']['config']['threads']} threads | "
                         f"remeasured {effort_label(view['fresh_efforts'])} | reused {tail}")
    fig.texts[-1].set_text(
        ("Arithmetic mean per-image bpp: each image has equal weight." if metric == "mean-bpp"
         else "Total compressed size in MiB.") + " No error bars.\n"
        f"Hollow green points: cached {tail}; only low efforts were remeasured.\n"
        "Timing excludes I/O and startup. Historical measurements are non-simultaneous.")
    return fig


def csv_data(view, series, metric, current_path, cached_path):
    stream = StringIO()
    plot.write_csv(stream, view, series, metric)
    reader = csv.DictReader(StringIO(stream.getvalue()))
    rows = list(reader)
    for row in rows:
        fresh = int(row["effort"]) in view["fresh_efforts"]
        row["cached"] = str(row["series"] != "new" or not fresh)
        row["report_source"] = str(current_path if fresh and row["series"] != "previous" else cached_path)
    return [*reader.fieldnames, "report_source"], rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    current = plot.bench.loads(args.current.read_bytes())
    cached = plot.bench.loads(args.cached.read_bytes())
    view, series = compose(current, cached)
    jobs = []
    for metric, name in (("mean-bpp", "bpp-time-e1-e9.png"), ("total-mib", "size-time-e1-e9.png")):
        out = args.outdir / name
        paths = plot.output_paths(args.current, out)
        plot.output_paths(args.cached, out)
        jobs.append((metric, paths))
    args.outdir.mkdir(parents=True, exist_ok=True)
    for metric, paths in jobs:
        fig = draw(view, series, metric)
        try:
            with paths[0].open("xb") as stream:
                fig.savefig(stream, format="png", dpi=200)
            with paths[1].open("xb") as stream:
                fig.savefig(stream, format="svg")
            fields, rows = csv_data(view, series, metric, args.current.resolve(), args.cached.resolve())
            with paths[2].open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
        print(paths[0])


if __name__ == "__main__":
    main()
