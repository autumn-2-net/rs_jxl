"""Plot mean bpp or total size against native encode time for a complete corpus.

Always exports PNG, SVG and CSV. Never encodes, decodes, or silently drops a case.
Intervals describe image-sample variation, not codec randomness or quality loss.
"""

import argparse
from contextlib import ExitStack
import csv
import math
from pathlib import Path
import sys

import benchmark_context_corpus as bench


CI_CONTRACT = ("95% percentile bootstrap CI of arithmetic mean per-image bpp; "
               "paired common image indices across all variants; not quality loss or codec stochastic error")
METRICS = {
    "mean-bpp": ("mean_bpp", 1, "Arithmetic mean of per-image bpp (lower is better)"),
    "total-mib": ("summed_bytes", 1024 ** 2, "Total compressed size (MiB; lower is better)"),
}


def validate_report(report):
    bench.require(isinstance(report, dict) and report.get("schema") == bench.SCHEMA,
                  "Not a context corpus report")
    bench.require(report.get("status") == "complete" and report.get("integrity_verified") is True
                  and report.get("errors") == [], "Cannot plot an incomplete or integrity-failed report")
    identity, digest = report["identity"], report["fingerprint"]
    bench.require(bench.fingerprint(identity) == digest, "Report identity fingerprint mismatch")
    config = identity["config"]
    bench.require(identity.get("schema") == bench.SCHEMA and config.get("serial") is True
                  and config.get("patches") == "auto"
                  and config.get("backend") == "direct native; no outer portfolio", "Invalid benchmark policy")
    bench.require(isinstance(config["efforts"], list) and config["efforts"]
                  and all(type(e) is int and 1 <= e <= 9 for e in config["efforts"])
                  and len(set(config["efforts"])) == len(config["efforts"]), "Invalid effort matrix")
    bench.require(isinstance(config["experiment_modes"], list) and config["experiment_modes"]
                  and set(config["experiment_modes"]) <= set(bench.EXPERIMENT_MODES)
                  and len(set(config["experiment_modes"])) == len(config["experiment_modes"]),
                  "Invalid experiment matrix")
    bench.require(type(config["repeats"]) is int and 1 <= config["repeats"] <= 100
                  and type(config["threads"]) is int and 1 <= config["threads"] <= 256,
                  "Invalid repeat/thread count")
    cases = identity["inputs"]
    bench.require(isinstance(cases, list) and 0 < len(cases) <= config["max_cases"], "Invalid case matrix")
    bench.require(len({c["path"] for c in cases}) == len(cases), "Duplicate input identities")
    for case in cases:
        bench.require(all(type(case.get(k)) is int and case[k] > 0 for k in ("width", "height")),
                      "Invalid case dimensions")
    points = bench.variants(config)
    bench.require(report["variants"] == points, "Report variant matrix mismatch")
    rows = report["trials"]
    expected = len(cases) * len(points) * config["repeats"]
    bench.require(isinstance(rows, list) and len(rows) == expected
                  and report.get("expected_trials") == expected and report.get("verified_trials") == expected
                  and report.get("failed_trials") == 0 and report.get("missing_trials") == 0,
                  "Report does not contain the full verified matrix")
    seen = set()
    for row in rows:
        bench.validate_row(row, identity, digest)
        bench.require(row["verified"] is True and bench.row_key(row) not in seen,
                      "Unverified or duplicate trial")
        seen.add(bench.row_key(row))
    bench.require(report.get("case_statuses") == [dict(case_index=i, status="verified") for i in range(len(cases))],
                  "Case statuses do not cover the complete matrix")
    images = bench.per_image(identity, rows)
    totals = bench.aggregate(images)
    bench.require(report.get("aggregate") == totals, "Aggregate does not match verified trials")
    return images, totals


def bootstrap_ci(images, resamples=10000, seed=20261112):
    """A shared index matrix retains pairing for later between-mode comparisons."""
    import numpy as np
    bench.require(type(resamples) is int and resamples >= 2, "Use at least two bootstrap resamples")
    bench.require(type(seed) is int and seed >= 0, "Bootstrap seed must be nonnegative")
    names = list(images)
    bench.require(bool(names), "Empty bootstrap matrix")
    n = len(images[names[0]])
    bench.require(n > 0 and all(len(images[name]) == n for name in names), "Unpaired bootstrap cases")
    for name in names:
        bench.require([v["case_index"] for v in images[name]] == list(range(n)),
                      "Bootstrap cases must have the same ordered identities")
    values = np.asarray([[v["bpp"] for v in images[name]] for name in names], dtype=float)
    bench.require(bool(np.isfinite(values).all()), "Nonfinite bootstrap values")
    rng = np.random.default_rng(seed)
    sampled_means = np.empty((len(names), resamples), dtype=float)
    for start in range(0, resamples, 128):
        count = min(128, resamples - start)
        indices = rng.integers(0, n, size=(count, n))
        sampled_means[:, start:start + count] = values[:, indices].mean(axis=2)
    limits = np.quantile(sampled_means, [.025, .975], axis=1, method="linear")
    return {name: dict(ci_low=float(limits[0, i]), ci_high=float(limits[1, i]))
            for i, name in enumerate(names)}


def pareto_frontier(totals, metric="mean-bpp"):
    key = METRICS[metric][0]
    return [name for name, value in sorted(totals.items(), key=lambda item: item[1]["native_seconds"])
            if not any(other["native_seconds"] <= value["native_seconds"]
                       and other[key] <= value[key]
                       and (other["native_seconds"] < value["native_seconds"]
                            or other[key] < value[key])
                       for other in totals.values())]


def place_labels(fig, ax, entries):
    """Place short IDs in display space; reserve all markers and existing labels."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundary = ax.get_window_extent(renderer).padded(-5)
    obstacles = []
    for x, y, _, _ in entries:
        px, py = ax.transData.transform((x, y))
        obstacles.append(Bbox.from_extents(px - 6, py - 6, px + 6, py + 6))
    labels = []
    offsets = [(10, 9), (10, -12), (-12, 10), (-12, -12)]
    for radius in range(24, 241, 16):
        offsets.extend((radius * math.cos(angle * math.pi / 8), radius * math.sin(angle * math.pi / 8))
                       for angle in range(16))
    for x, y, label, color in entries:
        annotation = ax.annotate(label, (x, y), xytext=(0, 0), textcoords="offset points",
                                 fontsize=8, color=color, ha="center", va="center",
                                 bbox=dict(facecolor="white", edgecolor="none", pad=.4, alpha=.9))
        placed = False
        for dx, dy in offsets:
            annotation.set_position((dx, dy))
            bounds = annotation.get_window_extent(renderer).padded(2)
            if (boundary.contains(bounds.x0, bounds.y0) and boundary.contains(bounds.x1, bounds.y1)
                    and not any(bounds.overlaps(obstacle) for obstacle in obstacles)):
                obstacles.append(bounds)
                if math.hypot(dx, dy) > 20:
                    ax.annotate("", (x, y), xytext=(dx, dy), textcoords="offset points",
                                arrowprops=dict(arrowstyle="-", color=color, lw=.5, alpha=.5))
                placed = True
                break
        bench.require(placed, "Could not place nonoverlapping labels; enlarge the chart")
        labels.append(annotation)
    return labels


def draw_plot(report, totals, intervals, pareto=False, x_scale="log", *,
              metric="mean-bpp", error_bars=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    key, divisor, ylabel = METRICS[metric]
    error_bars = error_bars and metric == "mean-bpp"
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.subplots_adjust(left=.09, right=.70, bottom=.18, top=.85)
    ax.set_xscale(x_scale)
    ax.set_xlabel("Sum of per-image median native encode time (seconds)", labelpad=12)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(True, color="#dedede", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=.14, y=.16)
    points = report["variants"]
    entries, handles = [], []
    for family, color, marker, label in (("rust", "#11775a", "o", "Native rs_jxl e1..9"),
                                          ("cjxl", "#b44240", "s", "Official cjxl 0.12 e1..9")):
        group = sorted((p for p in points if p["family"] == family), key=lambda p: p["effort"])
        ax.plot([totals[p["id"]]["native_seconds"] for p in group],
                [totals[p["id"]][key] / divisor for p in group], color=color, lw=1, alpha=.65)
        for point in group:
            value = totals[point["id"]]
            x, y = value["native_seconds"], value[key] / divisor
            if error_bars:
                ci = intervals[point["id"]]
                # Percentile endpoints need not bracket the observed sample mean.
                ax.vlines(x, ci["ci_low"], ci["ci_high"], color=color, lw=1)
            ax.plot(x, y, marker=marker, color=color, ms=6, linestyle="none")
            entries.append((x, y, f"e{point['effort']}", color))
        efforts = ",".join(str(p["effort"]) for p in group)
        handles.append(Line2D([], [], color=color, marker=marker,
                              label=label.replace("1..9", efforts)))
    experiments = [p for p in points if p["family"] == "experiment"]
    colors = ["#2469a0", "#9b6200", "#96569a", "#467a19", "#b33975", "#507c7d",
              "#695939", "#344ab0", "#a15d48", "#78629b", "#626e22", "#484848"]
    for index, point in enumerate(experiments):
        value = totals[point["id"]]
        x, y, color = value["native_seconds"], value[key] / divisor, colors[index]
        label = chr(ord("A") + index)
        if error_bars:
            ci = intervals[point["id"]]
            ax.vlines(x, ci["ci_low"], ci["ci_high"], color=color, lw=1)
        ax.plot(x, y, marker="D", color=color, ms=6, linestyle="none")
        entries.append((x, y, label, color))
        handles.append(Line2D([], [], color=color, marker="D", linestyle="none",
                              label=f"{label}  {point['mode']} (e9)"))
    if pareto:
        frontier = pareto_frontier(totals, metric)
        ax.plot([totals[name]["native_seconds"] for name in frontier],
                [totals[name][key] / divisor for name in frontier], linestyle="--",
                color="#777777", alpha=.65, lw=.8, zorder=0)
        handles.append(Line2D([], [], color="#777777", linestyle="--", label="Empirical Pareto frontier"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.025, 1),
              borderaxespad=0, frameon=False, fontsize=9, labelspacing=1)
    config = report["identity"]["config"]
    fig.suptitle("Native rs_jxl / official cjxl 0.12 / context experiments", x=.09, ha="left", y=.96, fontsize=17)
    fig.text(.09, .902, f"{len(report['identity']['inputs'])} matched verified images | "
             f"{config['threads']} threads | {config['repeats']} repeat(s) | patches AUTO", fontsize=11)
    statistic = ("Mean per-image bpp only; no error bars. Each image has equal weight on Y."
                 if metric == "mean-bpp" else
                 "Y is the sum of compressed bytes for all matched images / 1,048,576; no error bars.")
    if error_bars:
        statistic = ("Vertical intervals: 95% paired image-bootstrap CI of mean bpp. "
                     "Not quality loss or codec stochastic error.")
    fig.text(.09, .04, statistic + "\n"
             "Native timing excludes PNG/file I/O and process startup. cjxl rate rounding bounds are in CSV.\n"
             "Mean bpp, pixel-weighted bpp and summed bytes/MPix are separate CSV fields.\n"
             "Whole-pipeline patches AUTO comparison; not the earlier patches-OFF MA-isolation experiment.",
             fontsize=9, color="#444444", linespacing=1.6)
    try:
        place_labels(fig, ax, entries)
    except Exception:
        plt.close(fig)
        raise
    return fig


def write_csv(stream, report, images, totals, intervals, seed, resamples, metric="mean-bpp"):
    fields = ["scope", "variant_id", "family", "mode", "effort", "case_index", "source", "sha256",
              "n", "mean_bpp", "ci_low", "ci_high", "pixel_weighted_bpp", "summed_bytes", "summed_pixels",
              "summed_mpix", "native_seconds", "native_seconds_lower", "native_seconds_upper", "mpix_per_second",
              "pareto", "plot_metric", "threads", "repeats", "bootstrap_seed", "bootstrap_resamples", "ci_contract", "fingerprint"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    frontier = set(pareto_frontier(totals, metric))
    for point in report["variants"]:
        name = point["id"]
        common = dict(variant_id=name, family=point["family"], mode=point["mode"], effort=point["effort"],
                      threads=report["identity"]["config"]["threads"],
                      repeats=report["identity"]["config"]["repeats"], fingerprint=report["fingerprint"],
                      plot_metric=metric)
        ci = dict(**intervals[name], bootstrap_seed=seed, bootstrap_resamples=resamples,
                  ci_contract=CI_CONTRACT) if name in intervals else {}
        writer.writerow(dict(common, scope="corpus", **totals[name], **ci, pareto=name in frontier))
        for value in images[name]:
            case = report["identity"]["inputs"][value["case_index"]]
            writer.writerow(dict(common, scope="image", case_index=value["case_index"], source=case["path"],
                sha256=case["sha256"], n=1, mean_bpp=value["bpp"], pixel_weighted_bpp=value["bpp"],
                summed_bytes=value["bytes"], summed_pixels=value["pixels"], summed_mpix=value["pixels"] / 1e6,
                native_seconds=value["seconds"], native_seconds_lower=value["native_seconds_lower"],
                native_seconds_upper=value["native_seconds_upper"],
                mpix_per_second=value["pixels"] / 1e6 / value["seconds"]))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Complete summary.json from benchmark_context_corpus")
    parser.add_argument("--out", type=Path, required=True, help="PNG destination; also writes .svg and .csv")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261112)
    parser.add_argument("--x-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--metric", choices=tuple(METRICS), default="mean-bpp",
                        help="Y statistic; total-mib sums output bytes and never draws error bars")
    parser.add_argument("--no-error-bars", action="store_true",
                        help="Plot point estimates only, without bootstrap intervals")
    parser.add_argument("--pareto", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".png" or args.bootstrap_resamples < 2 or args.bootstrap_seed < 0:
        parser.error("Require PNG output, at least two resamples and a nonnegative seed")
    return args


def main(argv=None):
    fig = None
    try:
        args = arguments(argv)
        report = bench.loads(args.report.read_text(encoding="utf-8-sig"))
        images, totals = validate_report(report)
        paths = [args.out, args.out.with_suffix(".svg"), args.out.with_suffix(".csv")]
        resolved = [p.resolve() for p in paths]
        bench.require(len(set(resolved)) == len(resolved) and args.report.resolve() not in resolved,
                      "Output paths alias each other or the report")
        for i, path in enumerate(paths):
            if path.exists():
                bench.require(args.overwrite and path.is_file(), "Output exists; use --overwrite")
                bench.require(not path.samefile(args.report)
                              and not any(p.exists() and path.samefile(p) for p in paths[:i]),
                              "Output aliases report or another output")
            path.parent.mkdir(parents=True, exist_ok=True)
        error_bars = not args.no_error_bars and args.metric == "mean-bpp"
        intervals = bootstrap_ci(images, args.bootstrap_resamples, args.bootstrap_seed) if error_bars else {}
        fig = draw_plot(report, totals, intervals, args.pareto, args.x_scale,
                        metric=args.metric, error_bars=error_bars)
        with ExitStack() as stack:
            png = stack.enter_context(paths[0].open("wb" if args.overwrite else "xb"))
            svg = stack.enter_context(paths[1].open("wb" if args.overwrite else "xb"))
            stream = stack.enter_context(paths[2].open("w" if args.overwrite else "x", encoding="utf-8", newline=""))
            fig.savefig(png, format="png", dpi=200)
            fig.savefig(svg, format="svg")
            write_csv(stream, report, images, totals, intervals, args.bootstrap_seed, args.bootstrap_resamples,
                      metric=args.metric)
        for path in paths:
            print(path)
        return 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError, ImportError) as error:
        print(f"Context plot failure: {error}", file=sys.stderr)
        return 1
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
