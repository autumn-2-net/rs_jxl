"""Plot one complete benchmark_qlic_candidates schema-v1 report, without encoding.

Example: python tools/plot_patch_tradeoffs.py report.json --out tradeoffs.png \
    --svg --per-case --pareto
"""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import sys
import textwrap


TIMING = "Native discovery+admission+encode only; excludes startup and input/output IO"


@dataclass(frozen=True)
class Point:
    mode: str
    seconds: float
    bytes: float
    pixels: int
    n: int
    case_index: int = -1

    @property
    def bpp(self):
        return self.bytes * 8 / self.pixels


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=1):
    require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{name} must be finite and positive")
    return value


def matching_metrics(record, size, seconds, pixels, name, tolerance=1e-12):
    require(isinstance(record, dict), f"Missing {name}")
    for key, expected in (("bytes", size), ("seconds", seconds),
                          ("bpp", size * 8 / pixels),
                          ("mpix_per_second", pixels / 1e6 / seconds)):
        actual = positive(record.get(key), f"{name}.{key}")
        require(math.isclose(actual, expected, rel_tol=tolerance),
                f"Inconsistent {name}.{key}")


def digest(value, name):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"Invalid {name}")
    return value


def validate_report(report):
    """Recompute medians; reject missing trials/modes instead of intersecting cases.

    Verification is the harness's recorded oracle result, not a new pixel check.
    Plotting deliberately has no dependency on encoder/decoder executables.
    """
    require(isinstance(report, dict), "Report must be an object")
    require(type(report.get("schema_version")) is int and report["schema_version"] == 1,
            "Only benchmark_qlic_candidates schema_version 1 is supported")
    require(report.get("status") == "passed" and report.get("errors") == [],
            "Report must have passed with no errors")
    require(report.get("temporary_outputs_cleaned") is True, "Incomplete report cleanup")
    require(report.get("timing") == TIMING, "Unsupported or missing native timing contract")
    config = report.get("config")
    require(isinstance(config, dict), "Missing config")
    modes = config.get("modes")
    require(isinstance(modes, list) and modes
            and all(isinstance(m, str) and m.strip() == m and m for m in modes),
            "Missing/invalid modes")
    require(len(set(modes)) == len(modes) and "baseline" in modes,
            "Report needs distinct modes including baseline")
    effort = integer(config.get("effort"), "effort", 4)
    threads = integer(config.get("threads"), "threads")
    require(effort <= 10 and threads <= 256, "Invalid effort/threads")
    repeats = integer(config.get("repeats"), "repeats")
    warmups = integer(config.get("warmups"), "warmups", 0)
    cases = report.get("cases")
    require(isinstance(cases, list) and cases, "Report has no cases")
    expected_modes = set(modes)
    sources, points = set(), []
    for index, case in enumerate(cases):
        name = f"case[{index}]"
        require(isinstance(case, dict), f"Invalid {name}")
        source = case.get("source")
        require(isinstance(source, str) and source.strip(), f"Missing {name}.source")
        digest(case.get("sha256"), f"{name}.sha256")
        digest(case.get("snapshot_sha256"), f"{name}.snapshot_sha256")
        require(source not in sources, f"Duplicate case source: {source}")
        sources.add(source)
        width = integer(case.get("width"), f"{name}.width")
        height = integer(case.get("height"), f"{name}.height")
        pixels = integer(case.get("pixels"), f"{name}.pixels")
        require(pixels == width * height, f"Inconsistent {name}.pixels")
        require(type(case.get("depth")) is int and case["depth"] in (8, 16),
                f"Invalid {name}.depth")
        variants, summaries = case.get("variants"), case.get("summaries")
        require(isinstance(variants, dict) and isinstance(summaries, dict)
                and set(variants) == set(summaries) == expected_modes,
                f"{name}: all modes must have exactly the same cases")
        for mode in modes:
            label = f"{name}/{mode}"
            samples = variants[mode]
            require(isinstance(samples, list) and len(samples) == repeats + warmups,
                    f"Incomplete trials: {label}")
            seen, sizes, times = set(), [], []
            for sample in samples:
                require(isinstance(sample, dict), f"Invalid trial: {label}")
                repetition = sample.get("repetition")
                require(type(repetition) is int and -warmups <= repetition < repeats
                        and repetition not in seen, f"Invalid/duplicate repetition: {label}")
                seen.add(repetition)
                require(sample.get("warmup") is (repetition < 0), f"Invalid warmup: {label}")
                require(sample.get("verified") is True, f"Unverified trial: {label}")
                for process in ("encoder", "decoder"):
                    receipt = sample.get(process)
                    require(isinstance(receipt, dict)
                            and type(receipt.get("returncode")) is int
                            and receipt["returncode"] == 0, f"Failed/missing {process}: {label}")
                measurement = sample.get("measurement")
                require(isinstance(measurement, dict), f"Missing native measurement: {label}")
                size = integer(measurement.get("bytes"), f"{label}.bytes")
                seconds = positive(measurement.get("seconds"), f"{label}.seconds")
                require(type(measurement.get("width")) is int
                        and type(measurement.get("height")) is int
                        and (measurement["width"], measurement["height"]) == (width, height),
                        f"Mismatched measurement dimensions: {label}")
                # Native receipts print rounded decimal metrics; Python summaries do not.
                matching_metrics(measurement, size, seconds, pixels, label, tolerance=1e-5)
                if repetition >= 0:
                    sizes.append(size)
                    times.append(seconds)
            size, seconds = statistics.median(sizes), statistics.median(times)
            summary = summaries[mode]
            matching_metrics(summary, size, seconds, pixels, f"{label}.summary")
            for key, expected in (("min_seconds", min(times)), ("max_seconds", max(times))):
                require(positive(summary.get(key), f"{label}.{key}") == expected,
                        f"Inconsistent {label}.{key}")
            require(isinstance(summary.get("sizes"), list)
                    and all(type(s) is int for s in summary["sizes"])
                    and summary["sizes"] == sorted(set(sizes)), f"Inconsistent {label}.sizes")
            points.append(Point(mode, seconds, size, pixels, 1, index))
    aggregate = report.get("aggregate")
    require(isinstance(aggregate, dict) and set(aggregate) == expected_modes,
            "Missing/incomplete aggregate")
    corpus = []
    for mode in modes:
        measured = [p for p in points if p.mode == mode]
        point = Point(mode, sum(p.seconds for p in measured), sum(p.bytes for p in measured),
                      sum(p.pixels for p in measured), len(cases))
        matching_metrics(aggregate[mode], point.bytes, point.seconds, point.pixels,
                         f"aggregate/{mode}")
        corpus.append(point)
    return corpus, points


def pareto_frontier(points):
    """Minimize both axes; ties are nondominated, but draw each coordinate once."""
    coordinates = sorted({(positive(p.seconds, "point.seconds"), positive(p.bpp, "point.bpp"))
                          for p in points})
    frontier, best_bpp = [], math.inf
    for seconds, bpp in coordinates:
        if bpp < best_bpp:
            frontier.append((seconds, bpp))
            best_bpp = bpp
    return frontier


def load_report(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def case_labels(cases, overrides=()):
    """Display-only names; case matching and CSV identities always use raw records."""
    labels = []
    for index, case in enumerate(cases):
        stem = Path(case["source"]).stem
        stem = re.sub(r"(?i)(?<![0-9a-f])[0-9a-f]{16,}(?![0-9a-f])", "", stem)
        stem = " ".join(re.sub(r"[_-]+", " ", stem).split()).strip(" .")
        labels.append((stem[:57] + "..." if len(stem) > 60 else stem) or f"Case {index + 1}")
    seen = set()
    for override in overrides:
        index_text, separator, label = override.partition("=")
        require(separator and index_text.isdecimal(), "Use --case-label INDEX=LABEL (1-based)")
        index = int(index_text) - 1
        label = " ".join(label.split())
        require(0 <= index < len(cases) and index not in seen, "Unknown/duplicate case-label index")
        require(label and len(label) <= 60, "Case label must contain 1 to 60 characters")
        seen.add(index)
        labels[index] = label
    return labels


def draw_plot(report, groups, modes, x_scale, pareto, per_case=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
              "#222222", "#7B4AB5", "#677A00", "#B24A67", "#00858B", "#8A6200",
              "#882255", "#44AA99", "#999933", "#6699CC", "#AA4499", "#117733",
              "#888888", "#332288")
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*")
    styles = {mode: (colors[i % len(colors)], markers[i % len(markers)])
              for i, mode in enumerate(report["config"]["modes"])}
    columns = min(3, len(groups)) if per_case else 1
    rows = math.ceil(len(groups) / columns)
    legend_width = max(2.8, max(len(m) for m in modes) * 0.085 + 0.8)
    height = max(3.7 * rows + 2.1, 2.2 + 0.24 * (len(modes) + 2))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns + legend_width + 0.5, height),
                                 squeeze=False)
        fig.subplots_adjust(left=0.09, right=1 - legend_width / fig.get_figwidth(),
                            bottom=1.15 / fig.get_figheight(), top=1 - 1 / fig.get_figheight(),
                            hspace=0.65, wspace=0.4)
        for ax, (title, points) in zip(axes.flat, groups):
            if pareto:
                frontier = pareto_frontier(points)
                ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                        color="#666666", linestyle="--", linewidth=1.2, zorder=1)
            for point in points:
                color, marker = styles[point.mode]
                ax.scatter(point.seconds, point.bpp, s=85, marker=marker, color=color,
                           edgecolors="white", linewidths=0.65, zorder=3)
            ax.set_xscale(x_scale)
            ax.set_xlabel("Total native time (s)" if per_case else "Sum of per-image median native time (s)")
            ax.set_ylabel("Bits per pixel")
            ax.set_title(title, fontsize=11, pad=12)
            ax.grid(True, which="major", color="#dddddd", linewidth=0.65)
            ax.grid(True, which="minor", axis="x", color="#eeeeee", linewidth=0.4)
            ax.set_axisbelow(True)
            ax.margins(x=0.15, y=0.18)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        handles = [Line2D([], [], linestyle="", marker=styles[m][1], color=styles[m][0],
                          markersize=8, label=m) for m in modes]
        if pareto:
            handles.append(Line2D([], [], linestyle="--", color="#666666", label="Pareto frontier"))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(fig.subplotpars.right + 0.025,
                                                                     fig.subplotpars.top),
                   borderaxespad=0, frameon=False, title="Experiment mode")
        config = report["config"]
        title = {
            "ma_root_lookahead_and_optional_rct_joint": "MA lookahead and RCT tradeoffs",
            "static_activity_context": "Static MA context tradeoffs",
        }.get(report.get("experiment"), "Patch candidate tradeoffs")
        fig.suptitle(title + (" / per case" if per_case else " / corpus"),
                     x=0.09, ha="left", fontsize=16, y=1 - 0.18 / fig.get_figheight())
        fig.text(0.09, 1 - 0.6 / fig.get_figheight(),
                 f"n = {len(report['cases'])} matched, verified cases | effort = {config['effort']} | "
                 f"threads = {config['threads']} | repeats = {config['repeats']}", fontsize=10)
        fig.text(0.09, 0.18 / fig.get_figheight(),
                 "All native analysis/search/encode costs included; startup and input/output I/O excluded.\n"
                 "Warmups excluded; per-image medians. Corpus bpp = 8 x sum(bytes) / sum(pixels).",
                 fontsize=9, color="#444444", linespacing=1.5)
    return fig


def write_csv(stream, report, corpus, points, labels):
    fields = ["scope", "case_index", "case_label", "source", "sha256", "snapshot_sha256", "mode", "n",
              "bytes", "pixels", "native_seconds", "bpp", "pareto", "effort", "threads",
              "repeats", "warmups", "timing"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    frontiers = {-1: set(pareto_frontier(corpus))}
    for index in range(len(report["cases"])):
        frontiers[index] = set(pareto_frontier([p for p in points if p.case_index == index]))
    for point in [*corpus, *points]:
        case = report["cases"][point.case_index] if point.case_index >= 0 else {}
        row = {key: report["config"][key] for key in ("effort", "threads", "repeats", "warmups")}
        row.update(scope="corpus" if point.case_index < 0 else "case",
                   case_index="" if point.case_index < 0 else point.case_index + 1,
                   case_label="" if point.case_index < 0 else labels[point.case_index],
                   mode=point.mode, n=point.n, bytes=point.bytes, pixels=point.pixels,
                   native_seconds=point.seconds, bpp=point.bpp, timing=TIMING,
                   pareto=(point.seconds, point.bpp) in frontiers[point.case_index])
        row.update({key: case.get(key, "") for key in ("source", "sha256", "snapshot_sha256")})
        writer.writerow(row)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="One benchmark_qlic_candidates JSON report")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--modes", nargs="+", help="Report modes to plot (default: all)")
    parser.add_argument("--x-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--svg", action="store_true", help="Also write SVG with the same stem")
    parser.add_argument("--csv", type=Path, help="Audit CSV (default: PNG stem.csv)")
    parser.add_argument("--per-case", action="store_true", help="Also write *_per_case small multiples")
    parser.add_argument("--case-label", action="append", default=[], metavar="INDEX=LABEL",
                        help="Display label for a 1-based report case; repeatable, never changes identities")
    parser.add_argument("--pareto", action="store_true", help="Dashed nondominated frontier of selected modes")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing output files")
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".png":
        parser.error("--out must have a .png extension")
    return args


def main(argv=None):
    args = arguments(argv)
    figures = []
    try:
        report = load_report(args.report)
        corpus, points = validate_report(report)
        labels = case_labels(report["cases"], args.case_label)
        modes = args.modes if args.modes is not None else report["config"]["modes"]
        require(len(set(modes)) == len(modes) and set(modes) <= set(report["config"]["modes"]),
                "Select distinct modes present in the report")
        corpus = [p for p in corpus if p.mode in modes]
        points = [p for p in points if p.mode in modes]
        csv_path = args.csv if args.csv is not None else args.out.with_suffix(".csv")
        jobs = [(args.out, [(f"Corpus summary (n = {len(report['cases'])})", corpus)], False)]
        if args.per_case:
            groups = [(f"{i + 1}. {label} (n = 1)", [p for p in points if p.case_index == i])
                      for i, label in enumerate(labels)]
            # Keep long source names readable without overlapping adjacent panels.
            groups = [(textwrap.fill(title, width=36), group) for title, group in groups]
            jobs.append((args.out.with_name(args.out.stem + "_per_case.png"), groups, True))
        paths = [csv_path]
        for png, _, _ in jobs:
            paths.append(png)
            if args.svg:
                paths.append(png.with_suffix(".svg"))
        resolved = [path.resolve() for path in paths]
        require(len(set(resolved)) == len(resolved), "Output paths must be distinct")
        require(args.report.resolve() not in resolved, "Outputs must not overwrite the input report")
        for i, path in enumerate(paths):
            require(not path.is_dir(), f"Output is a directory: {path}")
            if path.exists():
                require(not path.samefile(args.report), "Output aliases the input report")
                require(not any(other.exists() and path.samefile(other) for other in paths[:i]),
                        "Output paths alias one another")
            require(args.overwrite or not path.exists(), f"Output exists (use --overwrite): {path}")
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve every destination before rendering; exclusive opens protect against races.
        with ExitStack() as stack:
            streams = {path: stack.enter_context(path.open("wb" if args.overwrite else "xb"))
                       for path in paths if path != csv_path}
            csv_stream = stack.enter_context(csv_path.open("w" if args.overwrite else "x",
                                                          encoding="utf-8", newline=""))
            for png, groups, per_case in jobs:
                fig = draw_plot(report, groups, modes, args.x_scale, args.pareto, per_case)
                figures.append(fig)
                fig.savefig(streams[png], format="png", dpi=220)
                if args.svg:
                    fig.savefig(streams[png.with_suffix(".svg")], format="svg")
            write_csv(csv_stream, report, corpus, points, labels)
        for path in paths:
            print(path)
        return 0
    except (OSError, ValueError, TypeError, OverflowError, ImportError) as error:
        print(f"Plot/report failure: {error}", file=sys.stderr)
        return 1
    finally:
        if figures:
            import matplotlib.pyplot as plt
            for fig in figures:
                plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
