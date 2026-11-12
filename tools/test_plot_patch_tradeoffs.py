"""Report-only tests. No builds, encodes, decodes, or external input files."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
from io import StringIO
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_patch_tradeoffs as plot


def metrics(size, seconds, pixels):
    return dict(bytes=size, seconds=seconds, bpp=size * 8 / pixels,
                mpix_per_second=pixels / 1e6 / seconds)


def report_fixture(modes=("baseline", "source-both16"), repeats=3, warmups=1):
    report = dict(schema_version=1, status="passed", errors=[], temporary_outputs_cleaned=True,
                  timing=plot.TIMING, config=dict(modes=list(modes), effort=9, threads=8,
                                                 repeats=repeats, warmups=warmups), cases=[])
    for index, pixels in enumerate((100, 900)):
        case = dict(source=f"/corpus/case-{index}.png", sha256=str(index) * 64,
                    snapshot_sha256=str(index + 2) * 64, pixels=pixels, width=10,
                    height=pixels // 10, depth=8 if index == 0 else 16,
                    variants={}, summaries={})
        for m, mode in enumerate(modes):
            times = ((1, 3, 8) if index == 0 else (2, 5, 12)) if m == 0 else (
                (0.5, 1, 20) if index == 0 else (2, 3, 4))
            sizes = ((100, 200, 600) if index == 0 else (50, 100, 900)) if m == 0 else (
                (100, 150, 200) if index == 0 else (200, 300, 400))
            samples, measured_sizes, measured_times = [], [], []
            for repetition in range(-warmups, repeats):
                size = 9999 if repetition < 0 else sizes[repetition % 3]
                seconds = 999 if repetition < 0 else times[repetition % 3]
                sample = dict(repetition=repetition, warmup=repetition < 0, verified=True,
                              encoder=dict(returncode=0, wall_seconds=99999),
                              decoder=dict(returncode=0, wall_seconds=99999),
                              measurement=dict(metrics(size, seconds, pixels), width=10,
                                               height=pixels // 10))
                samples.append(sample)
                if repetition >= 0:
                    measured_sizes.append(size)
                    measured_times.append(seconds)
            case["variants"][mode] = samples
            summary = metrics(statistics.median(measured_sizes), statistics.median(measured_times), pixels)
            summary.update(min_seconds=min(measured_times), max_seconds=max(measured_times),
                           sizes=sorted(set(measured_sizes)))
            case["summaries"][mode] = summary
        report["cases"].append(case)
    report["aggregate"] = {}
    for mode in modes:
        size = sum(c["summaries"][mode]["bytes"] for c in report["cases"])
        seconds = sum(c["summaries"][mode]["seconds"] for c in report["cases"])
        report["aggregate"][mode] = metrics(size, seconds, 1000)
    return report


class ArithmeticTests(unittest.TestCase):
    def test_case_display_names_remove_hashes_without_changing_identity(self):
        cases = [{"source": "/corpus/" + "a" * 64 + "_TMW_Screenshot_15.png"},
                 {"source": "/corpus/" + "b" * 64 + ".png"},
                 {"source": "/corpus/" + "long name " * 12 + ".png"}]
        original = copy.deepcopy(cases)
        labels = plot.case_labels(cases)
        self.assertEqual(labels[:2], ["TMW Screenshot 15", "Case 2"])
        self.assertLessEqual(len(labels[2]), 60)
        self.assertEqual(plot.case_labels(cases, ["2=Color chart"])[1], "Color chart")
        self.assertEqual(cases, original)
        for overrides in (["0=bad"], ["4=bad"], ["1="], ["missing"], ["1=a", "1=b"]):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                plot.case_labels(cases, overrides)

    def test_weighted_corpus_and_sum_of_native_medians_not_wall_or_mean(self):
        corpus, cases = plot.validate_report(report_fixture())
        baseline, candidate = corpus
        self.assertEqual((baseline.bytes, baseline.pixels, baseline.seconds, baseline.n), (300, 1000, 8, 2))
        self.assertEqual(baseline.bpp, 2.4)
        self.assertEqual((candidate.bytes, candidate.seconds, candidate.bpp), (450, 4, 3.6))
        self.assertEqual([p.bpp for p in cases if p.mode == "baseline"], [16, 8 / 9])
        self.assertNotAlmostEqual(baseline.bpp, (16 + 8 / 9) / 2)

    def test_single_and_even_repeats(self):
        for repeats in (1, 2, 4):
            with self.subTest(repeats=repeats):
                corpus, _ = plot.validate_report(report_fixture(repeats=repeats, warmups=0))
                self.assertEqual(corpus[0].n, 2)
        corpus, _ = plot.validate_report(report_fixture(repeats=2))
        self.assertEqual(corpus[0].bytes, 225)
        self.assertEqual(corpus[0].seconds, 5.5)

    def test_all_report_modes_without_hardcoded_names(self):
        modes = ("baseline", "source", "tiles", "offgrid", "block8", "block16",
                 "source-ref8", "source-ref16", "source-res8", "source-res16",
                 "source-both8", "source-both16", "source-ref16-parallel",
                 "source-res16-parallel", "source-both16-parallel", "baseline-parallel", "source-parallel")
        corpus, cases = plot.validate_report(report_fixture(modes=modes))
        self.assertEqual([p.mode for p in corpus], list(modes))
        self.assertEqual(len(cases), 34)

    def test_pareto_exact_ties_equal_x_equal_y_and_dominance(self):
        def point(x, y):
            return plot.Point("m", x, y, 8, 1)
        points = [point(5, 3), point(1, 8), point(2, 5), point(2, 6), point(3, 5),
                  point(4, 7), point(2, 5), point(5, 2), point(6, 2)]
        self.assertEqual(plot.pareto_frontier(points), [(1, 8), (2, 5), (5, 2)])
        self.assertEqual(plot.pareto_frontier(list(reversed(points))), [(1, 8), (2, 5), (5, 2)])
        self.assertEqual(plot.pareto_frontier([]), [])
        self.assertEqual(plot.pareto_frontier([point(1, 1)]), [(1, 1)])
        with self.assertRaises(ValueError):
            plot.pareto_frontier([point(0, 1)])


class SchemaTests(unittest.TestCase):
    def reject(self, mutate):
        report = report_fixture()
        mutate(report)
        with self.assertRaises(ValueError):
            plot.validate_report(report)

    def test_missing_top_level_fields(self):
        for key in ("schema_version", "status", "errors", "temporary_outputs_cleaned",
                    "timing", "config", "cases", "aggregate"):
            with self.subTest(key=key):
                self.reject(lambda r: r.pop(key))

    def test_failed_partial_wrong_schema_and_config(self):
        for key, value in (("schema_version", True), ("schema_version", 2), ("status", "failed"),
                           ("errors", ["failure"]), ("cases", []), ("cases", [None]),
                           ("temporary_outputs_cleaned", False), ("timing", "wall clock")):
            with self.subTest(key=key, value=value):
                self.reject(lambda r: r.update({key: value}))
        for key, value in (("modes", ["baseline", "baseline"]), ("modes", ["source"]),
                           ("modes", []), ("modes", [None]), ("effort", True), ("effort", 11),
                           ("threads", 0), ("threads", 257), ("repeats", 0), ("warmups", -1)):
            with self.subTest(key=key, value=value):
                self.reject(lambda r: r["config"].update({key: value}))

    def test_no_intersection_or_selection_hides_incomplete_case(self):
        for key in ("variants", "summaries"):
            self.reject(lambda r: r["cases"][1][key].pop("source-both16"))
        self.reject(lambda r: r["cases"][1]["variants"].update(other=[]))
        self.reject(lambda r: r["aggregate"].pop("source-both16"))
        self.reject(lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])))

    def test_case_id_dimensions_and_depth(self):
        for key, value in (("source", ""), ("source", "/corpus/case-1.png"),
                           ("sha256", "bad"), ("snapshot_sha256", None), ("width", True),
                           ("height", 0), ("pixels", 99), ("depth", 12)):
            with self.subTest(key=key):
                self.reject(lambda r: r["cases"][0].update({key: value}))

    def test_trials_all_verified_including_warmups_and_exact_repetitions(self):
        for key, value in (("verified", False), ("verified", 1), ("warmup", False),
                           ("repetition", True), ("repetition", 20), ("measurement", None),
                           ("encoder", {}), ("decoder", {"returncode": 1})):
            with self.subTest(key=key):
                self.reject(lambda r: r["cases"][0]["variants"]["baseline"][0].update({key: value}))
        self.reject(lambda r: r["cases"][0]["variants"]["baseline"].pop())
        self.reject(lambda r: r["cases"][0]["variants"]["baseline"][1].update(repetition=-1))

    def test_measurement_and_summary_metrics_finite_positive_and_consistent(self):
        for key in ("bytes", "seconds", "bpp", "mpix_per_second"):
            for value in (None, True, 0, -1, float("inf"), float("nan"), "1"):
                with self.subTest(key=key, value=value):
                    self.reject(lambda r: r["cases"][0]["variants"]["baseline"][1]["measurement"].update({key: value}))
                    self.reject(lambda r: r["cases"][0]["summaries"]["baseline"].update({key: value}))
                    self.reject(lambda r: r["aggregate"]["baseline"].update({key: value}))
        for key, value in (("width", 1), ("height", 10.0), ("bytes", 12.5), ("bpp", 123)):
            self.reject(lambda r: r["cases"][0]["variants"]["baseline"][1]["measurement"].update({key: value}))
        for key, value in (("min_seconds", 2), ("max_seconds", 999), ("sizes", [100]),
                           ("bytes", 201), ("seconds", 3.01)):
            self.reject(lambda r: r["cases"][0]["summaries"]["baseline"].update({key: value}))
        self.reject(lambda r: r["aggregate"]["baseline"].update(seconds=7))


class CliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.report = self.root / "report.json"
        self.report.write_text(json.dumps(report_fixture()), encoding="utf-8")
        self.out = self.root / "chart.png"

    def run_cli(self, *extra):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return plot.main([str(self.report), "--out", str(self.out), *extra])

    def test_png_svg_small_multiples_and_audit_csv(self):
        self.assertEqual(self.run_cli("--svg", "--per-case", "--pareto"), 0)
        for name in ("chart.png", "chart_per_case.png"):
            data = (self.root / name).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(data), 20000)
        for name in ("chart.svg", "chart_per_case.svg"):
            data = (self.root / name).read_text(encoding="utf-8")
            self.assertIn("<svg", data)
            self.assertIn("source-both16", data)
        with (self.root / "chart.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual((rows[0]["scope"], rows[0]["n"], rows[0]["native_seconds"], rows[0]["bpp"]),
                         ("corpus", "2", "8", "2.4"))
        self.assertEqual({r["case_index"] for r in rows if r["scope"] == "case"}, {"1", "2"})
        self.assertTrue(all(r["sha256"] for r in rows if r["scope"] == "case"))

    def test_selected_single_mode_linear_axis_and_custom_csv(self):
        audit = self.root / "audit.csv"
        self.assertEqual(self.run_cli("--modes", "source-both16", "--x-scale", "linear",
                                      "--csv", str(audit), "--case-label", "1=Color chart"), 0)
        with audit.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["mode"] for r in rows}, {"source-both16"})
        self.assertEqual(rows[1]["case_label"], "Color chart")
        self.assertEqual(rows[1]["source"], "/corpus/case-0.png")
        self.assertEqual(plot.arguments([str(self.report), "--out", str(self.out)]).x_scale, "log")

    def test_every_destination_protected_before_any_write(self):
        for name in ("chart.png", "chart.svg", "chart.csv", "chart_per_case.png", "chart_per_case.svg"):
            existing = self.root / name
            existing.write_bytes(b"keep")
            with self.subTest(name=name):
                self.assertEqual(self.run_cli("--svg", "--per-case"), 1)
                self.assertEqual(existing.read_bytes(), b"keep")
                self.assertEqual(len(list(self.root.iterdir())), 2)
            existing.unlink()
        self.out.write_bytes(b"keep")
        self.assertEqual(self.run_cli("--overwrite"), 0)
        self.assertTrue(self.out.read_bytes().startswith(b"\x89PNG"))

    def test_input_and_output_aliases_never_overwritten(self):
        original = self.report.read_bytes()
        self.assertEqual(self.run_cli("--csv", str(self.report), "--overwrite"), 1)
        self.assertEqual(self.report.read_bytes(), original)
        self.assertEqual(self.run_cli("--csv", str(self.out), "--overwrite"), 1)
        self.assertFalse(self.out.exists())

    def test_hardlinked_input_protected_even_with_overwrite(self):
        linked = self.root / "linked.csv"
        try:
            os.link(self.report, linked)
        except OSError as error:
            self.skipTest(f"Hard links unavailable: {error}")
        original = self.report.read_bytes()
        self.assertEqual(self.run_cli("--csv", str(linked), "--overwrite"), 1)
        self.assertEqual(self.report.read_bytes(), original)
        self.assertFalse(self.out.exists())

    def test_bad_json_and_unselected_invalid_mode_fail_without_outputs(self):
        for data in ('{"status":"passed","status":"failed"}', '{"seconds":NaN}', "{", "[]"):
            self.report.write_text(data, encoding="utf-8")
            self.assertEqual(self.run_cli(), 1)
            self.assertFalse(self.out.exists())
        report = report_fixture()
        report["cases"][1]["variants"]["source-both16"][1]["verified"] = False
        self.report.write_text(json.dumps(report), encoding="utf-8")
        self.assertEqual(self.run_cli("--modes", "baseline"), 1)
        self.assertFalse(self.out.exists())

    def test_unknown_and_duplicate_selection(self):
        self.assertEqual(self.run_cli("--modes", "missing"), 1)
        self.assertEqual(self.run_cli("--modes", "baseline", "baseline"), 1)
        self.assertFalse(self.out.exists())

    def test_legend_outside_axes_and_frontier_artist(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report = report_fixture()
        corpus, _ = plot.validate_report(report)
        fig = plot.draw_plot(report, [("Corpus", corpus)], report["config"]["modes"], "log", True)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        ax = fig.axes[0]
        legend_box = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
        self.assertGreater(legend_box.x0, ax.get_window_extent().x1)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.lines[0].get_linestyle(), "--")
        self.assertEqual(len(ax.collections), 2)

    def test_seventeen_modes_unique_symbols_and_legend_fits(self):
        import matplotlib.pyplot as plt
        modes = ("baseline",) + tuple(f"source-experiment-{i}-parallel" for i in range(16))
        report = report_fixture(modes=modes)
        corpus, _ = plot.validate_report(report)
        fig = plot.draw_plot(report, [("Corpus", corpus)], modes, "log", True)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        legend = fig.legends[0]
        box = legend.get_window_extent(fig.canvas.get_renderer())
        self.assertLessEqual(box.x1, fig.bbox.x1)
        self.assertGreater(box.y0, 0.6 * fig.dpi)
        handles = legend.legend_handles[:-1]
        self.assertEqual(len({(h.get_color(), h.get_marker()) for h in handles}), 17)
        renderer = fig.canvas.get_renderer()
        self.assertGreater(fig.axes[0].xaxis.label.get_window_extent(renderer).y0,
                           fig.texts[-1].get_window_extent(renderer).y1)

    def test_static_context_plot_is_not_labeled_as_patch_search(self):
        import matplotlib.pyplot as plt
        report = report_fixture()
        report["experiment"] = "static_activity_context"
        corpus, _ = plot.validate_report(report)
        fig = plot.draw_plot(report, [("Corpus", corpus)], report["config"]["modes"], "log", True)
        self.addCleanup(plt.close, fig)
        self.assertEqual(fig._suptitle.get_text(), "Static MA context tradeoffs / corpus")


if __name__ == "__main__":
    unittest.main()
