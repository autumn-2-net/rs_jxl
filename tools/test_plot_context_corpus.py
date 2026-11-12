"""Statistics, report completeness and small PNG/SVG/CSV render tests; no encodes."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_context_corpus as bench
import plot_context_corpus as plot
from test_benchmark_context_corpus import complete_report


class StatisticsTests(unittest.TestCase):
    def test_complete_report_recomputes_mean_and_native_totals(self):
        report = complete_report((2, 100))
        images, totals = plot.validate_report(report)
        value = totals["rust-e1"]
        self.assertAlmostEqual(value["mean_bpp"], (80 + 1.68) / 2)
        self.assertAlmostEqual(value["pixel_weighted_bpp"], 41 * 8 / 102)
        self.assertEqual(value["native_seconds"], .03)
        self.assertNotEqual(value["native_seconds"], sum(r["wall_seconds"] for r in report["trials"]))
        self.assertEqual(len(images["rust-e1"]), 2)

    def test_bootstrap_paired_indices_deterministic_and_image_not_pixel_weighted(self):
        images = {"a": [dict(case_index=i, bpp=v) for i, v in enumerate([1., 2., 9., 12.])],
                  "b": [dict(case_index=i, bpp=v + 10) for i, v in enumerate([1., 2., 9., 12.])]}
        first = plot.bootstrap_ci(images, 200, 123)
        self.assertEqual(first, plot.bootstrap_ci(images, 200, 123))
        self.assertAlmostEqual(first["b"]["ci_low"] - first["a"]["ci_low"], 10.)
        self.assertAlmostEqual(first["b"]["ci_high"] - first["a"]["ci_high"], 10.)
        self.assertLess(first["a"]["ci_low"], 6.)
        self.assertGreater(first["a"]["ci_high"], 6.)
        self.assertNotEqual(first, plot.bootstrap_ci(images, 200, 124))

    def test_one_image_degenerate_ci_and_pairing_validation(self):
        images = {"a": [dict(case_index=0, bpp=7.)]}
        self.assertEqual(plot.bootstrap_ci(images, 20, 1), {"a": dict(ci_low=7., ci_high=7.)})
        with self.assertRaises(ValueError):
            plot.bootstrap_ci(dict(a=images["a"], b=[]), 20, 1)
        with self.assertRaises(ValueError):
            plot.bootstrap_ci(dict(a=[dict(case_index=1, bpp=7)]), 20, 1)

    def test_incomplete_duplicate_unverified_or_tampered_report_rejected(self):
        good = complete_report()
        changes = [dict(status="incomplete"), dict(integrity_verified=False), dict(errors=["failed"]),
                   dict(verified_trials=1), dict(expected_trials=1), dict(aggregate={}),
                   dict(case_statuses=[]), dict(variants=[])]
        for change in changes:
            report = copy.deepcopy(good)
            report.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                plot.validate_report(report)
        report = copy.deepcopy(good)
        report["trials"].pop()
        with self.assertRaises(ValueError):
            plot.validate_report(report)
        report = copy.deepcopy(good)
        report["trials"][-1] = copy.deepcopy(report["trials"][0])
        with self.assertRaises(ValueError):
            plot.validate_report(report)
        report = copy.deepcopy(good)
        report["trials"][0].update(verified=False, status="failed")
        bench.seal_row(report["trials"][0])
        with self.assertRaises(ValueError):
            plot.validate_report(report)

    def test_empirical_pareto_uses_mean_not_weighted_bpp(self):
        totals = dict(a=dict(native_seconds=1, mean_bpp=10, pixel_weighted_bpp=1),
                      b=dict(native_seconds=2, mean_bpp=9, pixel_weighted_bpp=9),
                      c=dict(native_seconds=3, mean_bpp=11, pixel_weighted_bpp=.1))
        self.assertEqual(plot.pareto_frontier(totals), ["a", "b"])

    def test_size_pareto_uses_total_bytes_not_mean_bpp(self):
        totals = dict(a=dict(native_seconds=1, mean_bpp=10, summed_bytes=100),
                      b=dict(native_seconds=2, mean_bpp=9, summed_bytes=110),
                      c=dict(native_seconds=3, mean_bpp=11, summed_bytes=90))
        self.assertEqual(plot.pareto_frontier(totals), ["a", "b"])
        self.assertEqual(plot.pareto_frontier(totals, "total-mib"), ["a", "c"])

    def test_csv_exports_all_metrics_and_ci_semantics(self):
        report = complete_report()
        images, totals = plot.validate_report(report)
        intervals = plot.bootstrap_ci(images, 100, 7)
        stream = StringIO()
        plot.write_csv(stream, report, images, totals, intervals, 7, 100)
        rows = list(csv.DictReader(StringIO(stream.getvalue())))
        corpus = [row for row in rows if row["scope"] == "corpus"]
        self.assertEqual(len(corpus), len(report["variants"]))
        row = next(r for r in corpus if r["variant_id"] == "rust-e1")
        self.assertEqual(float(row["mean_bpp"]), totals["rust-e1"]["mean_bpp"])
        self.assertEqual(float(row["pixel_weighted_bpp"]), totals["rust-e1"]["pixel_weighted_bpp"])
        self.assertEqual(row["bootstrap_seed"], "7")
        self.assertIn("not quality loss", row["ci_contract"])
        self.assertIn("paired common image indices", row["ci_contract"])
        self.assertEqual(len(rows), len(report["variants"]) * 3)


class RenderTests(unittest.TestCase):
    def test_point_only_plots_use_requested_metric_and_no_intervals(self):
        report = complete_report()
        images, totals = plot.validate_report(report)
        for metric in ("mean-bpp", "total-mib"):
            with self.subTest(metric=metric):
                fig = plot.draw_plot(report, totals, {}, pareto=True, metric=metric, error_bars=False)
                try:
                    ax = fig.axes[0]
                    key, divisor, ylabel = plot.METRICS[metric]
                    self.assertEqual(ax.get_ylabel(), ylabel)
                    self.assertEqual(len(ax.collections), 0)
                    dots = [line for line in ax.lines if line.get_marker() in ("o", "s", "D")]
                    self.assertEqual(len(dots), len(report["variants"]))
                    expected = sorted((v["native_seconds"], v[key] / divisor) for v in totals.values())
                    actual = sorted((float(line.get_xdata()[0]), float(line.get_ydata()[0])) for line in dots)
                    self.assertEqual(actual, expected)
                    footer = "\n".join(text.get_text() for text in fig.texts)
                    self.assertIn("no error bars", footer)
                    self.assertNotIn("95%", footer)
                    stream = StringIO()
                    plot.write_csv(stream, report, images, totals, {}, 7, 100, metric=metric)
                    rows = list(csv.DictReader(StringIO(stream.getvalue())))
                    frontier = set(plot.pareto_frontier(totals, metric))
                    for row in rows:
                        self.assertEqual(row["ci_low"], "")
                        self.assertEqual(row["ci_contract"], "")
                        self.assertEqual(row["bootstrap_resamples"], "")
                        self.assertEqual(row["plot_metric"], metric)
                        if row["scope"] == "corpus":
                            self.assertEqual(row["pareto"], str(row["variant_id"] in frontier))
                finally:
                    import matplotlib.pyplot as plt
                    plt.close(fig)

    def test_point_only_cli_exports_without_bootstrap(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "summary.json"
            report_path.write_text(bench.serialized(complete_report()), encoding="utf-8")
            for metric, options in (("mean-bpp", ["--no-error-bars"]), ("total-mib", [])):
                with self.subTest(metric=metric):
                    out = root / f"{metric}.png"
                    with patch.object(plot, "bootstrap_ci", side_effect=AssertionError("Unexpected bootstrap")):
                        with redirect_stdout(StringIO()):
                            self.assertEqual(plot.main([str(report_path), "--out", str(out),
                                                        "--metric", metric, *options]), 0)
                    self.assertTrue(out.read_bytes().startswith(b"\x89PNG"))
                    self.assertTrue(out.with_suffix(".svg").is_file())
                    with out.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertTrue(all(row["plot_metric"] == metric and not row["ci_low"] for row in rows))

    def test_full_thirty_point_nonoverlapping_labels_and_outside_legend(self):
        report = complete_report(modes=bench.EXPERIMENT_MODES)
        # Use the complete default effort matrix, with compact synthetic receipts.
        from test_benchmark_context_corpus import verified_row
        identity = report["identity"]
        identity["config"]["efforts"] = list(range(1, 10))
        rows = [verified_row(identity, i, point, size=20 + j, seconds=.01 * (j + 1))
                for i in range(2) for j, point in enumerate(bench.variants(identity["config"]))]
        report = bench.make_report(identity, bench.fingerprint(identity), rows)
        images, totals = plot.validate_report(report)
        intervals = plot.bootstrap_ci(images, 50, 5)
        fig = plot.draw_plot(report, totals, intervals, pareto=True)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax = fig.axes[0]
            labels = [t for t in ax.texts if t.get_text()]
            self.assertEqual(len(labels), 30)
            bounds = [t.get_window_extent(renderer).padded(1) for t in labels]
            self.assertFalse(any(a.overlaps(b) for i, a in enumerate(bounds) for b in bounds[i + 1:]))
            legend = ax.get_legend().get_window_extent(renderer)
            self.assertGreater(legend.x0, ax.get_window_extent(renderer).x1)
            self.assertLessEqual(legend.x1, fig.bbox.x1)
            pixels = fig.canvas.buffer_rgba()
            import numpy as np
            self.assertGreater(int(np.asarray(pixels)[:, :, :3].std()), 5)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_png_svg_csv_outputs_and_incomplete_report_no_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "summary.json"
            report_path.write_text(bench.serialized(complete_report()), encoding="utf-8")
            out = root / "chart.png"
            argv = [str(report_path), "--out", str(out), "--bootstrap-resamples", "30"]
            with redirect_stdout(StringIO()):
                self.assertEqual(plot.main(argv), 0)
            self.assertTrue(out.read_bytes().startswith(b"\x89PNG"))
            self.assertIn("<svg", out.with_suffix(".svg").read_text(encoding="utf-8"))
            self.assertIn("mean_bpp", out.with_suffix(".csv").read_text(encoding="utf-8"))
            with redirect_stderr(StringIO()):
                self.assertEqual(plot.main(argv), 1)
            report = complete_report()
            report["status"] = "incomplete"
            report_path.write_text(bench.serialized(report), encoding="utf-8")
            bad = root / "bad.png"
            with redirect_stderr(StringIO()):
                self.assertEqual(plot.main([str(report_path), "--out", str(bad)]), 1)
            self.assertFalse(bad.exists())


if __name__ == "__main__":
    unittest.main()
