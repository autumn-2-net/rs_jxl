"""Tiny synthetic receipt tests only; codecs and source access are forbidden."""

import copy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_context_corpus as bench
import report_context_corpus as report_tool
from test_benchmark_context_corpus import tiny_identity, verified_row


def example_report():
    identity = tiny_identity((10, 100, 20), modes=bench.EXPERIMENT_MODES)
    identity["config"]["efforts"] = list(range(1, 10))
    identity["manifest"] = dict(path="/frozen/sample.json", sha256="d" * 64)
    rows = []
    for i in range(3):
        for point in bench.variants(identity["config"]):
            size, seconds = (100, 200, 100)[i], (.01, .02, .03)[i]
            if point["mode"] == "auto12":
                size, seconds = (90, 200, 120)[i], (.02, .02, .03)[i]
            elif point["mode"] == "activity25":
                size, seconds = size + 20, seconds * 3
            rows.append(verified_row(identity, i, point, size=size, seconds=seconds))
    return bench.make_report(identity, bench.fingerprint(identity), rows)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(bench.port, "run_process", side_effect=AssertionError("No codec processes"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.report = example_report()

    def test_full_table_and_interpretation(self):
        analysis = report_tool.analyze(self.report, 40, 7)
        text = report_tool.render_markdown(self.report, analysis, "summary.json", 40, 7)
        self.assertIn("3 images, 30 points, 90/90 verified trials", text)
        for point in self.report["variants"]:
            self.assertIn("| " + point["id"] + " |", text)
        for expected in ("Weighted bpp", "MPix/s", "Paired 95% delta CI", "Wins/ties/losses",
                         "not necessarily an IID", "not quality loss", "patches-OFF MA isolation",
                         "hidden RGB", "16-bit", "One repeat", "sample.json", "Run fingerprint"):
            self.assertIn(expected, text)
        self.assertEqual(len(analysis["size_regressions"]), 4)
        self.assertEqual(len(analysis["time_regressions"]), 4)

    def test_mean_and_bytes_deltas_are_distinct_with_exact_win_tie_loss_counts(self):
        analysis = report_tool.analyze(self.report, 40, 7)
        result = analysis["comparisons"]["experiment-auto12-e9"]
        self.assertEqual(result["mean_bpp_delta"], 0.)
        self.assertEqual(result["bytes_delta"], 10)
        self.assertEqual(result["bytes_delta_percent"], 2.5)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (1, 1, 1))
        self.assertAlmostEqual(result["time_ratio"], .07 / .06)

    def test_paired_delta_ci_uses_same_indices_and_retains_inputs(self):
        original = copy.deepcopy(self.report)
        actual = report_tool.plot.bootstrap_ci
        with patch.object(report_tool.plot, "bootstrap_ci", wraps=actual) as bootstrap:
            analysis = report_tool.analyze(self.report, 40, 9)
        bootstrap.assert_called_once()
        images = bootstrap.call_args.args[0]
        self.assertEqual([v["bpp"] for v in images["delta:experiment-auto12-e9"]], [-8., 0., 8.])
        self.assertEqual([v["case_index"] for v in images["delta:experiment-auto12-e9"]], [0, 1, 2])
        self.assertEqual(analysis, report_tool.analyze(self.report, 40, 9))
        self.assertEqual(self.report, original)
        self.assertLess(analysis["comparisons"]["experiment-auto12-e9"]["ci_low"], 0)
        self.assertGreater(analysis["comparisons"]["experiment-auto12-e9"]["ci_high"], 0)

    def test_regressions_are_positive_ranked_and_capped_at_five(self):
        images, totals = report_tool.plot.validate_report(self.report)
        for name in images:
            if name.startswith("experiment-"):
                for value in images[name]:
                    value["bytes"] *= 4
                    value["seconds"] *= 4
        with patch.object(report_tool.plot, "validate_report", return_value=(images, totals)):
            analysis = report_tool.analyze(self.report, 20, 5)
        for field in ("size_regressions", "time_regressions"):
            rows = analysis[field]
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["ratio"] > 1 and row["delta"] > 0 for row in rows))
            self.assertEqual([r["ratio"] for r in rows], sorted((r["ratio"] for r in rows), reverse=True))

    def test_incomplete_and_missing_baseline_rejected(self):
        self.report["status"] = "incomplete"
        with self.assertRaises(ValueError):
            report_tool.analyze(self.report, 20, 1)
        with patch.object(report_tool.plot, "validate_report", return_value=({}, {})):
            with self.assertRaisesRegex(ValueError, "Rust e9"):
                report_tool.analyze({}, 20, 1)

    def test_markdown_escapes_source_paths(self):
        self.assertEqual(report_tool.cell("a|b`c\n<d>"), "a&#124;b&#96;c &lt;d&gt;")

    def test_cli_writes_only_new_markdown_and_preserves_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, output = root / "summary.json", root / "report.md"
            original = bench.serialized(self.report)
            summary.write_text(original, encoding="utf-8")
            args = [str(summary), "--out", str(output), "--bootstrap-resamples", "20"]
            with redirect_stdout(StringIO()):
                self.assertEqual(report_tool.main(args), 0)
            markdown = output.read_bytes()
            self.assertIn(b"# Context Corpus Results", markdown)
            self.assertEqual(summary.read_text(encoding="utf-8"), original)
            with redirect_stderr(StringIO()):
                self.assertEqual(report_tool.main(args), 1)
            self.assertEqual(output.read_bytes(), markdown)
            incomplete = dict(self.report, status="incomplete")
            summary.write_text(bench.serialized(incomplete), encoding="utf-8")
            with redirect_stderr(StringIO()):
                self.assertEqual(report_tool.main([str(summary), "--out", str(root / "bad.md")]), 1)
            self.assertFalse((root / "bad.md").exists())


if __name__ == "__main__":
    unittest.main()
