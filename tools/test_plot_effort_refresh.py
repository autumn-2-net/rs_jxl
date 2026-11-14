"""Synthetic effort refresh rendering and output safety tests; no encodes."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_effort_refresh as plot


def fake_report():
    series = {}
    for j, (name, label, _, _, _) in enumerate(plot.STYLES):
        points = []
        for effort in range(1, 10):
            seconds = (1 + j * .22) * 1.6 ** effort
            size = 1000000 * (10 - effort * .3 + j * .15)
            points.append(dict(effort=effort, n=200, mean_bpp=7 - effort * .2 + j * .08,
                summed_bytes=size, pixel_weighted_bpp=size * 8 / 1e8, summed_pixels=100000000,
                native_seconds=seconds, mpix_per_second=100 / seconds,
                native_seconds_lower=seconds * .999, native_seconds_upper=seconds * 1.001))
        series[name] = dict(label=label, points=points)
    return dict(schema="effort-refresh-v1", series=series,
                identity=dict(config=dict(threads=8, efforts=list(range(1, 10)), repeats=1),
                              inputs=[dict(path=f"image-{i}.png") for i in range(200)]))


class PlotTests(unittest.TestCase):
    def validator(self, report):
        return patch.dict(sys.modules, benchmark_effort_refresh=SimpleNamespace(
            validate_report=Mock(return_value=copy.deepcopy(report["series"]))))

    def test_validator_delegation_and_series_mismatch(self):
        report = fake_report()
        with self.validator(report):
            self.assertEqual(plot.validate_report(report), report["series"])
            report["series"]["new"]["points"][0]["mean_bpp"] += 1
            with self.assertRaisesRegex(ValueError, "series mismatch"):
                plot.validate_report(report)
        validator = Mock(side_effect=ValueError("Incomplete cached evidence"))
        with patch.dict(sys.modules, benchmark_effort_refresh=SimpleNamespace(validate_report=validator)):
            with self.assertRaisesRegex(ValueError, "Incomplete cached evidence"):
                plot.validate_report(report)
            validator.assert_called_once_with(report)

    def test_both_metrics_visual_layout_and_numeric_data(self):
        import numpy as np
        for metric in plot.METRICS:
            with self.subTest(metric=metric):
                report = fake_report()
                fig = plot.draw_plot(report, report["series"], metric)
                try:
                    fig.canvas.draw()
                    ax = fig.axes[0]
                    self.assertEqual(ax.get_xscale(), "log")
                    self.assertEqual(len(ax.lines), 3)
                    self.assertEqual(len(ax.collections), 0)
                    self.assertEqual(len({line.get_marker() for line in ax.lines}), 3)
                    self.assertEqual(len({line.get_color() for line in ax.lines}), 3)
                    key, divisor, ylabel = plot.METRICS[metric]
                    self.assertEqual(ax.get_ylabel(), ylabel)
                    for line, (name, *_rest) in zip(ax.lines, plot.STYLES):
                        points = report["series"][name]["points"]
                        self.assertEqual(list(line.get_xdata()), [p["native_seconds"] for p in points])
                        self.assertEqual(list(line.get_ydata()), [p[key] / divisor for p in points])
                    labels = [t for t in ax.texts if t.get_text()]
                    self.assertEqual(len(labels), 27)
                    for effort in range(1, 10):
                        self.assertEqual(sum(t.get_text() == f"e{effort}" for t in labels), 3)
                    renderer = fig.canvas.get_renderer()
                    bounds = [t.get_window_extent(renderer).padded(1) for t in labels]
                    self.assertFalse(any(a.overlaps(b) for i, a in enumerate(bounds) for b in bounds[i + 1:]))
                    legend = ax.get_legend().get_window_extent(renderer)
                    self.assertGreater(legend.x0, ax.get_window_extent(renderer).x1)
                    self.assertLessEqual(legend.x1, fig.bbox.x1)
                    self.assertGreater(float(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].std()), 5)
                    footer = "\n".join(t.get_text() for t in fig.texts)
                    for text in ("cached", "non-simultaneous", "excludes I/O", "no error bars"):
                        self.assertIn(text, footer)
                    stream = StringIO()
                    plot.write_csv(stream, report, report["series"], metric)
                    rows = list(csv.DictReader(StringIO(stream.getvalue())))
                    self.assertEqual(len(rows), 27)
                    for row in rows:
                        p = report["series"][row["series"]]["points"][int(row["effort"]) - 1]
                        for field in plot.POINT_FIELDS:
                            self.assertEqual(float(row[field]), p[field])
                        self.assertEqual(row["cached"], str(row["series"] != "new"))
                        self.assertEqual(row["plot_metric"], metric)
                        self.assertEqual(float(row["plot_y"]), p[key] / divisor)
                finally:
                    import matplotlib.pyplot as plt
                    plt.close(fig)

    def test_cli_exports_and_overwrite_opt_in(self):
        report = fake_report()
        with tempfile.TemporaryDirectory() as temporary, self.validator(report):
            root = Path(temporary)
            source = root / "report.json"
            source.write_text(json.dumps(report), encoding="utf-8")
            for metric in plot.METRICS:
                out = root / f"{metric}.png"
                argv = [str(source), "--out", str(out), "--metric", metric]
                with redirect_stdout(StringIO()):
                    self.assertEqual(plot.main(argv), 0)
                self.assertTrue(out.read_bytes().startswith(b"\x89PNG"))
                self.assertIn("<svg", out.with_suffix(".svg").read_text(encoding="utf-8"))
                self.assertEqual(len(list(csv.DictReader(StringIO(
                    out.with_suffix(".csv").read_text(encoding="utf-8"))))), 27)
                before = out.read_bytes()
                with redirect_stderr(StringIO()):
                    self.assertEqual(plot.main(argv), 1)
                self.assertEqual(out.read_bytes(), before)
                with redirect_stdout(StringIO()):
                    self.assertEqual(plot.main([*argv, "--overwrite"]), 0)

    def test_invalid_evidence_and_json_create_no_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, out = root / "report.json", root / "chart.png"
            source.write_text(json.dumps(fake_report()), encoding="utf-8")
            with patch.object(plot, "validate_report", side_effect=ValueError("Incomplete")):
                with redirect_stderr(StringIO()):
                    self.assertEqual(plot.main([str(source), "--out", str(out)]), 1)
            source.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
            with patch.object(plot, "validate_report") as validator, redirect_stderr(StringIO()):
                self.assertEqual(plot.main([str(source), "--out", str(out)]), 1)
                validator.assert_not_called()
            self.assertEqual(list(root.iterdir()), [source])

    def test_output_aliases_and_existing_companions_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, out = root / "report.json", root / "chart.png"
            source.write_text("evidence", encoding="utf-8")
            with self.assertRaises(ValueError):
                plot.output_paths(out, out, True)
            for suffix in (".png", ".svg", ".csv"):
                target = out.with_suffix(suffix)
                target.write_text("keep", encoding="utf-8")
                with self.assertRaises(ValueError):
                    plot.output_paths(source, out)
                self.assertEqual(target.read_text(encoding="utf-8"), "keep")
                target.unlink()
            # Hard links exercise aliases that Path.resolve cannot detect.
            os.link(source, out.with_suffix(".csv"))
            with self.assertRaises(ValueError):
                plot.output_paths(source, out, True)
            self.assertEqual(source.read_text(encoding="utf-8"), "evidence")
            out.with_suffix(".csv").unlink()
            out.write_text("keep", encoding="utf-8")
            os.link(out, out.with_suffix(".svg"))
            with self.assertRaises(ValueError):
                plot.output_paths(source, out, True)
            out.with_suffix(".svg").unlink()
            out.unlink()
            out.mkdir()
            with self.assertRaises(ValueError):
                plot.output_paths(source, out, True)

    def test_argument_defaults_and_bad_extension(self):
        args = plot.arguments(["report.json", "--out", "chart.png"])
        self.assertEqual(args.metric, "mean-bpp")
        self.assertFalse(args.overwrite)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            plot.arguments(["report.json", "--out", "chart.svg"])


if __name__ == "__main__":
    unittest.main()
