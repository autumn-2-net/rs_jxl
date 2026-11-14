import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_effort_history as history
from test_plot_effort_refresh import fake_report


def reports():
    cached = fake_report()
    cached["identity"].update(previous_fingerprint="baseline", binaries=dict(decoder=dict(sha256="decoder")))
    cached["identity"]["config"].update(serial=True, patches="auto", search=False)
    current = copy.deepcopy(cached)
    current["identity"]["config"]["efforts"] = list(range(1, 6))
    for item in current["series"].values():
        item["points"] = item["points"][:5]
    current["series"]["new"]["points"][0]["native_seconds"] /= 10
    return current, cached


class HistoryTests(unittest.TestCase):
    def compose(self, current, cached):
        with patch.object(history.plot, "validate_report", side_effect=lambda r: copy.deepcopy(r["series"])):
            return history.compose(current, cached)

    def test_exact_tail_reuse_without_mutating_evidence(self):
        current, cached = reports()
        originals = copy.deepcopy((current, cached))
        view, series = self.compose(current, cached)
        self.assertEqual((current, cached), originals)
        self.assertEqual(view["fresh_efforts"], [1, 2, 3, 4, 5])
        self.assertEqual(view["cached_efforts"], [6, 7, 8, 9])
        for name in ("new", "cjxl"):
            self.assertEqual(series[name]["points"][:5], current["series"][name]["points"])
            self.assertEqual(series[name]["points"][5:], cached["series"][name]["points"][5:])
        self.assertEqual(series["previous"]["points"], cached["series"]["new"]["points"])
        _, rows = history.csv_data(view, series, "mean-bpp", "new.json", "old.json")
        self.assertEqual(len(rows), 27)
        self.assertEqual(sum(r["cached"] == "False" for r in rows), 5)
        self.assertEqual(sum(r["report_source"] == "old.json" for r in rows), 17)

    def test_rejects_mismatched_cohorts_settings_and_baselines(self):
        for mutation in ("cohort", "threads", "repeats", "decoder", "baseline", "point"):
            current, cached = reports()
            if mutation == "cohort":
                cached["identity"]["inputs"].pop()
            elif mutation in ("threads", "repeats"):
                cached["identity"]["config"][mutation] *= 2
            elif mutation == "decoder":
                cached["identity"]["binaries"]["decoder"]["sha256"] = "other"
            elif mutation == "baseline":
                cached["identity"]["previous_fingerprint"] = "other"
            else:
                cached["series"]["previous"]["points"][0]["mean_bpp"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compose(current, cached)

    def test_plot_labels_cached_points_and_has_no_error_bars(self):
        import matplotlib.pyplot as plt
        current, cached = reports()
        view, series = self.compose(current, cached)
        for metric in history.plot.METRICS:
            fig = history.draw(view, series, metric)
            try:
                fig.canvas.draw()
                ax = fig.axes[0]
                self.assertEqual(len(ax.lines), 4)
                self.assertEqual(len(ax.collections), 0)
                self.assertEqual(list(ax.lines[0].get_markevery()), [0, 1, 2, 3, 4])
                self.assertEqual(len(ax.lines[3].get_xdata()), 4)
                self.assertEqual(ax.lines[3].get_markerfacecolor(), "white")
                self.assertEqual(len([t for t in ax.texts if t.get_text()]), 27)
                self.assertIn("cached e6-e9", fig.texts[-1].get_text())
            finally:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
