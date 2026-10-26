"""Synthetic encoder report tests; no images, encoders or decoders are needed."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import statistics
import tempfile
import unittest

import summarize_encoder_sample as sample


def fixture(paths, sizes, seconds, *, variant="pilot", warmups=0, encoder_hash="a" * 64):
    cases, rows, summaries = [], [], []
    repeats = len(sizes[0])
    for index, path in enumerate(paths):
        width, height = (10, 10) if index == 0 else (30, 30)
        pixels = width * height
        cases.append(dict(index=index, path=str(path), label=Path(path).name, width=width,
                          height=height, pixels=pixels, depth=8, color_type=6, kind="png",
                          original_sha256=hashlib.sha256(str(path).encode()).hexdigest(),
                          payload_sha256=hashlib.sha256((str(path) + "payload").encode()).hexdigest()))
        for repeat in range(-warmups, repeats):
            size = 999 if repeat < 0 else sizes[index][repeat]
            duration = 999.0 if repeat < 0 else seconds[index][repeat]
            rows.append(dict(case_index=index, variant=variant, repeat=repeat, warmup=repeat < 0,
                             status="passed", bytes=size, seconds=duration, bpp=8 * size / pixels,
                             encode=dict(wall_seconds=duration + 100)))
        summaries.append(dict(case_index=index, variant=variant, trials=repeats,
                              median_bytes=statistics.median(sizes[index]),
                              median_call_seconds=statistics.median(seconds[index]),
                              median_bpp=statistics.median(sizes[index]) * 8 / pixels,
                              median_wall_seconds=statistics.median(seconds[index]) + 100))
    return dict(status="passed", effort=9, threads=16, repeats=repeats, warmups=warmups,
                primary_metric="call_mpix_per_second", encoder="fixture-encoder",
                encoder_sha256=encoder_hash, environment=dict(explicit={}),
                cases=cases, rows=rows, summaries=summaries)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = [self.root / "first" / "same.png", self.root / "second" / "same.png"]
        self.current = fixture(self.paths, [[50], [180]], [[2.0], [3.0]], variant="new")
        self.baseline = fixture(self.paths, [[40], [200]], [[1.0], [6.0]], encoder_hash="b" * 64)

    def write(self, name, report):
        path = self.root / name
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def summarize(self, baseline=None, current=None):
        current_path = self.write("current.json", self.current if current is None else current)
        old_path = self.write("baseline.json", self.baseline if baseline is None else baseline)
        return sample.summarize(current_path, "new", [(old_path, "pilot")])

    def test_totals_are_weighted_sums_and_deltas_use_current_minus_baseline(self):
        report = self.summarize()
        new, old, delta = (report["totals"][key] for key in ("current", "baseline", "delta"))
        self.assertEqual((new["case_count"], new["total_pixels"], new["bytes"], new["call_seconds"]),
                         (2, 1000, 230, 5))
        self.assertAlmostEqual(new["call_mpix_per_second"], 1000 / 1e6 / 5)
        self.assertAlmostEqual(new["bpp"], 1.84)
        self.assertEqual((old["bytes"], old["call_seconds"]), (240, 7))
        self.assertEqual((delta["bytes"], delta["call_seconds"]), (-10, -2))
        self.assertAlmostEqual(delta["bytes_pct"], -1000 / 240)
        self.assertAlmostEqual(delta["call_seconds_pct"], -200 / 7)
        self.assertEqual(len({row["path"] for row in report["cases"]}), 2)
        self.assertEqual(report["rankings"], dict(volume_regressions=[0], worst_slowdowns=[0]))
        self.assertFalse(report["timing_comparable"])
        self.assertFalse(any(path.exists() for path in self.paths))

    def test_first_priority_wins_not_best_size_or_time_and_absent_paths_fall_through(self):
        first = copy.deepcopy(self.baseline)
        for key in ("cases", "rows", "summaries"):
            first[key] = first[key][:1]
        first["rows"][0].update(bytes=100, seconds=10.0, bpp=8.0)
        first["summaries"][0].update(median_bytes=100, median_call_seconds=10.0, median_bpp=8.0)
        current = self.write("current.json", self.current)
        high = self.write("first.json", first)
        low = self.write("second.json", self.baseline)
        report = sample.summarize(current, "new", [(high, "pilot"), (low, "pilot")])
        a, b = report["cases"]
        self.assertEqual((a["baseline"]["bytes"], a["baseline"]["call_seconds"]), (100, 10))
        self.assertEqual([case["baseline"]["origin"]["priority"] for case in (a, b)], [1, 2])
        origin = a["baseline"]["origin"]
        self.assertEqual(origin["report"], str(high.resolve()))
        self.assertEqual(origin["report_sha256"], hashlib.sha256(high.read_bytes()).hexdigest())
        self.assertEqual(origin["encoder_sha256"], "b" * 64)
        self.assertEqual(origin["variant"], "pilot")

    def test_identity_mismatch_is_fatal_in_either_priority_not_a_fallback(self):
        current = self.write("current.json", self.current)
        good = self.write("good.json", self.baseline)
        bad = copy.deepcopy(self.baseline)
        bad["cases"][0]["payload_sha256"] = "f" * 64
        bad_path = self.write("bad.json", bad)
        for specs in ([(bad_path, "pilot"), (good, "pilot")],
                      [(good, "pilot"), (bad_path, "pilot")]):
            with self.subTest(specs=specs), self.assertRaisesRegex(
                    ValueError, "baseline identity mismatch.*payload_sha256"):
                sample.summarize(current, "new", specs)

    def test_hashes_dimensions_depth_and_color_type_are_identity_fields(self):
        for field, value in (("original_sha256", "f" * 64), ("payload_sha256", "f" * 64),
                             ("width", 5), ("height", 5), ("depth", 16), ("color_type", 2)):
            old = copy.deepcopy(self.baseline)
            old["cases"][0][field] = value
            if field in ("width", "height"):
                other = "height" if field == "width" else "width"
                old["cases"][0][other] = 20
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "baseline identity mismatch"):
                self.summarize(baseline=old)

    def test_missing_baseline_does_not_emit_partial_totals(self):
        old = copy.deepcopy(self.baseline)
        for key in ("cases", "rows", "summaries"):
            old[key] = old[key][:1]
        with self.assertRaisesRegex(ValueError, "No baseline for current case"):
            self.summarize(baseline=old)

    def test_case_paths_not_report_indices_determine_pairing(self):
        old = copy.deepcopy(self.baseline)
        for case in old["cases"]:
            case["index"] += 7
        for row in old["rows"] + old["summaries"]:
            row["case_index"] += 7
        old["cases"].reverse()
        old["summaries"].reverse()
        report = self.summarize(baseline=old)
        self.assertEqual([case["case_index"] for case in report["cases"]], [0, 1])
        self.assertEqual([case["baseline"]["case_index"] for case in report["cases"]], [7, 8])
        self.assertEqual([case["baseline"]["bytes"] for case in report["cases"]], [40, 200])

    def test_duplicate_paths_including_normalized_aliases_are_invalid(self):
        for path in (str(self.paths[0]), str(self.paths[0].parent / "child" / ".." / "same.png")):
            current = copy.deepcopy(self.current)
            current["cases"][1]["path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "duplicate case path"):
                self.summarize(current=current)

    def test_absolute_windows_paths_compare_case_insensitively_on_any_host(self):
        self.assertEqual(sample.case_path("X:/fixtures/a/../IMAGE.png"),
                         sample.case_path("x:\\fixtures\\image.png"))
        for path in ("relative.png", "X:relative.png", None):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "absolute"):
                sample.case_path(path)

    def test_effort_and_threads_must_match(self):
        for field, value in (("effort", 8), ("threads", 8)):
            old = copy.deepcopy(self.baseline)
            old[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field + " mismatch"):
                self.summarize(baseline=old)

    def test_repeats_medians_and_warmups_not_wall_times(self):
        current = fixture(self.paths, [[10, 11], [20, 22]], [[1, 3], [2, 4]],
                          variant="new", warmups=1)
        report = self.summarize(current=current)
        self.assertEqual(report["cases"][0]["current"]["bytes"], 10.5)
        self.assertEqual(report["cases"][0]["current"]["trials"], 2)
        self.assertEqual(report["totals"]["current"]["call_seconds"], 5)
        self.assertEqual(report["totals"]["current"]["bytes"], 31.5)

    def test_selected_rows_and_report_must_pass_including_warmups(self):
        for warmups, repeat in ((0, 0), (1, -1)):
            old = fixture(self.paths, [[40], [200]], [[1], [6]], warmups=warmups)
            next(row for row in old["rows"] if row["repeat"] == repeat)["status"] = "failed"
            with self.subTest(warmups=warmups), self.assertRaisesRegex(ValueError, "row is not passed"):
                self.summarize(baseline=old)
        old = copy.deepcopy(self.baseline)
        old["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "report is not passed"):
            self.summarize(baseline=old)

    def test_missing_variant_summary_rows_and_duplicate_summaries_are_invalid(self):
        for key in ("rows", "summaries"):
            old = copy.deepcopy(self.baseline)
            old[key] = []
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing summary|incomplete"):
                self.summarize(baseline=old)
        old = copy.deepcopy(self.baseline)
        old["summaries"].append(copy.deepcopy(old["summaries"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate summary"):
            self.summarize(baseline=old)
        old = copy.deepcopy(self.baseline)
        for row in old["summaries"]:
            row["variant"] = "not-pilot"
        with self.assertRaisesRegex(ValueError, "missing summary for selected variant"):
            self.summarize(baseline=old)

    def test_other_variants_are_not_combined_with_selected_measurements(self):
        old = copy.deepcopy(self.baseline)
        for key in ("rows", "summaries"):
            additions = copy.deepcopy(old[key])
            for row in additions:
                row["variant"] = "other"
            old[key].extend(additions)
        report = self.summarize(baseline=old)
        self.assertEqual(report["totals"]["baseline"]["bytes"], 240)

    def test_stale_or_nonfinite_summary_metrics_are_rejected(self):
        for field in ("median_bytes", "median_call_seconds", "median_bpp"):
            for value in (123.0, float("nan"), float("inf"), 0, -1, True):
                old = copy.deepcopy(self.baseline)
                old["summaries"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.summarize(baseline=old)

    def test_invalid_rows_and_incomplete_repeats_are_rejected(self):
        for field, value in (("seconds", 0), ("seconds", float("nan")), ("seconds", float("inf")),
                             ("bytes", -1), ("bpp", 999), ("warmup", True), ("repeat", 2),
                             ("case_index", 999)):
            old = copy.deepcopy(self.baseline)
            old["rows"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.summarize(baseline=old)
        old = copy.deepcopy(self.baseline)
        old["repeats"] = 2
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.summarize(baseline=old)
        old = fixture(self.paths, [[40, 40], [200, 200]], [[1, 1], [6, 6]])
        old["rows"][1]["repeat"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate repeat"):
            self.summarize(baseline=old)

    def test_invalid_case_metadata_and_wrong_timing_schema_are_rejected(self):
        for field, value in (("path", "relative.png"), ("pixels", 99), ("width", 0),
                             ("original_sha256", "missing"), ("payload_sha256", None)):
            current = copy.deepcopy(self.current)
            current["cases"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(current=current)
        old = copy.deepcopy(self.baseline)
        old["primary_metric"] = "wall_mpix_per_second"
        with self.assertRaisesRegex(ValueError, "encoder-call timing"):
            self.summarize(baseline=old)
        current = copy.deepcopy(self.current)
        current["cases"][1]["index"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate case index"):
            self.summarize(current=current)

    def test_rankings_use_absolute_regressions_and_preserve_current_case_order(self):
        current = fixture(self.paths, [[50], [240]], [[4], [8]], variant="new")
        report = self.summarize(current=current)
        self.assertEqual([case["case_index"] for case in report["cases"]], [0, 1])
        self.assertEqual(report["rankings"]["volume_regressions"], [1, 0])
        self.assertEqual(report["rankings"]["worst_slowdowns"], [0, 1])
        text = sample.markdown(report)
        self.assertIn("NOT a controlled speedup", text)
        self.assertIn("Volume Regressions", text)
        self.assertIn("Worst Slowdowns", text)
        self.assertIn("Report SHA256", text)
        self.assertIn("encoder SHA256", text)
        self.assertEqual(sample.cell("a|b\nc<d>"), "a&#124;b c&lt;d&gt;")

    def test_markdown_preserves_large_and_fractional_byte_counts(self):
        current = fixture(self.paths, [[14178604, 14178605], [200, 200]],
                          [[1, 1], [6, 6]], variant="new")
        text = sample.markdown(self.summarize(current=current))
        self.assertIn("14178604.5", text)
        self.assertIn("14178804.5", text)
        self.assertEqual(sample.byte_count(122163404), "122163404")
        self.assertEqual(sample.byte_count(-10.5, signed=True), "-10.5")
        self.assertEqual(sample.byte_count(10, signed=True), "+10")

    def test_no_regressions_produces_empty_rankings(self):
        current = fixture(self.paths, [[20], [100]], [[0.5], [3]], variant="new")
        report = self.summarize(current=current)
        self.assertEqual(report["rankings"], dict(volume_regressions=[], worst_slowdowns=[]))
        self.assertEqual(sample.markdown(report).count("\nNone.\n"), 2)

    def test_cli_creates_only_two_reports_and_never_overwrites(self):
        current = self.write("current.json", self.current)
        old = self.write("baseline.json", self.baseline)
        originals = {path: path.read_bytes() for path in (current, old)}
        out = self.root / "result"
        argv = ["--current", str(current), "--variant", "new", "--baseline", str(old), "pilot",
                "--baseline", str(old), "pilot",
                "--out", str(out)]
        with redirect_stdout(StringIO()):
            self.assertEqual(sample.main(argv), 0)
        self.assertEqual({path.name for path in out.iterdir()}, {"summary.json", "REPORT.md"})
        exported = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(exported["totals"]["current"]["bytes"], 230)
        self.assertEqual([origin["priority"] for origin in exported["baselines"]], [1, 2])
        self.assertEqual({case["baseline"]["origin"]["priority"] for case in exported["cases"]}, {1})
        snapshots = {path: path.read_bytes() for path in out.iterdir()}
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            sample.main(argv)
        self.assertEqual(error.exception.code, 2)
        for path, contents in {**originals, **snapshots}.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_cli_validation_fails_before_creating_output(self):
        current = self.write("current.json", self.current)
        old = self.write("baseline.json", dict(self.baseline, threads=8))
        out = self.root / "invalid"
        errors = StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as error:
            sample.main(["--current", str(current), "--variant", "new", "--baseline", str(old), "pilot",
                         "--out", str(out)])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("threads mismatch", errors.getvalue())
        self.assertFalse(out.exists())

    def test_invalid_json_has_clear_origin(self):
        path = self.root / "broken.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "broken.json.*invalid JSON"):
            sample.load_report(path, "pilot")


if __name__ == "__main__":
    unittest.main()
