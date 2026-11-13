"""Synthetic report tests; no builds, encodes, or benchmark subprocesses."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_source_patch_profile as profile


def record(value, seeds=1):
    return profile.DETAIL_PREFIX + " " + json.dumps(dict(
        stages_ns={stage: value * (index + 1) for index, stage in enumerate(profile.STAGES)},
        counts=dict(seeds=seeds, backgrounds=0)))


def sample(values):
    return dict(profile=True, included_in_timing=False, verified=True,
                encoder=dict(returncode=0, stderr="\n".join(record(v, v) for v in values)),
                decoder=dict(returncode=0))


def report():
    cases = []
    for source, trials, pixels in (("small.png", [[1, 3], [8], [30]], 1),
                                   ("large.png", [[100], [200]], 1000000)):
        cases.append(dict(source=source, pixels=pixels, diagnostics={
            "baseline": [sample(values) for values in trials],
            "candidate": [sample([v * 2 for v in values]) for values in trials]}))
    return dict(schema_version=1, status="passed", errors=[],
                profiling=dict(enabled=True, included_in_timing=False),
                encoders=dict(baseline={}, candidate={}), cases=cases)


class SummaryTests(unittest.TestCase):
    def test_trial_sums_then_case_medians_then_corpus_sum(self):
        result = profile.summarize(report())
        small = result["cases"][0]["variants"]["baseline"]
        self.assertEqual(small["trials"][0]["stages_ns"]["seed_scan"], 4)
        self.assertEqual(small["trials"][0]["counts"]["seeds"], 4)
        self.assertEqual(small["trials"][0]["detail_records"], 2)
        self.assertEqual(small["median"]["stages_ns"]["seed_scan"], 8)
        self.assertEqual(result["cases"][1]["variants"]["baseline"]["median"]
                         ["stages_ns"]["seed_scan"], 150)
        self.assertEqual(result["corpus"]["baseline"]["stages_ns"]["seed_scan"], 158)
        self.assertEqual(result["corpus"]["baseline"]["counts"]["seeds"], 158)
        self.assertEqual(result["corpus"]["candidate"]["stages_ns"]["seed_scan"], 316)
        self.assertEqual(result["corpus"]["baseline"]["counts"]["backgrounds"], 0)

    def test_existing_inclusive_timings_separate_from_detail(self):
        data = report()
        for case in data["cases"]:
            for samples in case["diagnostics"].values():
                for trial in samples:
                    trial["encoder"]["stderr"] += (
                        "\n[rs-jxl profile] inclusive process totals; nested/parallel times overlap\n"
                        + profile.TIMING_HEADER + "\nsource.patch | 2 | 0.001 | 0.0008\n"
                        "encode.patch_analysis | 1 | 0.003 | 0.003\n")
        result = profile.summarize(data)
        totals = result["corpus"]["baseline"]
        self.assertEqual(totals["timing_ns"]["source.patch"], 2000000)
        self.assertEqual(totals["timing_calls"]["source.patch"], 4)
        self.assertEqual(totals["stages_ns"]["seed_scan"], 158)
        stdout = StringIO()
        with redirect_stdout(stdout):
            profile.print_table(result)
        self.assertIn("Inclusive patch scopes (ms; overlapping, not additive)", stdout.getvalue())
        self.assertIn("source.patch | encode.patch_analysis", stdout.getvalue())

    def test_timing_scope_absent_from_case_contributes_zero(self):
        data = report()
        for samples in data["cases"][0]["diagnostics"].values():
            for trial in samples:
                trial["encoder"]["stderr"] += (
                    "\n" + profile.TIMING_HEADER + "\nsource.patch | 2 | 0.001 | 0.0008\n")
        result = profile.summarize(data)
        self.assertEqual(result["corpus"]["baseline"]["timing_ns"]["source.patch"], 1000000)
        self.assertEqual(result["corpus"]["baseline"]["timing_calls"]["source.patch"], 2)

    def test_no_background_zero_record_is_not_missing(self):
        parsed = profile.parse_trial(record(0, 0), "zero")
        self.assertEqual(parsed["detail_records"], 1)
        self.assertTrue(all(v == 0 for v in parsed["stages_ns"].values()))
        with self.assertRaisesRegex(ValueError, "missing source.patch.detail"):
            profile.parse_trial("[rs-jxl profile] no background", "missing")

    def test_bad_detail_fields(self):
        good = json.loads(record(1).split(profile.DETAIL_PREFIX)[1])
        for field in ("stages_ns", "counts"):
            for value in (-1, float("nan"), float("inf"), -float("inf"), True, "12", None, []):
                with self.subTest(field=field, value=value):
                    bad = deepcopy(good)
                    bad[field][next(iter(bad[field]))] = value
                    with self.assertRaises(ValueError):
                        profile.parse_trial(profile.DETAIL_PREFIX + json.dumps(bad), "bad")
            for value in ({}, [], None):
                bad = deepcopy(good)
                bad[field] = value
                with self.assertRaises(ValueError):
                    profile.parse_trial(profile.DETAIL_PREFIX + json.dumps(bad), "bad")
        for payload in ("{", "[]", "null", '{"counts":{},"counts":{}}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                profile.parse_trial(profile.DETAIL_PREFIX + payload, "bad")
        del good["stages_ns"]["seed_scan"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            profile.parse_trial(profile.DETAIL_PREFIX + json.dumps(good), "bad")

    def test_overflow_in_trial_sum_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            profile.parse_trial(record(1e307) + "\n" + record(1e308), "overflow")

    def test_key_alignment_within_trial_and_across_corpus(self):
        extra = json.loads(record(2).split(profile.DETAIL_PREFIX)[1])
        extra["counts"]["extra"] = 0
        changed = profile.DETAIL_PREFIX + json.dumps(extra)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            profile.parse_trial(record(1) + "\n" + changed, "bad")
        data = report()
        data["cases"][1]["diagnostics"]["candidate"][0]["encoder"]["stderr"] = changed
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            profile.summarize(data)

    def test_failed_incomplete_and_unverified_reports(self):
        mutations = [
            lambda r: r.update(status="failed"),
            lambda r: r.update(errors=["failure"]),
            lambda r: r.update(cases=[]),
            lambda r: r.update(schema_version=2),
            lambda r: r["profiling"].update(enabled=False),
            lambda r: r["profiling"].update(runs_per_case_version=3),
            lambda r: r["cases"][0]["diagnostics"].pop("candidate"),
            lambda r: r["cases"][0]["diagnostics"].update(baseline=[]),
        ]
        for mutate in mutations:
            data = report()
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                profile.summarize(data)
        for patch in (dict(verified=False), dict(error="failure"), dict(profile=False),
                      dict(included_in_timing=True), dict(decoder=dict(returncode=1)),
                      dict(encoder=dict(returncode=None, error="timeout")),
                      dict(encoder=dict(returncode=0, stderr="old trace only")),
                      dict(encoder=dict(returncode=0, stderr=record(1), error="failure"))):
            data = report()
            data["cases"][0]["diagnostics"]["baseline"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                profile.summarize(data)

    def test_consistent_declared_repetitions(self):
        data = report()
        data["cases"] = data["cases"][:1]
        data["profiling"]["runs_per_case_version"] = 3
        self.assertEqual(len(profile.summarize(data)["cases"]), 1)

    def test_failed_native_trial_even_when_report_claims_passed(self):
        data = report()
        data["cases"][0]["variants"] = {
            "baseline": [sample([1])], "candidate": [sample([1])]}
        data["cases"][0]["variants"]["baseline"][0]["verified"] = False
        with self.assertRaisesRegex(ValueError, "unverified"):
            profile.summarize(data)

    def test_malformed_timing_rows(self):
        for row in ("stage | 1 | NaN | 0", "stage | -1 | 1 | 0", "stage | 1 | 2",
                    "stage | 1 | 1 | -1", "stage | 1 | inf | 0"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                profile.parse_trial(record(1) + "\n" + profile.TIMING_HEADER + "\n" + row, "bad")

    def test_cli_json_table_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "report.json", Path(directory) / "summary.json"
            source.write_text(json.dumps(report()), encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(profile.main([str(source), "--out", str(out)]), 0)
            self.assertIn("baseline: detail stages (ms)", stdout.getvalue())
            self.assertIn("large.png", stdout.getvalue())
            self.assertIn("CORPUS", stdout.getvalue())
            self.assertEqual(json.loads(out.read_text())["corpus"]["baseline"]
                             ["stages_ns"]["seed_scan"], 158)
            original = out.read_bytes()
            with redirect_stderr(StringIO()):
                self.assertEqual(profile.main([str(source), "--out", str(out)]), 1)
                self.assertEqual(profile.main([str(source), "--out", str(source)]), 1)
            self.assertEqual(out.read_bytes(), original)
            bad = report()
            bad["status"] = "failed"
            source.write_text(json.dumps(bad), encoding="utf-8")
            missing = Path(directory) / "missing.json"
            with redirect_stderr(StringIO()):
                self.assertEqual(profile.main([str(source), "--out", str(missing)]), 1)
            self.assertFalse(missing.exists())
            for payload in ("{", "null", '{"status":"passed","status":"failed"}'):
                source.write_text(payload, encoding="utf-8")
                with redirect_stderr(StringIO()):
                    self.assertEqual(profile.main([str(source), "--out", str(missing)]), 1)
                self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
