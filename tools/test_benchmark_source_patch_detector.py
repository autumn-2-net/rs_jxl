"""Dependency-free harness tests; subprocess and image-oracle boundaries stubbed."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_source_patch_detector as bench
from test_benchmark_qlic_candidates import measurement
from test_compare_cpp_port import make_png


class HarnessTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.encoders = [self.root / name for name in ("base.exe", "integer.exe", "combined.exe")]
        self.decoder = self.root / "djxl.exe"
        for binary in [*self.encoders, self.decoder]:
            binary.write_bytes(binary.name.encode())
        self.source = self.root / "input.png"
        self.out = self.root / "report.json"
        self.argv = ["--decoder", str(self.decoder), "--case", str(self.source),
                     "--out", str(self.out)]
        for label, binary in zip(("baseline", "integer", "combined"), self.encoders):
            self.argv += ["--encoder", f"{label}={binary}"]

    def test_defaults_and_command(self):
        args = bench.arguments(self.argv)
        self.assertEqual((args.effort, args.threads, args.repeats, args.warmups, args.timeout),
                         (9, 8, 3, 0, 300))
        expected = [str(self.encoders[0]), "in.png", "--output", "out.jxl",
                    "--effort", "9", "--threads", "8", "--mode", "source"]
        self.assertEqual(bench.command(args, "baseline", "in.png", "out.jxl"), expected)
        self.assertEqual(bench.command(args, "baseline", "in.png", "out.jxl", True),
                         expected + ["--profile"])
        self.assertEqual(len(bench.arguments(self.argv[:-2]).encoder), 2)

    def test_invalid_cli_never_executes(self):
        extras = (["--threads", "0"], ["--threads", "257"], ["--repeats", "0"],
                  ["--repeats", "101"], ["--warmups", "-1"], ["--warmups", "101"],
                  ["--profile-repeats", "0"], ["--profile-repeats", "101"],
                  ["--timeout", "nan"], ["--timeout", "inf"], ["--timeout", "0"],
                  ["--effort", "3"], ["--effort", "11"], ["--mode", "off"],
                  ["--encoder", "missing-separator"], ["--encoder", "=path"],
                  ["--encoder", "label="], ["--encoder", "baseline=another.exe"],
                  ["--encoder", f"alias={self.encoders[0]}"], ["--learner", "foo"])
        for extra in extras:
            with self.subTest(extra=extra), redirect_stderr(StringIO()), \
                    patch.object(bench.port, "run_process") as run:
                with self.assertRaises(SystemExit):
                    bench.arguments(self.argv + extra)
                run.assert_not_called()
        for argv in (self.argv[:-4], self.argv[:2] + self.argv[4:]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(argv)
        self.out.write_text("keep", encoding="utf-8")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv)
        self.assertEqual(self.out.read_text(), "keep")

    def exercise(self, depth=16, color=6, profile=False, failure=None, synthetic=False, profile_repeats=1, detail=False):
        channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
        samples = ([259, 513, 65534, 0] if depth == 16 else [19, 81, 253, 0])[:channels]
        if color in (4, 6):
            samples[-1] = 0
        original = make_png(depth, color, [samples * 2])
        self.source.write_bytes(original)
        expected = SimpleNamespace(width=2, height=1, depth=depth)
        argv = self.argv[:2] + self.argv[4:] + ["--synthetic"] if synthetic else self.argv
        args = bench.arguments(argv + ["--warmups", "1", "--profile-repeats", str(profile_repeats)]
                               + (["--profile"] if profile else [])
                               + (["--profile-detail"] if detail else []))
        calls, visits, profile_visits = [], [], []

        def run(command, env, timeout):
            calls.append(command)
            self.assertEqual(timeout, 300)
            self.assertNotIn("JXL_TEST_POLICY", env)
            record = dict(returncode=0, stdout="", stderr="", command=command, wall_seconds=900)
            if Path(command[0]) == self.decoder:
                self.assertNotIn("RS_JXL_PROFILE", env)
                self.assertNotIn("RS_JXL_PATCH_SOURCE_PROFILE", env)
                self.assertIn(f"--bits_per_sample={depth}", command)
                Path(command[2]).write_bytes(original)
                if failure == "hidden":
                    changed = samples * 2
                    changed[0] += 1
                    Path(command[2]).write_bytes(make_png(depth, color, [changed]))
                return record
            binary = Path(command[0])
            label = next(label for label, path in args.encoder.items() if path == binary)
            diagnostic = "--profile" in command
            if diagnostic:
                self.assertEqual(env["RS_JXL_PROFILE"], "1")
                self.assertEqual(env.get("RS_JXL_PATCH_SOURCE_PROFILE"), "1" if detail else None)
                profile_visits.append(label)
                seconds = 999
                record["stderr"] = "raw source detector profile\n"
                if failure == "profile":
                    return record | dict(returncode=1, stderr="profile failure")
            else:
                self.assertNotIn("RS_JXL_PROFILE", env)
                self.assertNotIn("RS_JXL_PATCH_SOURCE_PROFILE", env)
                visits.append(label)
                seconds = len([v for v in visits if v == label])
            self.assertEqual(Path(command[1]).read_bytes(), original)
            if failure == "timeout":
                return record | dict(returncode=None, error="Process timeout after 300s", stderr="partial")
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"x" * 13)
            m = measurement("source", size=13, seconds=seconds, width=2, height=1)
            if failure == "receipt":
                m["bytes"] = 14
            if failure == "identity":
                self.encoders[0].write_bytes(b"changed")
            record["stdout"] = "not json" if failure == "json" else json.dumps(m)
            return record

        def compare(oracle, decoded):
            self.assertIs(oracle, expected)
            if decoded != original:
                raise ValueError("Pixel mismatch; hidden RGB is included")

        with patch.object(bench.qlic, "fixtures", return_value=[self.source]) as fixtures, \
                patch.dict(os.environ, {"RS_JXL_PROFILE": "inherited", "JXL_TEST_POLICY": "bad"}), \
                patch.object(bench, "prepare_png", return_value=(original, expected, "unit-oracle")) as prepare, \
                patch.object(bench.port, "png_samples", side_effect=lambda raw: raw), \
                patch.object(bench.port, "compare_pixels", side_effect=compare) as check, \
                patch.object(bench.port, "run_process", side_effect=run), redirect_stdout(StringIO()):
            report = bench.benchmark(args)
        prepare.assert_called_once()
        self.assertEqual(fixtures.call_count, int(synthetic))
        self.assertEqual(prepare.call_args.args[:2], (self.source.resolve(), original))
        if failure not in ("timeout",):
            self.assertTrue(check.called)
        self.assertEqual(json.loads(self.out.read_text())["status"], report["status"])
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertFalse(list(self.root.glob("source-patch-*")))
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),
                         sorted([p.name for p in self.encoders] + ["djxl.exe", "input.png", "report.json"]))
        return report, visits, profile_visits, calls

    def test_interleave_native_medians_and_separate_profiles(self):
        report, visits, diagnostics, calls = self.exercise(profile=True)
        self.assertEqual(report["status"], "passed", report["errors"])
        labels = ["baseline", "integer", "combined"]
        self.assertEqual(visits, [v for r in range(-1, 3) for v in bench.trial_order(labels, r, 0)])
        self.assertEqual(diagnostics, bench.trial_order(labels, 3, 0))
        case = report["cases"][0]
        for label in labels:
            self.assertEqual(case["summaries"][label]["seconds"], 3)
            self.assertEqual(case["summaries"][label]["bytes"], 13)
            self.assertEqual(report["aggregate"][label]["total_median_seconds"], 3)
            diagnostic = case["diagnostics"][label][0]
            self.assertEqual(diagnostic["measurement"]["seconds"], 999)
            self.assertIn("raw source", diagnostic["encoder"]["stderr"])
            self.assertFalse(diagnostic["included_in_timing"])
            identity = report["encoders"][label]
            self.assertEqual(identity["sha256"], bench.port.file_sha256(Path(identity["path"])))
        self.assertIn("RS_JXL_PROFILE", report["environment"]["removed_inherited_names"])
        self.assertEqual(len(calls), 30)

    def test_synthetic_uses_existing_qlic_fixture_provider(self):
        report, _, _, _ = self.exercise(synthetic=True)
        self.assertEqual(report["status"], "passed", report["errors"])

    def test_repeated_profiles_stay_out_of_native_medians(self):
        report, _, diagnostics, _ = self.exercise(profile=True, profile_repeats=3)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["profiling"]["runs_per_case_version"], 3)
        labels = list(report["encoders"])
        self.assertEqual(diagnostics, [label for r in range(3)
                                      for label in bench.trial_order(labels, 3 + r, 0)])
        for label in labels:
            self.assertEqual(len(report["cases"][0]["diagnostics"][label]), 3)
            self.assertEqual(report["aggregate"][label]["total_median_seconds"], 3)

    def test_detail_is_explicit_and_diagnostic_only(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv + ["--profile-detail"])
        report, _, _, _ = self.exercise(profile=True, detail=True)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["profiling"]["explicit"],
                         {"RS_JXL_PROFILE": "1", "RS_JXL_PATCH_SOURCE_PROFILE": "1"})

    def test_profile_failure_remains_separate_and_fails_report(self):
        report, _, _, _ = self.exercise(profile=True, failure="profile")
        self.assertEqual(report["status"], "failed")
        self.assertIn("profile failure", report["errors"][0])
        self.assertNotIn("aggregate", report)
        for summary in report["cases"][0]["summaries"].values():
            self.assertEqual(summary["seconds"], 3)

    def test_all_eight_formats_without_profiling(self):
        for depth in (8, 16):
            for color in (0, 2, 4, 6):
                with self.subTest(depth=depth, color=color):
                    self.out.unlink(missing_ok=True)
                    report, _, diagnostics, _ = self.exercise(depth=depth, color=color)
                    self.assertEqual(report["status"], "passed", report["errors"])
                    self.assertEqual(diagnostics, [])
                    self.assertEqual(report["cases"][0]["depth"], depth)

    def test_failures_are_reported_not_filtered(self):
        for failure in ("timeout", "hidden", "receipt", "json", "identity"):
            with self.subTest(failure=failure):
                self.out.unlink(missing_ok=True)
                report, _, _, _ = self.exercise(failure=failure)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                if failure != "identity":
                    self.assertNotIn("aggregate", report)
                    sample = next(samples[0] for samples in report["cases"][0]["variants"].values() if samples)
                    self.assertFalse(sample["verified"])
                    self.assertIn("encoder", sample)
        self.assertIn("changed during measurement", report["errors"][0])

    def test_manifest_reuses_loader_and_rejects_missing_cases(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(cases=[dict(original="missing.png")])), encoding="utf-8")
        args = bench.arguments(self.argv[:2] + self.argv[4:] + ["--manifest", str(manifest)])
        with patch.object(bench.port, "run_process") as run:
            report = bench.benchmark(args)
        run.assert_not_called()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["errors"])
        self.assertEqual(report["manifest"]["sha256"], bench.port.file_sha256(manifest))

    def test_aggregate_weights_pixels_and_sums_case_medians(self):
        cases = [dict(pixels=10, summaries={"v": dict(bytes=10, seconds=1, bpp=8)}),
                 dict(pixels=90, summaries={"v": dict(bytes=45, seconds=3, bpp=4)})]
        summary = bench.aggregate(cases, ["v"])["v"]
        self.assertEqual(summary["mean_bpp"], 6)
        self.assertEqual(summary["weighted_bpp"], 4.4)
        self.assertEqual(summary["total_bytes"], 55)
        self.assertEqual(summary["total_median_seconds"], 4)
        self.assertEqual(summary["mpix_per_second"], 100 / 1e6 / 4)


if __name__ == "__main__":
    unittest.main()
