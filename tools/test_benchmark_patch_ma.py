"""No builds/encodes: real validators, mocked subprocess/pixel boundaries."""

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
import benchmark_patch_ma as bench
from test_benchmark_qlic_candidates import measurement as qlic_measurement
from test_compare_cpp_port import make_png


POSITIVE = ["source", "source-gather", "source-prev2-residual", "source-prev2-both",
            "source-gather-prev2-residual", "source-gather-prev2-both"]
PROFILE_HEADER = "[rs-jxl profile] inclusive process totals; nested/parallel times overlap\n"


def measurement(variant, enabled=True, available=2, seconds=2):
    mode, gather, previous, target = bench.VARIANTS[variant]
    m = qlic_measurement(mode + "-parallel" if gather else mode,
                         size=13, seconds=seconds, width=2, height=1)
    if mode == "source":
        m.update(source_detector_enabled=enabled, selected_mode="source" if enabled else "off")
    reference = mode == "source" and enabled
    main_set = previous is not None and target in ("residual", "both")
    ref_set = previous is not None and target in ("reference", "both") and reference
    m.update(parallel_gather=gather, source_patch_selected=False if mode == "baseline" else None,
             gather_mode="waves" if "waves" in variant else "channels",
             channels=dict(requested=previous, target=target,
                           main=dict(configured=main_set,
                                     configured_previous_channels=previous if main_set else None,
                                     effective_previous_channels=min(previous if main_set else 2, available)),
                           reference=dict(available=reference, configured=ref_set,
                                          configured_previous_channels=(previous if ref_set else 2) if reference else None,
                                          effective_previous_channels=min(previous if ref_set else 2, available) if reference else None)))
    return m


class HarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.encoder, self.decoder = self.root / "patch_lab.exe", self.root / "djxl.exe"
        for path in (self.encoder, self.decoder):
            path.write_bytes(path.name.encode())
        self.source, self.out = self.root / "input.png", self.root / "report.json"
        self.argv = ["--encoder", str(self.encoder), "--djxl", str(self.decoder),
                     "--case", str(self.source), "--out", str(self.out)]

    def test_arguments_and_commands(self):
        args = bench.arguments(self.argv)
        self.assertEqual(args.variants, bench.DEFAULT_VARIANTS)
        self.assertEqual((args.effort, args.threads, args.repeats), (9, 8, 3))
        for effort in range(4, 11):
            self.assertEqual(bench.arguments(self.argv + ["--effort", str(effort)]).effort, effort)
        for variant, (mode, gather, previous, target) in bench.VARIANTS.items():
            command = bench.command(args, "in.png", "out.jxl", variant)
            self.assertNotIn("--block-size", command)
            self.assertEqual("--parallel-gather" in command, gather)
            self.assertEqual("--gather-waves" in command, "waves" in variant)
            self.assertEqual(command[command.index("--mode") + 1], "off" if mode == "baseline" else mode)
            self.assertEqual("--previous-channels" in command, previous is not None)
            if previous is not None:
                self.assertEqual(command[command.index("--previous-channels") + 1], str(previous))
                self.assertEqual(command[command.index("--channel-target") + 1], target)
            self.assertEqual(bench.command(args, "in.png", "out.jxl", variant, True), command + ["--profile"])
        self.assertEqual(bench.arguments(self.argv + ["--variants", *POSITIVE]).variants, POSITIVE)
        self.assertEqual(bench.arguments(self.argv + ["--variants", "source"]).variants, ["source"])

    def test_invalid_arguments_and_no_overwrite(self):
        for flags in (["--variants", "off", "off"], ["--variants", "invalid"],
                      ["--threads", "0"], ["--threads", "257"], ["--repeats", "0"],
                      ["--repeats", "101"], ["--warmups", "-1"], ["--timeout", "nan"],
                      ["--timeout", "inf"], ["--effort", "3"], ["--effort", "11"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv + flags)
        self.out.write_text("keep")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv)
        self.assertEqual(self.out.read_text(), "keep")

    def test_real_validators_all_routes_default_zero_gray_and_gated(self):
        for variant in bench.VARIANTS:
            for enabled in (False, True):
                for available in (0, 1, 2):
                    bench.validate_measurement(measurement(variant, enabled, available), variant, 13, 2, 1)
        for edit in (lambda m: m.pop("channels"),
                     lambda m: m.update(parallel_gather=False),
                     lambda m: m["channels"]["reference"].update(configured=False),
                     lambda m: m["channels"]["main"].update(effective_previous_channels=3),
                     lambda m: m.update(source_patch_selected=True),
                     lambda m: m.update(bytes=14),
                     lambda m: m.update(block_size=16)):
            m = measurement("source-gather-prev2-both")
            edit(m)
            with self.assertRaises(ValueError):
                bench.validate_measurement(m, "source-gather-prev2-both", 13, 2, 1)

    def test_diagnostic_emission_and_times(self):
        self.assertFalse(bench.patch_profile(PROFILE_HEADER, True)["source_patch_selected"])
        stages = "encode.patch_reference | 1 | 0.000000 | 0.000000\nencode.patch_residual | 1 | 0.200000 | 0.200000\n"
        result = bench.patch_profile(PROFILE_HEADER + stages, True)
        self.assertTrue(result["source_patch_selected"])
        self.assertEqual((result["reference_seconds"], result["residual_seconds"]), (0, .2))
        for stderr, enabled in (("", True), (PROFILE_HEADER + stages, False),
                                (PROFILE_HEADER + stages.splitlines()[0], True),
                                (PROFILE_HEADER + stages.replace("0.200000", "nan"), True),
                                (PROFILE_HEADER + stages + stages, True)):
            with self.assertRaises(ValueError):
                bench.patch_profile(stderr, enabled)

    def exercise(self, failure=None, variants=None, profile=False, manifest=False):
        original = make_png(16, 6, [[259, 513, 65534, 0] * 2])
        self.source.write_bytes(original)
        expected = SimpleNamespace(width=2, height=1, depth=16)
        argv = self.argv + ["--warmups", "1", "--repeats", "2"]
        if variants:
            argv += ["--variants", *variants]
        if profile:
            argv += ["--profile"]
        if manifest:
            path = self.root / "cases.json"
            path.write_text(json.dumps({"cases": [{"original": self.source.name}]}))
            argv = argv[:4] + ["--manifest", str(path)] + argv[6:]
        args = bench.arguments(argv)
        visits = []
        counts = {}

        def process(command, env, timeout):
            self.assertEqual(timeout, 300)
            self.assertNotIn("JXL_TEST_POLICY", env)
            self.assertNotIn("RS_JXL_PATCH_SOURCE_PROFILE", env)
            record = dict(returncode=0, stdout="", stderr="", command=command, wall_seconds=999)
            if Path(command[0]) == self.decoder:
                self.assertNotIn("RS_JXL_PROFILE", env)
                self.assertIn("--bits_per_sample=16", command)
                if failure == "decoder":
                    return record | dict(returncode=1, stderr="decoder failed")
                Path(command[2]).write_bytes(b"hidden-mismatch" if failure == "hidden" else original)
                return record
            diagnostic = "--profile" in command
            self.assertEqual(env.get("RS_JXL_PROFILE"), "1" if diagnostic else None)
            variant = next(v for v in args.variants if bench.command(args, command[1],
                command[command.index("--output") + 1], v, diagnostic) == command)
            visits.append((variant, diagnostic))
            counts[variant] = counts.get(variant, 0) + 1
            self.assertEqual(Path(command[1]).read_bytes(), original)
            if failure == "timeout" and len(visits) == 1:
                return record | dict(returncode=None, error="Process timeout", stderr="partial")
            Path(command[command.index("--output") + 1]).write_bytes(b"x" * 13)
            m = measurement(variant, seconds=999 if diagnostic else counts[variant])
            if failure == "receipt" and len(visits) == 1:
                m["channels"]["main"]["configured"] = "false"
            if failure == "identity":
                self.encoder.write_bytes(b"changed")
            stderr = PROFILE_HEADER
            if diagnostic and variant.startswith("source"):
                stderr += "encode.patch_reference | 1 | 0.1 | 0.1\nencode.patch_residual | 1 | 0.2 | 0.2\n"
            return record | dict(stdout="bad json" if failure == "json" else json.dumps(m), stderr=stderr)

        def compare(oracle, actual):
            self.assertIs(oracle, expected)
            if actual != original:
                raise ValueError("Pixel mismatch; hidden RGB is included")

        with patch.dict(os.environ, {"RS_JXL_PROFILE": "inherited", "JXL_TEST_POLICY": "bad",
                                     "RS_JXL_PATCH_SOURCE_PROFILE": "1"}), \
                patch.object(bench, "prepare_png", return_value=(original, expected, "unit-oracle")), \
                patch.object(bench.port, "run_process", side_effect=process), \
                patch.object(bench.port, "png_samples", side_effect=lambda raw: raw), \
                patch.object(bench.port, "compare_pixels", side_effect=compare), redirect_stdout(StringIO()):
            report = bench.benchmark(args)
        self.assertEqual(json.loads(self.out.read_text())["status"], report["status"])
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertFalse(list(self.root.glob("patch-ma-*")))
        return report, visits, args

    def test_balanced_orders_native_medians_diagnostics_and_summary(self):
        report, visits, args = self.exercise(variants=POSITIVE, profile=True)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(visits, [(v, r == 2) for r in range(-1, 3)
                                 for v in bench.trial_order(args.variants, r, 0)])
        for variant in args.variants:
            summary = report["cases"][0]["summaries"][variant]
            self.assertEqual(summary["seconds"], 2.5)
            aggregate = report["aggregate"][variant]
            self.assertEqual(aggregate["total_median_seconds"], 2.5)
            self.assertEqual(aggregate["total_bytes"], 13)
            self.assertEqual(aggregate["mean_bpp"], 52)
            self.assertEqual(aggregate["weighted_bpp"], 52)
            self.assertAlmostEqual(aggregate["mpix_per_second"], 2 / 1e6 / 2.5)
            diagnostic = report["cases"][0]["diagnostics"][variant][0]
            self.assertFalse(diagnostic["included_in_timing"])
            self.assertTrue(diagnostic["patch_application"]["source_patch_selected"])
            self.assertIsNone(diagnostic["measurement"]["source_patch_selected"])

    def test_failures_retained_and_remaining_trials_run(self):
        for failure in ("timeout", "decoder", "hidden", "receipt", "json", "identity"):
            with self.subTest(failure=failure):
                report, visits, args = self.exercise(failure=failure)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(visits), 3 * len(args.variants))
                self.assertTrue(report["errors"])
                if failure != "identity":
                    self.assertTrue(any("error" in s for c in report["cases"]
                                        for samples in c["variants"].values() for s in samples))
                    self.assertTrue(any(a.get("status") == "incomplete" for a in report["aggregate"].values()))
                self.out.unlink()

    def test_manifest_relative_paths_and_identity(self):
        report, _, _ = self.exercise(manifest=True)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["manifest"]["sha256"], bench.port.file_sha256(self.root / "cases.json"))
        self.assertEqual(report["cases"][0]["sha256"], bench.port.file_sha256(self.source))

    def test_preparation_failure_is_recorded_without_running_processes(self):
        self.source.write_bytes(b"invalid PNG")
        with patch.object(bench.port, "run_process") as process:
            report = bench.benchmark(bench.arguments(self.argv))
        process.assert_not_called()
        self.assertEqual(report["status"], "failed")
        self.assertIn("error", report["cases"][0])
        self.assertTrue(report["errors"])
        self.assertTrue(all(a["status"] == "incomplete" for a in report["aggregate"].values()))

    def test_aggregate_is_pixel_weighted_not_mean_and_all_cases_required(self):
        cases = [dict(pixels=2, summaries={"off": dict(bytes=13, seconds=2, bpp=52)}),
                 dict(pixels=8, summaries={"off": dict(bytes=20, seconds=3, bpp=20)})]
        aggregate = bench.source_bench.aggregate(cases, ["off"])["off"]
        self.assertEqual(aggregate["mean_bpp"], 36)
        self.assertEqual(aggregate["weighted_bpp"], 26.4)
        self.assertEqual(aggregate["total_median_seconds"], 5)
        self.assertEqual(aggregate["mpix_per_second"], 10 / 1e6 / 5)


if __name__ == "__main__":
    unittest.main()
