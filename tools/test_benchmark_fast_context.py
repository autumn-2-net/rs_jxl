import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_fast_context import (
    VARIANTS, add_ratios, arguments, benchmark, encoder_command, load_cases,
    summarize, validate_measurement, variant_parameters,
)


class FastContextBenchmarkTests(unittest.TestCase):
    def argv(self, root):
        return ["--encoder", str(root / "encoder"), "--decoder", str(root / "djxl"),
                "--case", str(root / "input.png"), "--out", str(root / "report.json")]

    def measurement(self, variant="auto12", scan="auto"):
        parameters = variant_parameters(variant, scan)
        receipt = None if parameters is None else dict(
            **parameters, trees=1, leaves=2, pixels=300, offset_leaves=0,
            horizontal=1, vertical=1, single=1)
        return dict(bytes=100, seconds=0.5, bpp=8, mpix_per_second=0.0002,
                    search=False, model_costs=dict(tree_bits=40, entropy_header_bits=80, total_bits=120),
                    fast_context=receipt)

    def test_defaults_and_argument_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.argv(root)
            args = arguments(argv)
            self.assertEqual((args.effort, args.threads, args.repeats, args.warmups, args.timeout),
                             (9, 8, 2, 0, 180))
            self.assertEqual(args.variants, list(VARIANTS))
            for extra in (["--threads", "0"], ["--repeats", "0"], ["--warmups", "-1"],
                          ["--timeout", "nan"], ["--timeout", "inf"], ["--timeout", "0"],
                          ["--variants", "leaf"], ["--variants", "baseline", "baseline"],
                          ["--scan", "neon"], ["--variants", "baseline", "--scan", "scalar"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        arguments(argv + extra)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(["--encoder", "e", "--decoder", "d", "--out", "r.json"])
            (root / "report.json").touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(argv)

    def test_manifest_uses_original_not_jxl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.png").touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(cases=[dict(original="input.png", jxl="wrong.jxl")])))
            args = SimpleNamespace(case=[], manifest=manifest)
            self.assertEqual(load_cases(args), [(root / "input.png").resolve()])
            args.case = [root / "input.png"]
            with self.assertRaises(ValueError):
                load_cases(args)
            manifest.write_text('{"cases": [{"jxl": "wrong.jxl"}]}')
            args.case = []
            with self.assertRaises(ValueError):
                load_cases(args)

    def test_commands_exact_flags_and_no_outer_search(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(self.argv(Path(directory)) + ["--scan", "scalar"])
            for variant in VARIANTS:
                command = encoder_command(args, "source.png", "output.jxl", variant)
                self.assertNotIn("--search", command)
                self.assertIn("--model-costs", command)
                if variant == "baseline":
                    for flag in ("--fast-context", "--context-bias", "--context-scan"):
                        self.assertNotIn(flag, command)
                else:
                    params = variant_parameters(variant, "scalar")
                    for flag, key in (("--fast-context", "mode"), ("--context-bias", "bias"),
                                      ("--context-scan", "scan")):
                        self.assertEqual(command[command.index(flag) + 1], params[key])

    def test_receipts_accept_all_variants_and_channel_plans_not_trees(self):
        for variant in VARIANTS:
            for scan in ("auto", "scalar", "avx2", "avx512"):
                validate_measurement(self.measurement(variant, scan), variant, scan, 100, 100)

    def test_receipts_reject_zero_work_and_mismatched_parameters(self):
        for key, value in (("trees", 0), ("leaves", 0), ("pixels", 0), ("trees", True),
                           ("leaves", 2.0), ("offset_leaves", -1), ("horizontal", "1"),
                           ("mode", "Auto12"), ("mode", "leaf"), ("bias", "static"),
                           ("scan", "scalar")):
            data = self.measurement()
            data["fast_context"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_measurement(data, "auto12", "auto", 100, 100)
        data = self.measurement()
        for key in ("horizontal", "vertical", "single"):
            data["fast_context"][key] = 0
        with self.assertRaises(ValueError):
            validate_measurement(data, "auto12", "auto", 100, 100)
        for receipt in (None, {}, "auto12"):
            data["fast_context"] = receipt
            with self.assertRaises(ValueError):
                validate_measurement(data, "auto12", "auto", 100, 100)
        with self.assertRaises(ValueError):
            validate_measurement(self.measurement(), "baseline", "auto", 100, 100)
        data = self.measurement("baseline")
        del data["fast_context"]
        with self.assertRaises(ValueError):
            validate_measurement(data, "baseline", "auto", 100, 100)

    def test_invalid_native_measurements_and_costs(self):
        for key, value in (("seconds", 0), ("seconds", True), ("seconds", float("nan")),
                           ("bytes", 99), ("bytes", 100.0), ("search", True),
                           ("search", 0), ("bpp", 1), ("mpix_per_second", 1),
                           ("model_costs", None)):
            data = self.measurement()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_measurement(data, "auto12", "auto", 100, 100)
        for costs in (dict(tree_bits=True, entropy_header_bits=80, total_bits=81),
                      dict(tree_bits=40, entropy_header_bits=80, total_bits=121),
                      dict(tree_bits=400, entropy_header_bits=500, total_bits=900)):
            data = self.measurement()
            data["model_costs"] = costs
            with self.assertRaises(ValueError):
                validate_measurement(data, "auto12", "auto", 100, 100)

    def test_medians_ratios_and_warmup_exclusion(self):
        samples = []
        for seconds, size, warmup in ((99, 999, True), (1, 100, False), (3, 200, False)):
            data = copy.deepcopy(self.measurement())
            data.update(seconds=seconds, bytes=size)
            samples.append(dict(measurement=data, warmup=warmup, verified=True))
        baseline = summarize(samples, 100)
        self.assertEqual((baseline["seconds"], baseline["bytes"], baseline["bpp"]), (2, 150, 12))
        self.assertEqual(baseline["sizes"], [100, 200])
        summaries = dict(baseline=baseline, auto12=dict(
            seconds=1, bytes=75, bpp=6, mpix_per_second=0.0001))
        add_ratios(summaries)
        self.assertEqual(set(baseline["ratios_to_baseline"].values()), {1})
        self.assertEqual(summaries["auto12"]["ratios_to_baseline"],
                         dict(seconds=0.5, bytes=0.5, bpp=0.5, mpix_per_second=2))

    def run_mock_benchmark(self, root, *, bad_receipt=False, pixel_error=False,
                           execution_error=False, bad_time=False):
        for name in ("encoder", "djxl", "input.png"):
            (root / name).touch()
        args = arguments(self.argv(root) + ["--variants", "baseline", "auto12-adaptive",
                                          "--scan", "scalar", "--warmups", "1"])
        calls, artifacts = [], []

        def process(command, env, timeout):
            self.assertEqual(timeout, 180)
            if command[0] == str(args.encoder):
                variant = "auto12-adaptive" if "--fast-context" in command else "baseline"
                calls.append(variant)
                if execution_error:
                    return dict(returncode=1, stdout="", stderr="encoder failed")
                artifact = Path(command[command.index("--output") + 1])
                artifacts.append(artifact)
                artifact.write_bytes(b"x" * 100)
                data = self.measurement(variant, "scalar")
                if bad_receipt and variant != "baseline":
                    data["fast_context"]["trees"] = 0
                if bad_time:
                    data["seconds"] = float("nan")
                return dict(returncode=0, stdout=json.dumps(data), stderr="")
            Path(command[2]).touch()
            return dict(returncode=0, stdout="", stderr="")

        with patch("benchmark_fast_context.run_process", side_effect=process), \
                patch("benchmark_fast_context.png_samples", return_value=SimpleNamespace(
                    width=10, height=10, depth=16)), \
                patch("benchmark_fast_context.compare_pixels",
                      side_effect=ValueError("hidden RGB mismatch") if pixel_error else None) as compare, \
                contextlib.redirect_stdout(io.StringIO()):
            report = benchmark(args)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertTrue(all(not artifact.exists() for artifact in artifacts))
        self.assertEqual(json.loads(args.out.read_text())["status"], report["status"])
        return report, calls, compare.call_count

    def test_interleaving_every_artifact_verification_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checks = self.run_mock_benchmark(Path(directory))
            self.assertEqual(report["status"], "passed", report["errors"])
            self.assertEqual(calls, ["auto12-adaptive", "baseline", "baseline", "auto12-adaptive",
                                     "auto12-adaptive", "baseline"])
            self.assertEqual(checks, 6)
            self.assertEqual(len(report["cases"][0]["summaries"]), 2)

    def test_receipt_and_pixel_failures_are_visible_and_cleaned(self):
        for kwargs in (dict(bad_receipt=True), dict(pixel_error=True), dict(bad_time=True)):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                report, calls, checks = self.run_mock_benchmark(Path(directory), **kwargs)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                self.assertEqual(checks, 1)

    def test_execution_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checks = self.run_mock_benchmark(Path(directory), execution_error=True)
            self.assertEqual(report["status"], "failed")
            self.assertIn("encoder failed", report["errors"][0])
            self.assertEqual(checks, 0)

    def test_output_exclusive_even_if_created_after_argument_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(self.argv(root))
            args.out.write_text("existing report")
            with self.assertRaises(FileExistsError):
                benchmark(args)
            self.assertEqual(args.out.read_text(), "existing report")


if __name__ == "__main__":
    unittest.main()
