import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_tree_learning import (
    VARIANTS, aggregate, arguments, benchmark, command, load_cases, validate_measurement,
)


class TreeLearningBenchmarkTests(unittest.TestCase):
    def argv(self, root):
        return ["--encoder", str(root / "encoder"), "--decoder", str(root / "djxl"),
                "--case", str(root / "input.png"), "--out", str(root / "report.json")]

    def measurement(self, variant="baseline", size=100, seconds=0.5):
        data = dict(bytes=size, seconds=seconds, bpp=size * 8 / 100,
                    mpix_per_second=100 / 1e6 / seconds, search=False,
                    model_costs=dict(tree_bits=40, entropy_header_bits=80, total_bits=120))
        if variant != "frozen":
            data["tree_learning"] = "legacy" if variant == "baseline" else variant
            data["tree_learning_work"] = (None if variant == "baseline" else dict(
                learners=1, records=100, cost_splits=2, symmetric_splits=0))
        return data

    def test_defaults_and_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.argv(root)
            args = arguments(argv)
            self.assertEqual(args.variants, ["baseline", "cost", "symmetric"])
            self.assertEqual((args.effort, args.threads, args.repeats, args.timeout), (9, 8, 3, 180))
            self.assertEqual(arguments(argv + ["--frozen", "old"]).variants, list(VARIANTS))
            for extra in (["--threads", "0"], ["--threads", "257"], ["--repeats", "0"],
                          ["--warmups", "-1"], ["--effort", "0"], ["--effort", "11"],
                          ["--buffering", "3"], ["--timeout", "nan"], ["--timeout", "inf"],
                          ["--timeout", "0"], ["--timeout", "-1"], ["--variants", "cost"],
                          ["--variants", "baseline", "baseline"],
                          ["--variants", "baseline", "frozen"], ["--variants", "legacy"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        arguments(argv + extra)
            no_cases = ["--encoder", "e", "--decoder", "d", "--out", str(root / "new.json")]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(no_cases)
            self.assertTrue(arguments(no_cases + ["--synthetic"]).synthetic)
            (root / "report.json").touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(argv)

    def test_flag_routing_and_binary_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(self.argv(Path(directory)) + ["--frozen", "old", "--buffering", "2",
                                                          "--effort", "10", "--threads", "4"])
            for variant in VARIANTS:
                cmd = command(args, "input.png", "out.jxl", variant)
                expected = [str(args.frozen if variant == "frozen" else args.encoder), "input.png",
                            "--effort", "10", "--threads", "4", "--model-costs", "--output", "out.jxl",
                            "--buffering", "2"]
                if variant in ("cost", "symmetric"):
                    expected += ["--tree-learning", variant]
                self.assertEqual(cmd, expected)
            args.buffering = None
            self.assertNotIn("--buffering", command(args, "in", "out", "baseline"))
            with self.assertRaises(ValueError):
                command(args, "in", "out", "unknown")

    def test_effort_applicability(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = self.argv(Path(directory))
            for effort in (1, 2, 3):
                args = arguments(argv + ["--effort", str(effort), "--variants", "baseline"])
                self.assertNotIn("--tree-learning", command(args, "in", "out", "baseline"))
                for variant in ("cost", "symmetric"):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        arguments(argv + ["--effort", str(effort), "--variants", "baseline", variant])
                    with self.assertRaises(ValueError):
                        command(args, "in", "out", variant)
            for effort in (4, 9, 10):
                args = arguments(argv + ["--effort", str(effort)])
                for variant in ("cost", "symmetric"):
                    self.assertIn("--tree-learning", command(args, "in", "out", variant))

    def test_mode_evidence_rejects_stale_and_wrong_binary(self):
        for variant in VARIANTS:
            validate_measurement(self.measurement(variant), variant, 100, 100)
        data = self.measurement("frozen")
        data["tree_learning"] = "legacy"
        validate_measurement(data, "frozen", 100, 100)
        for variant in VARIANTS:
            expected = "legacy" if variant in ("baseline", "frozen") else variant
            for wrong in (None, True, {}, "Legacy", "legacy", "cost", "symmetric"):
                if wrong == expected:
                    continue
                data = self.measurement(variant)
                data["tree_learning"] = wrong
                with self.subTest(variant=variant, wrong=wrong), self.assertRaises(ValueError):
                    validate_measurement(data, variant, 100, 100)
        for variant in ("baseline", "cost", "symmetric"):
            data = self.measurement(variant)
            del data["tree_learning"]
            with self.assertRaises(ValueError):
                validate_measurement(data, variant, 100, 100)

    def test_native_and_model_cost_evidence_is_still_required_for_frozen(self):
        for key, value in (("seconds", 0), ("seconds", float("nan")), ("seconds", True),
                           ("bytes", 99), ("bytes", 100.0), ("search", True), ("bpp", 7),
                           ("mpix_per_second", 1), ("model_costs", None),
                           ("model_costs", dict(tree_bits=40, entropy_header_bits=80, total_bits=121)),
                           ("model_costs", dict(tree_bits=500, entropy_header_bits=500, total_bits=1000))):
            data = self.measurement("frozen")
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_measurement(data, "frozen", 100, 100)
        with self.assertRaises(ValueError):
            validate_measurement([], "baseline", 100, 100)

    def test_completed_work_counters_allow_zero_without_requiring_symmetry(self):
        for variant in ("cost", "symmetric"):
            data = self.measurement(variant)
            data["tree_learning_work"] = dict(learners=0, records=0, cost_splits=0, symmetric_splits=0)
            validate_measurement(data, variant, 100, 100)
        data = self.measurement("symmetric")
        data["tree_learning_work"]["symmetric_splits"] = 3
        validate_measurement(data, "symmetric", 100, 100)
        data = self.measurement("frozen")
        data["tree_learning_work"] = None
        validate_measurement(data, "frozen", 100, 100)

    def test_missing_malformed_or_unrequested_work_is_rejected(self):
        for variant in ("baseline", "cost", "symmetric"):
            data = self.measurement(variant)
            del data["tree_learning_work"]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_measurement(data, variant, 100, 100)
        for variant in ("baseline", "frozen"):
            data = self.measurement(variant)
            data["tree_learning_work"] = {}
            with self.assertRaises(ValueError):
                validate_measurement(data, variant, 100, 100)
        for variant in ("cost", "symmetric"):
            for work in (None, [], "work", {}, dict(learners=1)):
                data = self.measurement(variant)
                data["tree_learning_work"] = work
                with self.subTest(variant=variant, work=work), self.assertRaises(ValueError):
                    validate_measurement(data, variant, 100, 100)
            for key in ("learners", "records", "cost_splits", "symmetric_splits"):
                for value in (-1, True, 1.0, "1", None):
                    data = self.measurement(variant)
                    data["tree_learning_work"][key] = value
                    with self.subTest(variant=variant, key=key, value=value), self.assertRaises(ValueError):
                        validate_measurement(data, variant, 100, 100)
        data = self.measurement("cost")
        data["tree_learning_work"]["symmetric_splits"] = 1
        with self.assertRaises(ValueError):
            validate_measurement(data, "cost", 100, 100)

    def test_manifest_and_repeated_case_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one.png", root / "two.png"
            first.touch()
            second.touch()
            manifest = root / "cases.json"
            manifest.write_text(json.dumps(dict(cases=[dict(original="two.png", jxl="ignored.jxl")])))
            args = arguments(self.argv(root) + ["--manifest", str(manifest)])
            args.case = [first]
            self.assertEqual(load_cases(args), [first.resolve(), second.resolve()])
            args.case.append(second)
            with self.assertRaises(ValueError):
                load_cases(args)
            args.manifest = None
            self.assertEqual(load_cases(args), [first.resolve(), second.resolve()])

    def test_empty_manifest_has_clear_failure_report_without_execution(self):
        for manifest_data in ({"cases": []}, {}, []):
            with self.subTest(manifest=manifest_data), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("encoder", "djxl"):
                    (root / name).write_bytes(b"binary")
                manifest = root / "empty.json"
                manifest.write_text(json.dumps(manifest_data))
                args = arguments(["--encoder", str(root / "encoder"), "--decoder", str(root / "djxl"),
                                  "--manifest", str(manifest), "--out", str(root / "report.json")])
                with patch("benchmark_tree_learning.run_process") as process:
                    report = benchmark(args)
                process.assert_not_called()
                self.assertEqual(report["status"], "failed")
                self.assertTrue(any("nonempty cases list" in error for error in report["errors"]))
                self.assertFalse(any("ZeroDivision" in error for error in report["errors"]))
                self.assertEqual(json.loads(args.out.read_text())["status"], "failed")
        with self.assertRaisesRegex(ValueError, "empty case list"):
            aggregate([], ["baseline", "cost"])

    def mock_run(self, root, failure=None):
        for name in ("encoder", "djxl", "old"):
            (root / name).write_bytes(b"binary")
        original = b"original PNG with hidden RGB"
        (root / "input.png").write_bytes(original)
        args = arguments(self.argv(root) + ["--frozen", str(root / "old"),
                                          "--repeats", "2", "--warmups", "1"])
        calls, artifacts = [], []

        def process(cmd, env, timeout):
            self.assertEqual(timeout, 180)
            if cmd[0] == str(args.decoder):
                self.assertIn("--bits_per_sample=16", cmd)
                if failure == "decoder":
                    return dict(returncode=1, stdout="", stderr="djxl failed")
                Path(cmd[2]).write_bytes(original)
                return dict(returncode=0, stdout="", stderr="")
            variant = (cmd[cmd.index("--tree-learning") + 1] if "--tree-learning" in cmd
                       else "frozen" if cmd[0] == str(args.frozen) else "baseline")
            calls.append(variant)
            self.assertNotEqual(Path(cmd[1]), root / "input.png")
            self.assertEqual(Path(cmd[1]).read_bytes(), original)
            # Subsequent trials must keep using the snapshot, not this changed source.
            (root / "input.png").write_bytes(b"changed source")
            if failure == "process":
                return dict(returncode=None, stdout="", stderr="", error="Process timeout")
            artifact = Path(cmd[cmd.index("--output") + 1])
            artifacts.append(artifact)
            size = 100 if variant in ("baseline", "frozen") else 200
            seconds = 0.5 if size == 100 else 1.5
            artifact.write_bytes(variant.encode()[:1] * size)
            data = self.measurement(variant, size, seconds)
            if failure == "stale":
                data.pop("tree_learning", None)
            if failure == "time":
                data["seconds"] = float("nan")
            if failure == "work" and variant != "frozen":
                data.pop("tree_learning_work", None)
            if failure == "binary":
                args.encoder.write_bytes(b"replaced binary")
            return dict(returncode=0, stdout="not JSON" if failure == "json" else json.dumps(data), stderr="")

        with patch("benchmark_tree_learning.run_process", side_effect=process), \
                patch("benchmark_tree_learning.png_samples", return_value=SimpleNamespace(
                    width=10, height=10, depth=16)), \
                patch("benchmark_tree_learning.compare_pixels",
                      side_effect=ValueError("hidden RGB mismatch") if failure == "pixels" else None) as compare, \
                contextlib.redirect_stdout(io.StringIO()):
            report = benchmark(args)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertTrue(all(not artifact.exists() for artifact in artifacts))
        self.assertEqual(json.loads(args.out.read_text())["status"], report["status"])
        return report, calls, compare.call_count

    def test_regressions_are_successful_measurements_and_order_rotates(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checks = self.mock_run(Path(directory))
            self.assertEqual(report["status"], "passed", report["errors"])
            self.assertEqual(checks, 12)
            self.assertEqual(calls, ["frozen", "symmetric", "cost", "baseline",
                                     "baseline", "cost", "symmetric", "frozen",
                                     "cost", "baseline", "frozen", "symmetric"])
            for variant in ("cost", "symmetric"):
                summary = report["aggregate"][variant]
                self.assertEqual(summary["ratios_to_baseline"]["bytes"], 2)
                self.assertEqual(summary["ratios_to_baseline"]["seconds"], 3)
                self.assertAlmostEqual(summary["ratios_to_baseline"]["mpix_per_second"], 1 / 3)
                self.assertEqual(summary["bpp"], 16)
            self.assertEqual(report["cases"][0]["depth"], 16)

    def test_aggregate_is_ratio_of_totals_not_average_of_ratios(self):
        cases = [dict(pixels=100, summaries=dict(baseline=dict(bytes=100, seconds=1),
                                               cost=dict(bytes=200, seconds=2))),
                 dict(pixels=300, summaries=dict(baseline=dict(bytes=900, seconds=9),
                                               cost=dict(bytes=900, seconds=9)))]
        result = aggregate(cases, ["baseline", "cost"])
        self.assertAlmostEqual(result["cost"]["ratios_to_baseline"]["bytes"], 1.1)
        self.assertAlmostEqual(result["cost"]["ratios_to_baseline"]["seconds"], 1.1)
        self.assertEqual(result["baseline"]["bpp"], 20)

    def test_actual_errors_fail_loud_and_cleanup(self):
        for failure in ("stale", "work", "time", "binary", "process", "decoder", "pixels", "json"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, checks = self.mock_run(Path(directory), failure)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                if failure == "binary":
                    self.assertTrue(any("changed during benchmark" in error for error in report["errors"]))
                if failure in ("process", "decoder"):
                    self.assertEqual(checks, 0)

    def test_output_exclusive_after_argument_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(self.argv(Path(directory)))
            args.out.write_text("existing report")
            with self.assertRaises(FileExistsError):
                benchmark(args)
            self.assertEqual(args.out.read_text(), "existing report")


if __name__ == "__main__":
    unittest.main()
