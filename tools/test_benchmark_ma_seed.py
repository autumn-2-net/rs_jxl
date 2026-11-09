import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_ma_seed as bench


class MaSeedBenchmarkTests(unittest.TestCase):
    def argv(self, root):
        return ["--encoder", str(root / "encoder"), "--decoder", str(root / "djxl"),
                "--case", str(root / "input.png"), "--out", str(root / "report.json")]

    def measurement(self, variant, effort=2):
        parameters = bench.variant_parameters(variant)
        fast = parameters["fast_context"]
        if fast is not None:
            fast = dict(fast, scan="auto", trees=1, leaves=2, pixels=100,
                        offset_leaves=0, horizontal=1, vertical=0, single=0)
        seed = parameters["ma_seed"]
        if seed is not None:
            seed = dict(seed, **dict.fromkeys(bench.SEED_COUNTERS, 0))
            seed["learners"] = 1
        size = 100 if variant == "baseline" else 80
        seconds = effort if variant == "baseline" else effort / 2
        return dict(bytes=size, seconds=seconds, bpp=size * 8 / 100,
                    mpix_per_second=100 / 1e6 / seconds, search=False,
                    model_costs=dict(tree_bits=10, entropy_header_bits=20, total_bits=30),
                    fast_context=fast, ma_seed=seed)

    def test_arguments_defaults_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.argv(root)
            args = bench.arguments(argv)
            self.assertEqual(args.efforts, [2, 3, 5, 9])
            self.assertEqual(args.ma_seed_depth, 3)
            self.assertEqual(args.variants, list(bench.VARIANTS))
            for extra in (["--efforts", "2", "2"], ["--efforts", "0"],
                          ["--ma-seed-depth", "0"], ["--ma-seed-depth", "9"],
                          ["--variants", "seed-only"], ["--variants", "baseline", "baseline"],
                          ["--threads", "0"], ["--repeats", "0"], ["--warmups", "-1"],
                          ["--timeout", "nan"], ["--timeout", "inf"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        bench.arguments(argv + extra)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(["--encoder", "e", "--decoder", "d", "--out", "r"])

    def test_exact_commands_and_mutually_exclusive_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)))
            for variant in bench.VARIANTS:
                command = bench.encoder_command(args, "input.png", "out.jxl", 5, variant)
                expected = [str(args.encoder), "input.png", "--effort", "5", "--threads", "8",
                            "--model-costs", "--output", "out.jxl"]
                if variant.startswith("fast-"):
                    expected += ["--fast-context", "auto12", "--context-bias", "static",
                                 "--context-predictors", variant[5:]]
                elif variant.startswith("seed-"):
                    expected += ["--ma-seed", variant[5:], "--ma-seed-depth", "3"]
                self.assertEqual(command, expected)

    def test_receipts_accept_zero_prepared_records_and_optional_predictors(self):
        for variant in bench.VARIANTS:
            data = self.measurement(variant)
            bench.validate_measurement(data, variant, 3, data["bytes"], 100)
            if data["fast_context"] is not None:
                del data["fast_context"]["predictors"]
                bench.validate_measurement(data, variant, 3, data["bytes"], 100)

    def test_seed_receipt_strict_types_modes_depth_and_counters(self):
        for key, value in ([('mode', 'repair'), ('depth', True), ('depth', 2), ('depth', 3.0),
                            ('learners', 0)] +
                           [(key, value) for key in bench.SEED_COUNTERS for value in (-1, True, 0.5)]):
            data = self.measurement("seed-only")
            data["ma_seed"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate_measurement(data, "seed-only", 3, 80, 100)
        for key in bench.SEED_COUNTERS:
            data = self.measurement("seed-only")
            del data["ma_seed"][key]
            with self.assertRaises(ValueError):
                bench.validate_measurement(data, "seed-only", 3, 80, 100)

    def test_common_receipts_and_inactive_modes_are_enforced(self):
        for key, value in (("ma_seed", {}), ("fast_context", {}), ("search", True),
                           ("bytes", 99), ("seconds", float("nan")), ("bpp", 3),
                           ("mpix_per_second", 1), ("model_costs", None)):
            data = self.measurement("baseline")
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_measurement(data, "baseline", 3, 100, 100)
        for variant, key in (("baseline", "ma_seed"), ("baseline", "fast_context"),
                             ("seed-only", "ma_seed")):
            data = self.measurement(variant)
            del data[key]
            with self.assertRaises(ValueError):
                bench.validate_measurement(data, variant, 3, data["bytes"], 100)
        data = self.measurement("fast-full")
        data["fast_context"]["predictors"] = "basic"
        with self.assertRaises(ValueError):
            bench.validate_measurement(data, "fast-full", 3, 80, 100)

    def test_manifest_reuses_original_png_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            source.touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(cases=[dict(original="input.png", jxl="wrong.jxl")])))
            args = SimpleNamespace(case=[], manifest=manifest)
            self.assertEqual(bench.load_cases(args), [source.resolve()])
            args.case = [source]
            with self.assertRaises(ValueError):
                bench.load_cases(args)

    def mock_run(self, root, failure=None):
        for name in ("encoder", "djxl", "input.png"):
            (root / name).write_bytes(b"original")
        args = bench.arguments(self.argv(root) + ["--efforts", "2", "5", "--warmups", "1"])
        calls, artifacts = [], []

        def process(command, env, timeout):
            if command[0] == str(args.encoder):
                effort = int(command[command.index("--effort") + 1])
                variant = "baseline"
                if "--ma-seed" in command:
                    variant = "seed-" + command[command.index("--ma-seed") + 1]
                if "--context-predictors" in command:
                    variant = "fast-" + command[command.index("--context-predictors") + 1]
                calls.append((effort, variant))
                self.assertEqual(Path(command[1]).read_bytes(), b"original")
                if failure == "process":
                    return dict(returncode=1, stdout="", stderr="encoder failed")
                data = self.measurement(variant, effort)
                artifact = Path(command[command.index("--output") + 1])
                artifact.write_bytes(b"x" * data["bytes"])
                artifacts.append(artifact)
                if failure == "receipt":
                    data["ma_seed"] = {}
                if failure == "time":
                    data["seconds"] = float("nan")
                if failure == "provenance":
                    (root / "encoder").write_bytes(b"changed")
                return dict(returncode=0, stdout=json.dumps(data), stderr="")
            self.assertIn("--bits_per_sample=16", command)
            Path(command[2]).touch()
            return dict(returncode=0, stdout="", stderr="")

        with patch.object(bench, "run_process", side_effect=process), \
                patch.object(bench, "png_samples", return_value=SimpleNamespace(width=10, height=10, depth=16)), \
                patch.object(bench, "compare_pixels", side_effect=(
                    ValueError("hidden RGB mismatch") if failure == "pixels" else None)) as compare, \
                contextlib.redirect_stdout(io.StringIO()):
            report = bench.benchmark(args)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertTrue(all(not artifact.exists() for artifact in artifacts))
        self.assertEqual(json.loads(args.out.read_text())["status"], report["status"])
        return report, calls, compare.call_count

    def test_full_mock_matrix_order_medians_ratios_receipts_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, calls, checks = self.mock_run(root)
            self.assertEqual(report["status"], "passed", report["errors"])
            args = bench.arguments(self.argv(root)[:-2] + ["--out", str(root / "unused.json"),
                                                         "--efforts", "2", "5", "--warmups", "1"])
            expected = [item for repetition in (-1, 0, 1)
                        for item in bench.matrix_order(args, repetition, 0)]
            self.assertEqual(calls, expected)
            self.assertEqual(checks, 36)
            self.assertEqual([v for e, v in calls[:6]], list(bench.trial_order(list(bench.VARIANTS), -1, 0)))
            for effort, result in report["cases"][0]["efforts"].items():
                self.assertEqual(result["summaries"]["baseline"]["seconds"], int(effort))
                for variant in bench.VARIANTS[1:]:
                    ratios = result["summaries"][variant]["ratios_to_baseline"]
                    self.assertEqual((ratios["bytes"], ratios["seconds"]), (0.8, 0.5))
                for sample in result["variants"]["seed-only"]:
                    self.assertFalse(sample["seed_prepared_work"])
                    self.assertEqual(sample["measurement"]["ma_seed"]["prepared_records"], 0)
            self.assertIn("sha256", report["encoder"])

    def test_failure_reports_verify_artifacts_and_clean_up(self):
        for failure in ("process", "receipt", "time", "pixels", "provenance"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, checks = self.mock_run(Path(directory), failure)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                if failure in ("receipt", "time"):
                    self.assertEqual(checks, 1)
                if failure == "process":
                    self.assertEqual(checks, 0)

    def test_warmups_excluded_from_medians_and_output_is_exclusive(self):
        samples = []
        for seconds, warmup in ((999, True), (1, False), (3, False)):
            data = copy.deepcopy(self.measurement("baseline"))
            data["seconds"] = seconds
            samples.append(dict(warmup=warmup, verified=True, measurement=data))
        self.assertEqual(bench.summarize(samples, 100)["seconds"], 2)
        with tempfile.TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)))
            args.out.write_text("keep")
            with self.assertRaises(FileExistsError):
                bench.benchmark(args)
            self.assertEqual(args.out.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
