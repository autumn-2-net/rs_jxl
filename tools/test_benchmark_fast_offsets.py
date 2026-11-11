"""Focused adapter tests; all encoder/decoder processes are mocked."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_fast_offsets as bench
import benchmark_refinement as refinement
from benchmark_ma_models import trial_order


def measurement(variant, size=100, pixels=67 * 51):
    plan = bench.parameters(variant)
    fast = None if plan is None else dict(
        plan, trees=1, leaves=2, pixels=pixels, offset_leaves=0,
        horizontal=0, vertical=1, single=0)
    offsets = None
    if variant in bench.PAIRS:
        offsets = dict(entropy=False, leaves=False, offset_mode="distribution",
                       offset_sample_limit=1024, domains=1, models=2,
                       refit_proposals=0, refit_wins=0, entropy_wins=0,
                       saved_bits=0, refit_seconds=0.0, model_seconds=0.01,
                       write_seconds=0.01, offset_proposals=1, offset_wins=0,
                       offset_leaves=1, offset_samples=100, offset_population=pixels,
                       offset_candidates=10, offset_seconds=0.01)
    return dict(bytes=size, seconds=0.1, bpp=size * 8 / pixels,
                mpix_per_second=pixels / 1e6 / 0.1, search=False,
                model_costs=dict(tree_bits=1, entropy_header_bits=1, total_bits=2),
                fast_context=fast, refinement=offsets, ma_seed=None)


class FastOffsetsTests(unittest.TestCase):
    def argv(self, folder):
        return ["--encoder", str(folder / "encoder"), "--decoder", str(folder / "djxl"),
                "--synthetic", "--out", str(folder / "report.json")]

    def test_arguments_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            args = bench.arguments(self.argv(folder))
            self.assertEqual(args.variants, list(bench.VARIANTS))
            self.assertEqual((args.effort, args.repeats, args.threads), (9, 3, 8))
            for flag, value in (("--threads", "0"), ("--repeats", "0"),
                                ("--timeout", "nan"), ("--leaf-offset-samples", "0"),
                                ("--effort", "11")):
                with self.subTest(flag=flag), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    bench.arguments(self.argv(folder) + [flag, value])
            args.out.write_text("preserve", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv(folder))
            self.assertEqual(args.out.read_text(encoding="utf-8"), "preserve")

    def test_commands_keep_plans_and_only_add_distribution_to_pairs(self):
        with TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)) + ["--leaf-offset-samples", "37"])
            for variant in bench.VARIANTS:
                command = bench.command(args, "in.png", "out.jxl", variant)
                self.assertNotIn("--refine-leaves", command)
                self.assertNotIn("--ma-seed", command)
                self.assertNotIn("--search", command)
                self.assertEqual("--fast-context" in command, variant != "baseline")
                if variant != "baseline":
                    for flag, key in (("--fast-context", "mode"), ("--context-bias", "bias"),
                                      ("--context-predictors", "predictors")):
                        self.assertEqual(command[command.index(flag) + 1], bench.parameters(variant)[key])
                self.assertEqual("--leaf-offset" in command, variant in bench.PAIRS)
                if variant in bench.PAIRS:
                    self.assertEqual(command[-4:], ["--leaf-offset", "distribution", "--leaf-offset-samples", "37"])

    def test_receipts_require_requested_plan_predictors_and_offset_mode(self):
        for variant in bench.VARIANTS:
            bench.validate_receipt(measurement(variant), variant)
        variant = "fast-full-static-distribution"
        for location, key, value in (("fast_context", "predictors", "basic"),
                                     ("fast_context", "bias", "off"),
                                     ("fast_context", "mode", "leaf"),
                                     ("fast_context", "trees", 0),
                                     ("fast_context", "scan", "scalar"),
                                     ("refinement", "offset_mode", "entropy"),
                                     ("refinement", "leaves", True),
                                     ("refinement", "offset_wins", 2)):
            data = measurement(variant)
            data[location][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_receipt(data, variant)
        for key, value in (("fast_context", None), ("refinement", None), ("ma_seed", {})):
            data = dict(measurement(variant), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_receipt(data, variant)
        data = measurement("baseline")
        del data["fast_context"]
        with self.assertRaises(ValueError):
            bench.validate_receipt(data, "baseline")

    def test_measurement_checks_actual_size_and_native_timing(self):
        for changes in (dict(bytes=99), dict(seconds=0), dict(bpp=1), dict(search=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.validate_measurement(dict(measurement("baseline"), **changes),
                                           "baseline", "auto", 100, 67 * 51)

    def test_size_guard_uses_matching_fast_plan_and_repetition_not_e9(self):
        def sample(repeat, size):
            return dict(repetition=repeat, measurement=dict(bytes=size))
        variants = {v: [sample(0, 200), sample(1, 220)] for v in bench.VARIANTS}
        variants["baseline"] = [sample(0, 50), sample(1, 50)]
        bench.validate_sizes(variants)
        for refined, baseline in bench.PAIRS.items():
            variants[refined][0]["measurement"]["bytes"] = 201
            with self.subTest(refined=refined), self.assertRaisesRegex(ValueError, f"vs {baseline}"):
                bench.validate_sizes(variants)
            variants[refined][0]["measurement"]["bytes"] = 200
        variants["fast-basic-off-distribution"] = [sample(1, 200), sample(1, 200)]
        with self.assertRaisesRegex(ValueError, "paired repetitions"):
            bench.validate_sizes(variants)

    def test_runner_hooks_are_private(self):
        runner = bench.make_runner()
        self.assertIs(runner.command, bench.command)
        self.assertIsNot(refinement.command, bench.command)
        self.assertIsNot(refinement.validate_sizes, bench.validate_sizes)

    def test_mocked_runner_rotation_original_pixels_failures_and_hashes(self):
        for failure in (None, "growth", "pixels", "receipt", "process", "identity"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                folder = Path(directory)
                source = folder / "source.png"
                source.write_bytes(b"original-16bit-hidden-RGB")
                for name in ("encoder", "djxl"):
                    (folder / name).write_bytes(name.encode("ascii"))
                args = bench.arguments(self.argv(folder) + ["--case", str(source), "--repeats", "2"])
                args.synthetic = False
                runner = bench.make_runner()
                calls = []
                original = source.read_bytes()
                expected = SimpleNamespace(width=67, height=51, depth=16, data=original)

                def process(command, env, timeout):
                    if command[0] == str(args.encoder):
                        self.assertEqual(Path(command[1]).read_bytes(), b"metadata-neutral")
                        full = "--context-predictors" in command and command[command.index("--context-predictors") + 1] == "full"
                        variant = "fast-full-static" if full else "fast-basic-off" if "--fast-context" in command else "baseline"
                        if "--leaf-offset" in command:
                            variant += "-distribution"
                        calls.append(variant)
                        size = 50 if variant == "baseline" else 101 if failure == "growth" and variant in bench.PAIRS else 100
                        Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                        data = measurement(variant, size)
                        if failure == "receipt" and full:
                            data["fast_context"]["predictors"] = "basic"
                        if failure == "identity":
                            args.encoder.write_bytes(b"changed binary")
                        return dict(returncode=1 if failure == "process" else 0,
                                    stdout=json.dumps(data), stderr="", wall_seconds=0.2)
                    self.assertEqual(command[0], str(args.decoder))
                    Path(command[2]).write_bytes(b"wrong pixels" if failure == "pixels" else original)
                    return dict(returncode=0, stdout="", stderr="", wall_seconds=0.3)

                def compare(left, right):
                    self.assertIs(left, expected)
                    if left.data != right.data:
                        raise ValueError("Pixel mismatch including hidden RGB/16bit")

                with (patch.object(runner, "run_process", side_effect=process),
                      patch.object(runner, "prepare_png", return_value=(b"metadata-neutral", expected, "mock-original")) as prepare,
                      patch.object(runner, "png_samples", side_effect=lambda data: SimpleNamespace(data=data)),
                      patch.object(runner, "compare_pixels", side_effect=compare) as oracle,
                      patch.object(bench, "make_runner", return_value=runner),
                      redirect_stdout(StringIO())):
                    report = bench.benchmark(args)
                self.assertEqual(report["status"], "passed" if failure is None else "failed", report["errors"])
                self.assertEqual(prepare.call_count, 1)
                saved = json.loads(args.out.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], report["status"])
                if failure is None:
                    self.assertEqual(calls, trial_order(list(bench.VARIANTS), 0, 0)
                                     + trial_order(list(bench.VARIANTS), 1, 0))
                    self.assertEqual(oracle.call_count, 10)
                    summary = report["cases"][0]["summaries"]["fast-full-static-distribution"]
                    self.assertEqual(summary["ratios_to_fast_baseline"]["bytes"], 1)
                    self.assertEqual(summary["ratios_to_baseline"]["bytes"], 2)
                    self.assertIn("mpix_per_second", summary)
                    self.assertEqual(report["cases"][0]["depth"], 16)
                else:
                    self.assertTrue(saved["errors"])


if __name__ == "__main__":
    unittest.main()
