"""Narrow cross-channel adapter tests; encoder/decoder calls are mocked."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_fast_cross as bench
import benchmark_fast_offsets as offsets
import benchmark_refinement as refinement
from benchmark_ma_models import trial_order


def measurement(variant, size=100, pixels=67 * 51, selected=0):
    plan = bench.parameters(variant)
    receipt = None if plan is None else dict(
        plan, trees=1, leaves=2, pixels=pixels, offset_leaves=0,
        horizontal=0, vertical=1, single=1, cross_channel_plans=selected)
    return dict(bytes=size, seconds=0.1, bpp=size * 8 / pixels,
                mpix_per_second=pixels / 1e6 / 0.1, search=False,
                model_costs=dict(tree_bits=1, entropy_header_bits=1, total_bits=2),
                fast_context=receipt, refinement=None, ma_seed=None)


class FastCrossTests(unittest.TestCase):
    def argv(self, folder):
        return ["--encoder", str(folder / "encoder"), "--decoder", str(folder / "djxl"),
                "--synthetic", "--out", str(folder / "report.json")]

    def test_arguments_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            args = bench.arguments(self.argv(folder))
            self.assertEqual(args.variants, list(bench.VARIANTS))
            self.assertEqual((args.effort, args.threads, args.repeats), (9, 8, 3))
            args = bench.arguments(self.argv(folder) + ["--case", "a.png", "--manifest", "m.json",
                                                       "--effort", "7", "--threads", "2", "--repeats", "1"])
            self.assertEqual((args.case, args.manifest), ([Path("a.png")], Path("m.json")))
            self.assertEqual((args.effort, args.threads, args.repeats), (7, 2, 1))
            for flag, value in (("--threads", "0"), ("--repeats", "0"),
                                ("--effort", "11"), ("--timeout", "nan")):
                with self.subTest(flag=flag), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    bench.arguments(self.argv(folder) + [flag, value])
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments([v for v in self.argv(folder) if v != "--synthetic"])
            args.out.write_text("preserve", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv(folder))
            self.assertEqual(args.out.read_text(encoding="utf-8"), "preserve")

    def test_commands_only_add_cross_flag_to_matching_fast_plan(self):
        with TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)))
            for variant in bench.VARIANTS:
                command = bench.command(args, "in.png", "out.jxl", variant)
                base = offsets.command(args, "in.png", "out.jxl", bench.base_variant(variant))
                self.assertEqual(command, base + (["--context-cross-channel"] if variant in bench.PAIRS else []))
                for flag in ("--leaf-offset", "--leaf-offset-samples", "--refine-leaves",
                             "--refine-entropy", "--ma-seed", "--search"):
                    self.assertNotIn(flag, command)
            self.assertNotIn("--fast-context", bench.command(args, "in", "out", "baseline"))
            with self.assertRaises(ValueError):
                bench.command(args, "in", "out", "unknown")

    def test_receipts_include_strict_requested_bool_and_nonnegative_counts(self):
        for variant in bench.VARIANTS:
            bench.validate_receipt(measurement(variant), variant)
        for variant in bench.PAIRS:
            bench.validate_receipt(measurement(variant, selected=3), variant)
        for variant in bench.VARIANTS[1:]:
            for key, value in (("cross_channel", None), ("cross_channel", 1),
                               ("cross_channel", variant not in bench.PAIRS),
                               ("cross_channel_plans", -1), ("cross_channel_plans", True),
                               ("cross_channel_plans", 0.5), ("cross_channel_plans", None),
                               ("predictors", "wrong"), ("mode", "leaf"),
                               ("bias", "adaptive"), ("scan", "scalar"), ("trees", 0)):
                data = measurement(variant)
                data["fast_context"][key] = value
                with self.subTest(variant=variant, key=key, value=value), self.assertRaises(ValueError):
                    bench.validate_receipt(data, variant)
            for key in ("cross_channel", "cross_channel_plans"):
                data = measurement(variant)
                del data["fast_context"][key]
                with self.assertRaises(ValueError):
                    bench.validate_receipt(data, variant)
        for variant in ("fast-basic-off", "fast-full-static"):
            with self.assertRaises(ValueError):
                bench.validate_receipt(measurement(variant, selected=1), variant)
        for key, value in (("refinement", {}), ("ma_seed", {}), ("fast_context", None)):
            with self.assertRaises(ValueError):
                bench.validate_receipt(dict(measurement("fast-basic-cross"), **{key: value}), "fast-basic-cross")
        with self.assertRaises(ValueError):
            bench.validate_receipt(measurement("fast-basic-off"), "baseline")

    def test_actual_bytes_bpp_native_time_and_pairing_not_growth(self):
        for changes in (dict(bytes=99), dict(seconds=0), dict(bpp=1),
                        dict(mpix_per_second=1), dict(search=True)):
            with self.assertRaises(ValueError):
                bench.validate_measurement(dict(measurement("baseline"), **changes),
                                           "baseline", "auto", 100, 67 * 51)
        variants = {v: [dict(repetition=r, measurement=dict(bytes=1000 if v in bench.PAIRS else 50))
                        for r in (0, 1)] for v in bench.VARIANTS}
        bench.validate_sizes(variants)
        variants["fast-basic-cross"].reverse()
        bench.validate_sizes(variants)
        variants["fast-basic-cross"][0]["repetition"] = 0
        with self.assertRaisesRegex(ValueError, "paired repetitions"):
            bench.validate_sizes(variants)
        variants["fast-basic-cross"] = []
        with self.assertRaisesRegex(ValueError, "paired repetitions"):
            bench.validate_sizes(variants)

    def test_hooks_are_private(self):
        runner = bench.make_runner()
        self.assertIs(runner.command, bench.command)
        self.assertIs(runner.validate_receipt, bench.validate_receipt)
        self.assertIs(runner.validate_sizes, bench.validate_sizes)
        self.assertIsNot(refinement.command, bench.command)
        self.assertIsNot(offsets.make_runner().validate_receipt, bench.validate_receipt)

    def test_mocked_original_pixels_rotation_ratios_and_failure_reports(self):
        for depth in (8, 16):
            for failure in (None, "pixels", "receipt", "process", "identity"):
                with self.subTest(depth=depth, failure=failure), TemporaryDirectory() as directory:
                    folder = Path(directory)
                    source = folder / "source.png"
                    original = b"original-hidden-RGB"
                    source.write_bytes(original)
                    for name in ("encoder", "djxl"):
                        (folder / name).write_bytes(name.encode("ascii"))
                    args = bench.arguments(self.argv(folder) + ["--case", str(source), "--repeats", "2"])
                    args.synthetic = False
                    runner = bench.make_runner()
                    expected = SimpleNamespace(width=67, height=51, depth=depth, data=original)
                    calls = []

                    def process(command, env, timeout):
                        if command[0] == str(args.encoder):
                            self.assertEqual(Path(command[1]).read_bytes(), b"metadata-neutral")
                            full = "--context-predictors" in command and command[command.index("--context-predictors") + 1] == "full"
                            cross = "--context-cross-channel" in command
                            variant = ("fast-full-cross" if cross else "fast-full-static") if full else (
                                "fast-basic-cross" if cross else "fast-basic-off" if "--fast-context" in command else "baseline")
                            calls.append(variant)
                            size = 150 if cross else 100 if variant != "baseline" else 50
                            Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                            data = measurement(variant, size, selected=int(cross))
                            if failure == "receipt" and cross:
                                data["fast_context"]["cross_channel"] = False
                            if failure == "identity":
                                args.encoder.write_bytes(b"changed binary")
                            return dict(returncode=int(failure == "process"), stdout=json.dumps(data),
                                        stderr="", wall_seconds=0.2)
                        self.assertEqual(command[0], str(args.decoder))
                        Path(command[2]).write_bytes(b"wrong" if failure == "pixels" else original)
                        return dict(returncode=0, stdout="", stderr="", wall_seconds=0.3)

                    def compare(left, right):
                        self.assertIs(left, expected)
                        if left.data != right.data:
                            raise ValueError("Original samples differ including hidden RGB")

                    with (patch.object(runner, "run_process", side_effect=process),
                          patch.object(runner, "prepare_png", return_value=(b"metadata-neutral", expected, "mock-original")),
                          patch.object(runner, "png_samples", side_effect=lambda data: SimpleNamespace(data=data)),
                          patch.object(runner, "compare_pixels", side_effect=compare) as oracle,
                          patch.object(bench, "make_runner", return_value=runner), redirect_stdout(StringIO())):
                        report = bench.benchmark(args)
                    saved = json.loads(args.out.read_text(encoding="utf-8"))
                    self.assertEqual(report["status"], "passed" if failure is None else "failed", report["errors"])
                    self.assertEqual(saved["status"], report["status"])
                    if failure is None:
                        self.assertEqual(calls, trial_order(list(bench.VARIANTS), 0, 0)
                                         + trial_order(list(bench.VARIANTS), 1, 0))
                        self.assertEqual(oracle.call_count, 10)
                        self.assertEqual(report["cases"][0]["depth"], depth)
                        for cross, base in bench.PAIRS.items():
                            summary = saved["cases"][0]["summaries"][cross]
                            self.assertEqual(summary["paired_baseline"], base)
                            self.assertEqual(summary["ratios_to_fast_baseline"]["bytes"], 1.5)
                            self.assertEqual(summary["ratios_to_baseline"]["bytes"], 3)
                            self.assertEqual(summary["seconds"], 0.1)
                    else:
                        self.assertTrue(saved["errors"])


if __name__ == "__main__":
    unittest.main()
