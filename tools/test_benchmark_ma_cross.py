"""Ordinary MA adapter tests; all native encoder/decoder calls are mocked."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_ma_cross as bench
import benchmark_fast_offsets as offsets
import benchmark_refinement as refinement
from benchmark_ma_models import trial_order


def measurement(variant, size=100, pixels=67 * 51, actual=None):
    requested = bench.previous_channels(variant)
    return dict(bytes=size, seconds=0.1, bpp=size * 8 / pixels,
                mpix_per_second=pixels / 1e6 / 0.1, search=False,
                model_costs=dict(tree_bits=1, entropy_header_bits=1, total_bits=2),
                fast_context=None, refinement=None, ma_seed=None,
                max_previous_channels=requested,
                ma_previous_channels=requested if actual is None else actual)


class MaCrossTests(unittest.TestCase):
    def argv(self, folder):
        return ["--encoder", str(folder / "encoder"), "--decoder", str(folder / "djxl"),
                "--synthetic", "--out", str(folder / "report.json")]

    def test_arguments_reuse_defaults_inputs_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            args = bench.arguments(self.argv(folder))
            self.assertEqual(args.variants, ["baseline", "ma-prev1", "ma-prev2"])
            self.assertEqual((args.effort, args.threads, args.repeats, args.buffering), (9, 8, 3, None))
            self.assertEqual(vars(args), dict(vars(offsets.arguments(self.argv(folder))), variants=list(bench.VARIANTS)))
            custom = bench.arguments(self.argv(folder) + ["--case", "a.png", "--manifest", "m.json",
                                                         "--threads", "2", "--repeats", "1"])
            self.assertEqual((custom.case, custom.manifest, custom.threads, custom.repeats),
                             ([Path("a.png")], Path("m.json"), 2, 1))
            for flag, value in (("--threads", "0"), ("--repeats", "0"), ("--timeout", "nan")):
                with self.subTest(flag=flag), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    bench.arguments(self.argv(folder) + [flag, value])
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments([v for v in self.argv(folder) if v != "--synthetic"])
            args.out.write_text("preserve", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv(folder))
            with self.assertRaises(FileExistsError):
                bench.benchmark(args)
            self.assertEqual(args.out.read_text(encoding="utf-8"), "preserve")

    def test_commands_only_add_previous_channels_to_ordinary_backend(self):
        with TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)))
            baseline = refinement.command(args, "in.png", "out.jxl", "baseline")
            for variant in bench.VARIANTS:
                requested = bench.previous_channels(variant)
                command = bench.command(args, "in.png", "out.jxl", variant)
                self.assertEqual(command, baseline + (["--previous-channels", str(requested)] if requested else []))
                for flag in ("--fast-context", "--context-cross-channel", "--leaf-offset", "--buffering",
                             "--rct", "--learning-fraction", "--palette-grouping", "--palette-colors",
                             "--refine-leaves", "--refine-entropy", "--ma-seed", "--search"):
                    self.assertNotIn(flag, command)
            with self.assertRaises(ValueError):
                bench.command(args, "in", "out", "unknown")

    def test_requested_actual_projection_grayscale_and_strict_types(self):
        for variant in bench.VARIANTS:
            for actual in range(bench.previous_channels(variant) + 1):
                bench.validate_receipt(measurement(variant, actual=actual), variant)
            for key in ("max_previous_channels", "ma_previous_channels"):
                for value in (None, True, False, -1, 0.5, "1", 3):
                    with self.subTest(variant=variant, key=key, value=value), self.assertRaises(ValueError):
                        bench.validate_receipt(dict(measurement(variant), **{key: value}), variant)
                data = measurement(variant)
                del data[key]
                with self.assertRaises(ValueError):
                    bench.validate_receipt(data, variant)
        with self.assertRaises(ValueError):
            bench.validate_receipt(measurement("ma-prev2"), "ma-prev1")
        with self.assertRaises(ValueError):
            bench.validate_receipt(measurement("ma-prev1"), "baseline")
        with self.assertRaises(ValueError):
            bench.validate_receipt(measurement("baseline", actual=1), "baseline")

    def test_no_fast_context_offset_seed_or_other_experiments(self):
        for key in ("fast_context", "refinement", "ma_seed", "palette_grouping",
                    "match_extension", "lz_scout", "lz_quick"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_receipt(dict(measurement("ma-prev1"), **{key: {}}), "ma-prev1")
        with self.assertRaises(ValueError):
            bench.validate_receipt(dict(measurement("ma-prev2"), tree_learning="cost"), "ma-prev2")
        for key in ("fast_context", "refinement"):
            data = measurement("baseline")
            del data[key]
            with self.assertRaises(ValueError):
                bench.validate_receipt(data, "baseline")

    def test_actual_measurements_and_matched_repetitions_allow_growth(self):
        for changes in (dict(bytes=99), dict(seconds=0), dict(bpp=1),
                        dict(mpix_per_second=1), dict(search=True)):
            with self.assertRaises(ValueError):
                bench.validate_measurement(dict(measurement("baseline"), **changes), "baseline", "auto", 100, 67 * 51)
        variants = {v: [dict(repetition=r, measurement=dict(bytes=50 if v == "baseline" else 999))
                        for r in (0, 1)] for v in bench.VARIANTS}
        variants["ma-prev2"].reverse()
        bench.validate_sizes(variants)
        variants["baseline"][0]["repetition"] = 1
        with self.assertRaisesRegex(ValueError, "paired repetitions"):
            bench.validate_sizes(variants)
        variants["baseline"][0]["repetition"] = 0
        variants["ma-prev1"].pop()
        with self.assertRaisesRegex(ValueError, "paired repetitions"):
            bench.validate_sizes(variants)

    def test_private_hooks_shared_synthetics_and_main(self):
        runner = bench.make_runner()
        self.assertIs(runner.command, bench.command)
        self.assertIs(runner.validate_receipt, bench.validate_receipt)
        self.assertIs(runner.validate_sizes, bench.validate_sizes)
        self.assertIs(runner.prepare_png, refinement.prepare_png)
        self.assertIs(runner.add_ratios, offsets.fast.add_ratios)
        self.assertEqual(runner.fixtures.__code__.co_code, refinement.fixtures.__code__.co_code)
        self.assertIsNot(refinement.command, bench.command)
        self.assertIsNot(offsets.make_runner().command, bench.command)
        with TemporaryDirectory() as directory:
            for status in ("passed", "failed"):
                with (patch.object(bench, "benchmark", return_value=dict(status=status, errors=[])) as run,
                      redirect_stdout(StringIO())):
                    self.assertEqual(bench.main(self.argv(Path(directory))), int(status != "passed"))
                self.assertEqual(run.call_args.args[0].variants, list(bench.VARIANTS))

    def test_mocked_runner_original_pixels_rotation_projection_and_failures(self):
        for depth in (8, 16):
            for grayscale in (False, True):
                for failure in (None, "pixels", "receipt", "process", "identity"):
                    with self.subTest(depth=depth, grayscale=grayscale, failure=failure), TemporaryDirectory() as directory:
                        folder = Path(directory)
                        source = folder / "source.png"
                        original = b"original-samples-including-hidden-RGB"
                        source.write_bytes(original)
                        for name in ("encoder", "djxl"):
                            (folder / name).write_bytes(name.encode("ascii"))
                        argv = [v for v in self.argv(folder) if v != "--synthetic"]
                        args = bench.arguments(argv + ["--case", str(source), "--repeats", "2"])
                        runner = bench.make_runner()
                        expected = SimpleNamespace(width=67, height=51, depth=depth, data=original)
                        calls = []

                        def process(command, env, timeout):
                            if command[0] == str(args.encoder):
                                self.assertEqual(Path(command[1]).read_bytes(), b"metadata-neutral")
                                count = int(command[command.index("--previous-channels") + 1]) if "--previous-channels" in command else 0
                                variant = bench.VARIANTS[count]
                                calls.append(variant)
                                size = 50 * (count + 1)
                                Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                                data = measurement(variant, size, actual=0 if grayscale else count)
                                if failure == "receipt":
                                    data["ma_previous_channels"] = count + 1
                                if failure == "identity":
                                    args.encoder.write_bytes(b"changed binary")
                                return dict(returncode=int(failure == "process"), stdout=json.dumps(data),
                                            stderr="", wall_seconds=0.2)
                            self.assertEqual(command[0], str(args.decoder))
                            self.assertEqual(command, runner.decoder_command(args.decoder, Path(command[1]), Path(command[2]), depth))
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
                        self.assertEqual(source.read_bytes(), original)
                        if failure is None:
                            self.assertEqual(calls, trial_order(list(bench.VARIANTS), 0, 0) + trial_order(list(bench.VARIANTS), 1, 0))
                            self.assertEqual(oracle.call_count, 6)
                            case = saved["cases"][0]
                            self.assertEqual(case["depth"], depth)
                            self.assertEqual(case["summaries"]["ma-prev2"]["ratios_to_baseline"]["bytes"], 3)
                            receipt = case["variants"]["ma-prev2"][0]["measurement"]
                            self.assertEqual(receipt["max_previous_channels"], 2)
                            self.assertEqual(receipt["ma_previous_channels"], 0 if grayscale else 2)
                        else:
                            self.assertTrue(saved["errors"])


if __name__ == "__main__":
    unittest.main()
