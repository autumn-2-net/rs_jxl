from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_refinement import (
    OFFSET_MODES, VARIANTS, arguments, benchmark, command, decoder_summary,
    fixtures, validate_receipt, validate_sizes,
)
from compare_cpp_port import png_samples


class RefinementHarnessTests(unittest.TestCase):
    @staticmethod
    def receipt(variant="combined", **changes):
        receipt = dict(entropy=variant in ("entropy", "combined"),
                       leaves=variant in ("leaves", "combined"), domains=2, models=6,
                       refit_proposals=0, refit_wins=0, entropy_wins=0,
                       saved_bits=0, refit_seconds=0.1, model_seconds=0.1, write_seconds=0.1,
                       offset_mode=variant.removeprefix("offset-") if variant.startswith("offset-") else None,
                       offset_proposals=0, offset_wins=0, offset_leaves=0, offset_samples=0,
                       offset_population=0, offset_candidates=0, offset_seconds=0.0)
        receipt.update(changes)
        return receipt

    def test_generated_fixtures_preserve_full_depth_and_hidden_colors(self):
        if any(importlib.util.find_spec(name) is None for name in ("cv2", "numpy")):
            self.skipTest("PNG oracle requires numpy and opencv-python")
        with TemporaryDirectory() as directory:
            paths = fixtures(Path(directory))
            self.assertEqual(len(paths), 8)
            for path in paths:
                pixels = png_samples(path.read_bytes())
                self.assertEqual((pixels.width, pixels.height), (67, 51))
                self.assertEqual(pixels.depth, 16 if "16bit" in path.name else 8)

    def test_switches_are_independent_and_baseline_unchanged(self):
        args = SimpleNamespace(encoder=Path("new"), frozen=Path("old"),
                               effort=9, threads=8, buffering=None, leaf_offset_samples=37)
        for variant in VARIANTS:
            cmd = command(args, "input.png", "out.jxl", variant)
            self.assertEqual("--refine-entropy" in cmd, variant in ("entropy", "combined"))
            self.assertEqual("--refine-leaves" in cmd, variant in ("leaves", "combined"))
            self.assertEqual(cmd[0], "old" if variant == "frozen" else "new")
            if variant.startswith("offset-"):
                self.assertEqual(cmd[-4:], ["--leaf-offset", variant.removeprefix("offset-"),
                                           "--leaf-offset-samples", "37"])
            else:
                self.assertNotIn("--leaf-offset", cmd)
                self.assertNotIn("--leaf-offset-samples", cmd)

    def test_arguments_default_and_explicit_fitting_capacity(self):
        with TemporaryDirectory() as directory:
            argv = ["--encoder", "encoder_lab", "--decoder", "djxl", "--synthetic",
                    "--out", str(Path(directory) / "report.json")]
            defaults = arguments(argv)
            self.assertEqual(defaults.leaf_offset_samples, 1024)
            self.assertEqual(defaults.variants, ["baseline", "entropy", "leaves", "combined"])
            args = arguments(argv + ["--variants", "offset-entropy", "baseline",
                                     "offset-mean", "offset-median", "--leaf-offset-samples", "17"])
            self.assertEqual(args.leaf_offset_samples, 17)
            for invalid in ("0", "-1", "1.5"):
                with self.subTest(invalid=invalid), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    arguments(argv + ["--leaf-offset-samples", invalid])

    def test_receipts_distinguish_requested_and_winning_work(self):
        receipt = self.receipt(
                       refit_proposals=1, refit_wins=0, entropy_wins=1,
                       saved_bits=8)
        validate_receipt({"refinement": receipt}, "combined")
        validate_receipt({"refinement": None}, "baseline")
        for field, invalid in (("models", 0), ("refit_wins", 2), ("refit_proposals", 3),
                               ("saved_bits", -1), ("write_seconds", float("nan"))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_receipt({"refinement": dict(receipt, **{field: invalid})}, "combined")
        with self.assertRaises(ValueError):
            validate_receipt({"refinement": receipt}, "baseline")
        with self.assertRaises(ValueError):
            validate_receipt({}, "leaves")

    def test_active_offset_receipts_allow_no_improvement_or_no_eligible_leaves(self):
        for mode in OFFSET_MODES:
            variant = f"offset-{mode}"
            for changes in ({}, dict(offset_population=10000, offset_samples=1024,
                                     offset_candidates=3, offset_leaves=1, offset_seconds=0.1),
                            dict(offset_proposals=1, offset_wins=0),
                            dict(offset_proposals=1, offset_wins=1, saved_bits=8)):
                with self.subTest(mode=mode, changes=changes):
                    validate_receipt({"refinement": self.receipt(variant, **changes)}, variant)

    def test_invalid_offset_receipts(self):
        receipt = self.receipt("offset-mean", offset_population=5000, offset_samples=1024,
                               offset_proposals=1, offset_wins=1, offset_leaves=2,
                               offset_candidates=4, offset_seconds=0.1)
        changes = [("offset_mode", None), ("offset_mode", "median"), ("leaves", True),
                   ("entropy", True), ("offset_proposals", 3), ("offset_wins", 2),
                   ("offset_samples", 5001), ("offset_seconds", float("inf")),
                   ("offset_seconds", True), ("offset_seconds", -1)]
        for key in ("offset_proposals", "offset_wins", "offset_leaves", "offset_samples",
                    "offset_population", "offset_candidates"):
            changes.extend([(key, -1), (key, True), (key, 1.5)])
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_receipt({"refinement": dict(receipt, **{key: value})}, "offset-mean")
        for key in receipt:
            if key.startswith("offset_"):
                missing = dict(receipt)
                del missing[key]
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    validate_receipt({"refinement": missing}, "offset-mean")
        for changes in (dict(offset_mode="mean"), dict(offset_samples=1, offset_population=1),
                        dict(offset_seconds=0.1)):
            with self.subTest(unrequested=changes), self.assertRaises(ValueError):
                validate_receipt({"refinement": self.receipt("leaves", **changes)}, "leaves")

    def test_size_guard_pairs_repetitions_and_does_not_trust_saved_bits(self):
        def sample(repeat, size):
            return dict(repetition=repeat, measurement=dict(bytes=size))
        variants = dict(baseline=[sample(-1, 110), sample(0, 100), sample(1, 120)],
                        frozen=[sample(0, 200)],
                        **{"offset-mean": [sample(-1, 110), sample(0, 99), sample(1, 119)]})
        validate_sizes(variants)
        variants["offset-mean"][1]["measurement"]["bytes"] = 101
        with self.assertRaisesRegex(ValueError, "Actual output size regressed"):
            validate_sizes(variants)

    def test_decoder_wall_summary_excludes_warmup_and_unverified_samples(self):
        def sample(seconds, warmup=False, verified=True):
            return dict(warmup=warmup, verified=verified, decoder=dict(wall_seconds=seconds))
        summary = decoder_summary([sample(99, warmup=True), sample(55, verified=False),
                                   sample(0.2), sample(0.4)])
        self.assertAlmostEqual(summary["decoder_wall_seconds"], 0.3)
        self.assertEqual(summary["min_decoder_wall_seconds"], 0.2)
        self.assertEqual(summary["max_decoder_wall_seconds"], 0.4)
        for invalid in (0, -1, True, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                decoder_summary([sample(invalid)])

    def test_mocked_trials_check_actual_size_and_official_decoder_pixels(self):
        for failure in (None, "growth", "size-report", "pixels"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                folder = Path(directory)
                source = fixtures(folder)[-1]
                original = source.read_bytes()
                changed = fixtures(folder)[0].read_bytes()
                encoder, decoder = folder / "encoder_lab", folder / "djxl"
                encoder.write_bytes(b"fake encoder identity")
                decoder.write_bytes(b"fake decoder identity")
                args = arguments(["--encoder", str(encoder), "--decoder", str(decoder),
                                  "--case", str(source), "--out", str(folder / "report.json"),
                                  "--variants", "offset-mean", "baseline", "--repeats", "1"])
                last_variant = None

                def process(cmd, env, timeout):
                    nonlocal last_variant
                    if cmd[0] == str(encoder):
                        last_variant = "offset-mean" if "--leaf-offset" in cmd else "baseline"
                        size = 101 if failure == "growth" and last_variant != "baseline" else 100
                        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"x" * size)
                        reported = 99 if failure == "size-report" else size
                        pixels = 67 * 51
                        measurement = dict(bytes=reported, seconds=0.1, bpp=reported * 8 / pixels,
                                           mpix_per_second=pixels / 1e6 / 0.1, search=False,
                                           model_costs=dict(tree_bits=1, entropy_header_bits=1, total_bits=2),
                                           fast_context=None,
                                           refinement=None if last_variant == "baseline" else self.receipt(last_variant))
                        return dict(returncode=0, stdout=json.dumps(measurement), stderr="", wall_seconds=0.2)
                    self.assertEqual(cmd[0], str(decoder))
                    Path(cmd[2]).write_bytes(changed if failure == "pixels" else original)
                    return dict(returncode=0, stdout="", stderr="", wall_seconds=0.3)

                def read_pixels(data):
                    return SimpleNamespace(width=67, height=51, depth=16, data=data)

                def compare(expected, actual):
                    if expected.data != actual.data:
                        raise ValueError("Pixel mismatch (mock oracle)")

                with (patch("benchmark_refinement.run_process", side_effect=process),
                      patch("benchmark_refinement.prepare_png", side_effect=lambda source, data, scratch: (data, read_pixels(data), "mock-original")),
                      patch("benchmark_refinement.png_samples", side_effect=read_pixels),
                      patch("benchmark_refinement.compare_pixels", side_effect=compare) as pixel_check,
                      redirect_stdout(StringIO())):
                    report = benchmark(args)
                self.assertEqual(report["status"], "passed" if failure is None else "failed", report["errors"])
                self.assertGreater(pixel_check.call_count, 0)
                if failure is None:
                    summary = report["cases"][0]["summaries"]["offset-mean"]
                    self.assertEqual(summary["decoder_wall_seconds"], 0.3)
                    self.assertEqual(summary["ratios_to_baseline"]["decoder_wall_seconds"], 1)
                    self.assertIn("input/output", report["decoder_timing"])
                else:
                    self.assertTrue(report["errors"])
                    if failure == "growth":
                        self.assertIn("Actual output size regressed", report["errors"][0])
                    elif failure == "size-report":
                        self.assertIn("actual nonempty codestream", report["errors"][0])
                    elif failure == "pixels":
                        self.assertIn("Pixel mismatch", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
