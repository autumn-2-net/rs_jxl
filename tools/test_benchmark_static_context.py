"""Focused receipts and harness tests; native processes mocked, pixel oracle real."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_static_context as bench
import plot_patch_tradeoffs as plot
from test_compare_cpp_port import fixture, make_png
from test_corpus_inputs import png as packed_png


def measurement(mode, size=20, seconds=1.0, costs=False):
    result = dict(bytes=size, seconds=seconds, bpp=size * 4,
                  mpix_per_second=2 / 1e6 / seconds, width=2, height=1,
                  mode=mode, effort=9, threads=8, previous_channels=0,
                  patches="Off", patches_cleared=True, search=False, native_encode_calls=1,
                  timing=bench.NATIVE_TIMING, model_cost_contract=bench.MODEL_COST_CONTRACT,
                  model_costs=dict(tree_bits=3, entropy_header_bits=4, total_bits=7) if costs else None,
                  fast_context=None, seed=None, seed_options=None)
    if mode in bench.SEED_MODES:
        result["seed_options"] = bench.seed_parameters(mode)
        result["seed"] = dict.fromkeys(bench.SEED_COUNTERS, 0)
        result["seed"].update(learners=1, prepared_records=6, seed_splits=24, refined_leaves=25,
                              repaired_branches=int(mode.endswith("repair")), histogram_reuses=8)
    elif mode != "baseline":
        result["fast_context"] = dict(mode=mode, bias="off", scan="auto", predictors="basic",
                                      cross_channel=False, **dict.fromkeys(bench.COUNTERS, 0))
        result["fast_context"].update(trees=1, leaves=2, pixels=6,
                                      horizontal=int(mode == "auto12"), joint=int(mode != "auto12"))
    return result


class ArgumentsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.argv = ["--encoder", "lab.exe", "--decoder", "djxl.exe", "--synthetic",
                     "--out", str(self.root / "report.json")]

    def test_defaults_and_hard_bounds(self):
        args = bench.arguments(self.argv)
        self.assertEqual((args.repeats, args.warmups, args.max_cases), (2, 0, 32))
        self.assertEqual(args.modes, ["baseline", "auto12", "activity25", "activity81"])
        self.assertFalse(args.model_costs)
        for flags in (["--effort", "3"], ["--effort", "11"], ["--threads", "0"],
                      ["--threads", "257"], ["--max-cases", "33"], ["--max-cases", "0"],
                      ["--max-pixels", "0"], ["--repeats", "0"], ["--repeats", "101"],
                      ["--warmups", "-1"], ["--warmups", "101"], ["--timeout", "nan"],
                      ["--timeout", "inf"], ["--modes", "activity25"],
                      ["--modes", "baseline", "baseline"], ["--modes", "baseline", "greedy"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv + flags)
        for effort in (4, 10):
            for threads in (1, 256):
                args = bench.arguments(self.argv + ["--effort", str(effort), "--threads", str(threads)])
                self.assertEqual((args.effort, args.threads), (effort, threads))
        for mode in bench.SEED_MODES:
            args = bench.arguments(self.argv + ["--modes", "baseline", mode])
            self.assertEqual(args.modes, ["baseline", mode])

    def test_only_supported_cli_flags_and_optional_emitted_costs(self):
        for costs in (False, True):
            args = bench.arguments(self.argv + (["--model-costs"] if costs else []))
            for mode in bench.MODES:
                cmd = bench.command(args, "in.png", "out.jxl", mode)
                self.assertEqual(cmd, ["lab.exe", "in.png", "--output", "out.jxl", "--effort", "9",
                                       "--threads", "8", "--mode", mode] + (["--model-costs"] if costs else []))
        with self.assertRaises(ValueError):
            bench.command(args, "i", "o", "guarded")

    def test_relative_manifest_and_duplicate_cases(self):
        source, manifest = self.root / "input.png", self.root / "manifest.json"
        source.write_bytes(fixture(8))
        manifest.write_text(json.dumps(dict(cases=[dict(original="input.png")])), encoding="utf-8")
        args = bench.arguments(self.argv + ["--manifest", str(manifest)])
        self.assertEqual(bench.load_cases(args), [source.resolve()])
        args.case = [source]
        with self.assertRaises(ValueError):
            bench.load_cases(args)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.args = bench.arguments(["--encoder", "lab.exe", "--decoder", "djxl.exe", "--synthetic",
                                     "--out", str(Path(temporary.name) / "report.json")])

    def check(self, m, mode="activity25", wall=999):
        bench.validate_measurement(m, mode, 20, 2, 1, self.args, wall)

    def test_modes_default_options_joint_and_zero_work_shortcuts(self):
        for mode in bench.MODES:
            m = measurement(mode)
            self.check(m, mode)
            if mode in bench.SEED_MODES:
                m["seed"].update(dict.fromkeys(bench.SEED_COUNTERS, 0))
                self.check(m, mode)
            elif mode != "baseline":
                m["fast_context"].update(dict.fromkeys(bench.COUNTERS, 0))
                self.check(m, mode)
        m = measurement("baseline")
        m["fast_context"] = measurement("auto12")["fast_context"]
        with self.assertRaises(ValueError):
            self.check(m, "baseline")

    def test_invalid_actual_metrics_common_policy_and_single_encode(self):
        for changes in (dict(bytes=True), dict(bytes=19), dict(seconds=0), dict(seconds=True),
                        dict(seconds=float("nan")), dict(seconds=float("inf")), dict(seconds=1001),
                        dict(width=2.0), dict(height=2), dict(bpp=1), dict(mpix_per_second=1),
                        dict(mode="auto12"), dict(effort=10), dict(threads=True), dict(previous_channels=1),
                        dict(patches="Default"), dict(patches_cleared=False), dict(search=True),
                        dict(native_encode_calls=2), dict(native_encode_calls=True),
                        dict(timing="encode only"), dict(fast_context=None),
                        dict(model_cost_contract="estimated=wire"), dict(model_costs={})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check(measurement("activity25") | changes)
        for missing in ("fast_context", "model_costs", "native_encode_calls", "timing"):
            m = measurement("activity25")
            m.pop(missing)
            with self.assertRaises(ValueError):
                self.check(m)
        for text in ('{"seconds":NaN}', '{"seconds":Infinity}', '{"seconds":1e999}'):
            with self.assertRaises(ValueError):
                bench.native_json(text)

    def test_invalid_fast_context_defaults_and_counters(self):
        for changes in (dict(mode="activity81"), dict(bias="adaptive"), dict(scan="scalar"),
                        dict(predictors="full"), dict(cross_channel=True), dict(joint=True),
                        dict(joint=-1), dict(trees=3), dict(offset_leaves=1),
                        dict(cross_channel_plans=1), dict(pixels=1.0)):
            m = measurement("activity25")
            m["fast_context"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check(m)
        m = measurement("auto12")
        m["fast_context"]["joint"] = 1
        with self.assertRaises(ValueError):
            self.check(m, "auto12")
        m = measurement("activity81")
        del m["fast_context"]["joint"]
        with self.assertRaises(ValueError):
            self.check(m, "activity81")

    def test_emitted_cost_receipts_are_optional_actual_bits(self):
        self.args.model_costs = True
        self.check(measurement("activity25", costs=True))
        for costs in (None, dict(tree_bits=True, entropy_header_bits=4, total_bits=5),
                      dict(tree_bits=3, entropy_header_bits=4, total_bits=8),
                      dict(tree_bits=160, entropy_header_bits=1, total_bits=161),
                      dict(tree_bits=-1, entropy_header_bits=4, total_bits=3)):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                self.check(measurement("activity25") | dict(model_costs=costs))

    def test_seed_settings_receipts_and_no_restricted_fast_learner(self):
        for mode in bench.SEED_MODES:
            m = measurement(mode)
            self.check(m, mode)
            self.assertEqual(m["seed_options"]["topology"], mode.split("-")[0])
            self.assertEqual(m["seed_options"]["mode"], mode.split("-")[1])
            # Activity-grid statistics must not be capped by quantile depth=3.
            m["seed"].update(seed_splits=80, refined_leaves=81)
            self.check(m, mode)
            for changes in (dict(seed=None), dict(seed_options=None), dict(native_encode_calls=2),
                            dict(effort=8), dict(previous_channels=1), dict(patches_cleared=False),
                            dict(fast_context=measurement("activity81")["fast_context"])):
                with self.subTest(mode=mode, changes=changes), self.assertRaises(ValueError):
                    self.check(measurement(mode) | changes, mode)
            for key in ("seed", "seed_options", "fast_context"):
                m = measurement(mode)
                del m[key]
                with self.subTest(mode=mode, missing=key), self.assertRaises(ValueError):
                    self.check(m, mode)
            for changes in (dict(topology="quantile"), dict(mode="only"), dict(depth=True), dict(depth=3.0), dict(depth=4)):
                m = measurement(mode)
                m["seed_options"].update(changes)
                with self.subTest(mode=mode, changes=changes), self.assertRaises(ValueError):
                    self.check(m, mode)
            for key in bench.SEED_COUNTERS:
                for bad in (True, -1, 1.0, None):
                    m = measurement(mode)
                    m["seed"][key] = bad
                    with self.subTest(mode=mode, key=key, bad=bad), self.assertRaises(ValueError):
                        self.check(m, mode)
                del m["seed"][key]
                with self.assertRaises(ValueError):
                    self.check(m, mode)

    def test_static_receipts_remain_compatible_but_reject_seed_activation(self):
        for mode in bench.DEFAULT_MODES:
            m = measurement(mode)
            m.pop("seed")
            m.pop("seed_options")
            self.check(m, mode)
            for changes in (dict(seed={}), dict(seed_options=bench.seed_parameters("activity81-refine"))):
                with self.assertRaises(ValueError):
                    self.check(m | changes, mode)


class HarnessTests(unittest.TestCase):
    def exercise(self, depth=16, failure=None, extra=(), input_png=None, decoded_png=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source, encoder, decoder = (root / name for name in ("input.png", "lab.exe", "djxl.exe"))
        original = fixture(depth) if input_png is None else input_png
        source.write_bytes(original)
        encoder.write_bytes(b"frozen encoder")
        decoder.write_bytes(b"frozen official decoder")
        out = root / "report.json"
        argv = ["--encoder", str(encoder), "--decoder", str(decoder), "--case", str(source),
                "--warmups", "1", "--out", str(out), *extra]
        calls, visits = [], []

        def run(cmd, env, timeout):
            calls.append(cmd)
            self.assertNotIn("RS_JXL_PROFILE", env)
            self.assertNotIn("JXL_TEST", env)
            if cmd[0] == str(encoder):
                self.assertEqual(Path(cmd[1]).read_bytes(), original)
                mode = cmd[cmd.index("--mode") + 1]
                visits.append(mode)
                size = {"baseline": 20, "auto12": 30, "activity25": 25, "activity81": 18,
                        "activity25-refine": 24, "activity81-refine": 22, "activity81-repair": 21}[mode]
                Path(cmd[cmd.index("--output") + 1]).write_bytes(b"x" * size)
                m = measurement(mode, size, float(len(visits)), "--model-costs" in cmd)
                if failure == "receipt":
                    m.pop("fast_context")
                if failure == "nan":
                    m["seconds"] = float("nan")
                if failure == "wall":
                    m["seconds"] = 1000
                if failure in ("encoder_changed", "decoder_changed", "input_changed"):
                    dict(encoder_changed=encoder, decoder_changed=decoder, input_changed=source)[failure].write_bytes(b"changed")
                return dict(returncode=int(failure == "encoder"), stdout=json.dumps(m), stderr="", wall_seconds=999)
            self.assertEqual(cmd[0], str(decoder))
            self.assertIn(f"--bits_per_sample={depth}", cmd)
            decoded = fixture(depth) if decoded_png is None else decoded_png
            if failure == "hidden":
                samples = ([259 ^ 1, 513, 65534, 0, 1, 257, 32769, 65535] if depth == 16
                           else [19 ^ 1, 81, 253, 0, 7, 11, 13, 255])
                decoded = make_png(depth, 6, [samples])
            if failure == "narrowed":
                decoded = fixture(8)
            Path(cmd[2]).write_bytes(decoded)
            return dict(returncode=int(failure == "decoder"), stdout="", stderr="", wall_seconds=999)

        with patch.object(bench.port, "run_process", side_effect=run), \
                patch.dict(os.environ, {"RS_JXL_PROFILE": "1", "JXL_TEST": "1"}), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = bench.main(argv)
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, int(report["status"] != "passed"))
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual({p.name for p in root.iterdir()}, {"input.png", "lab.exe", "djxl.exe", "report.json"})
        return report, calls, visits

    def test_rotated_repeats_native_medians_regressions_and_plot_schema(self):
        for depth in (8, 16):
            report, calls, visits = self.exercise(depth, extra=("--model-costs",))
            self.assertEqual(report["status"], "passed", report["errors"])
            self.assertEqual(report["experiment"], "static_activity_context")
            self.assertEqual(visits, [mode for repeat in (-1, 0, 1)
                                      for mode in bench.trial_order(list(bench.DEFAULT_MODES), repeat, 0)])
            self.assertEqual(len(calls), 24)
            corpus, points = plot.validate_report(report)
            self.assertEqual((len(corpus), len(points)), (4, 4))
            summaries = report["cases"][0]["summaries"]
            self.assertEqual(summaries["baseline"]["seconds"], 7.5)
            self.assertTrue(summaries["auto12"]["regressions"]["bytes"])
            self.assertEqual(summaries["activity81"]["bytes"], 18)
            for name in ("encoder", "decoder"):
                self.assertEqual(len(report[name]["sha256"]), 64)

    def test_opt_in_seed_modes_are_independent_verified_plot_points(self):
        modes = ["baseline", *bench.SEED_MODES]
        report, calls, visits = self.exercise(extra=("--warmups", "0", "--repeats", "1", "--modes", *modes))
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(visits, modes)
        self.assertEqual(len(calls), 8)
        corpus, points = plot.validate_report(report)
        self.assertEqual((len(corpus), len(points)), (4, 4))
        for mode in bench.SEED_MODES:
            sample = report["cases"][0]["variants"][mode][0]
            self.assertTrue(sample["verified"])
            self.assertIsNone(sample["measurement"]["fast_context"])
            self.assertEqual(sample["measurement"]["seed_options"], bench.seed_parameters(mode))

    def test_pixel_failures_bad_receipts_process_failures_and_frozen_inputs(self):
        for depth in (8, 16):
            for failure in ("hidden", "receipt", "nan", "wall", "encoder", "decoder",
                            "encoder_changed", "decoder_changed", "input_changed"):
                with self.subTest(depth=depth, failure=failure):
                    report, calls, _ = self.exercise(depth, failure)
                    self.assertEqual(report["status"], "failed")
                    self.assertTrue(report["errors"])
                    if failure in ("receipt", "nan", "wall"):
                        self.assertEqual(len(calls), 2)  # Oracle still checks malformed receipts' outputs.
                        self.assertFalse(any(s["verified"] for v in report["cases"][0]["variants"].values() for s in v))
                    if failure == "hidden":
                        self.assertIn("hidden RGB", report["errors"][0])
                    if failure.endswith("changed"):
                        self.assertIn("changed during measurement", report["errors"][-1])
        report, _, _ = self.exercise(16, "narrowed")
        self.assertEqual(report["status"], "failed")

    def test_packed_grayscale_palette_kept_native_with_lossless_oracle_expansion(self):
        extra = ("--warmups", "0", "--repeats", "1", "--modes", "baseline", "activity25")
        for depth in (1, 2, 4):
            for color in (0, 3):
                with self.subTest(depth=depth, color=color):
                    if color == 0:
                        packed = packed_png(depth, color, [[1, 0]], 2, [(b"tRNS", b"\0\1")])
                        scaled = 255 // ((1 << depth) - 1)
                        expanded = make_png(8, 6, [[scaled, scaled, scaled, 0, 0, 0, 0, 255]])
                    else:
                        packed = packed_png(depth, color, [[0, 1]], 2,
                                            [(b"PLTE", bytes([19, 81, 253, 7, 11, 13])),
                                             (b"tRNS", b"\0")])
                        expanded = fixture(8)
                    report, calls, visits = self.exercise(8, extra=extra,
                                                         input_png=packed, decoded_png=expanded)
                    self.assertEqual(report["status"], "passed", report["errors"])
                    self.assertEqual((len(calls), visits), (4, ["baseline", "activity25"]))
                    case = report["cases"][0]
                    self.assertEqual((case["width"], case["height"], case["depth"]), (2, 1, 8))
                    self.assertEqual(case["preparation"], "original-packed-png; oracle-expanded-to-8bit")
                    self.assertEqual(case["snapshot_sha256"], bench.port.sha256(packed))
                    self.assertIn("packed grayscale/palette", report["correctness"])
                    self.assertIn("16-bit preserved", report["correctness"])
                    plot.validate_report(report)
                    if color == 3:
                        failed, _, _ = self.exercise(8, "hidden", extra=extra,
                                                     input_png=packed, decoded_png=expanded)
                        self.assertEqual(failed["status"], "failed")
                        self.assertIn("hidden RGB", failed["errors"][0])

    def test_admission_limits_before_native_calls(self):
        for extra in (("--max-pixels", "1"), ("--synthetic", "--max-cases", "1")):
            report, calls, _ = self.exercise(extra=extra)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(calls, [])

    def test_no_overwrite_and_missing_binary_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "report.json"
            argv = ["--encoder", str(Path(directory) / "missing.exe"), "--decoder", "djxl.exe",
                    "--synthetic", "--out", str(out)]
            with redirect_stderr(StringIO()), patch.object(bench.port, "run_process") as run:
                self.assertEqual(bench.main(argv), 1)
                run.assert_not_called()
                original = out.read_bytes()
                with self.assertRaises(SystemExit):
                    bench.arguments(argv)
                self.assertEqual(out.read_bytes(), original)
                args = bench.arguments(argv[:-1] + [str(out.with_suffix(".new"))])
                args.out = out
                with self.assertRaises(FileExistsError):
                    bench.benchmark(args)
                self.assertEqual(out.read_bytes(), original)


class FixtureTests(unittest.TestCase):
    def test_reuses_six_original_rgb_rgba_controls_with_hidden_wide_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = bench.fixtures(Path(directory) / "synthetic")
            self.assertEqual(len(paths), 6)
            formats = set()
            for path in paths:
                _, info = bench.png_framing(path.read_bytes())
                formats.add((info["bit_depth"], info["color_type"]))
                self.assertLessEqual(info["width"] * info["height"], 1048576)
                if info["color_type"] == 6:
                    pixels = bench.port.png_samples(path.read_bytes())
                    hidden = pixels.rgba[:, :, 3] == 0
                    self.assertTrue(hidden.any())
                    self.assertTrue((pixels.rgba[:, :, :3][hidden] != 0).any())
                    if pixels.depth == 16:
                        self.assertTrue((pixels.rgba[:, :, :3][hidden] > 255).any())
                        self.assertTrue((pixels.rgba[:, :, :3] % 257 != 0).any())
            self.assertEqual(formats, {(8, 2), (8, 6), (16, 2), (16, 6)})


if __name__ == "__main__":
    unittest.main()
