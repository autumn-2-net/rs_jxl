"""Pixel benchmark contract tests; fake processes only, no builds or codecs."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import benchmark_jpeg_pixel_decode as bench
import compare_cpp_port as port


class PixelMetricTests(unittest.TestCase):
    def test_u8_subtraction_is_signed_and_counts_every_channel(self):
        actual = np.array([[[0, 255, 10], [12, 20, 33]],
                           [[40, 50, 60], [70, 80, 90]]], dtype=np.uint8)
        expected = np.array([[[255, 0, 10], [12, 22, 30]],
                             [[40, 50, 60], [70, 80, 90]]], dtype=np.uint8)
        result = bench.differences(actual, expected)
        self.assertFalse(result["exact"])
        self.assertEqual((result["samples"], result["pixels"], result["channels"]), (12, 4, 3))
        self.assertEqual(result["different_samples"], 4)
        self.assertEqual(result["different_pixels"], 2)
        self.assertEqual(result["different_pixel_percent"], 50)
        self.assertEqual(result["max_abs_error"], 255)
        self.assertAlmostEqual(result["mean_abs_error"], 515 / 12)
        self.assertAlmostEqual(result["rmse"], math.sqrt((255 ** 2 * 2 + 4 + 9) / 12))
        self.assertEqual(result["first_difference"],
                         dict(x=0, y=0, channel=0, rust=0.0, official=255.0))

    def test_gray_exactness_and_first_difference_coordinates(self):
        for dtype in (np.uint8, np.float32):
            with self.subTest(dtype=dtype):
                expected = np.zeros((2, 3, 1), dtype=dtype)
                actual = expected.copy()
                exact = bench.differences(actual, expected)
                self.assertTrue(exact["exact"])
                self.assertIsNone(exact["first_difference"])
                actual[1, 2, 0] = 1
                result = bench.differences(actual, expected)
                self.assertEqual(result["first_difference"],
                                 dict(x=2, y=1, channel=0, rust=1.0, official=0.0))
                self.assertEqual(result["different_samples"], 1)
                self.assertEqual(result["different_pixels"], 1)
                self.assertAlmostEqual(result["mean_abs_error"], 1 / 6)

    def test_f32_signed_zero_is_not_bit_exact_but_has_zero_numerical_error(self):
        actual = np.array([[[-0.0, 0.0, 1.0]]], dtype=np.float32)
        expected = np.array([[[0.0, -0.0, 1.0]]], dtype=np.float32)
        result = bench.differences(actual, expected)
        self.assertFalse(result["exact"])
        self.assertEqual(result["different_samples"], 2)
        self.assertEqual(result["different_pixels"], 1)
        for key in ("max_abs_error", "mean_abs_error", "rmse"):
            self.assertEqual(result[key], 0)
        self.assertEqual(math.copysign(1, result["first_difference"]["rust"]), -1)
        self.assertEqual(math.copysign(1, result["first_difference"]["official"]), 1)

    def test_f32_rejects_nonfinite_samples_even_when_both_words_match(self):
        for value in (np.nan, np.inf, -np.inf):
            for side in ("rust", "official", "both"):
                with self.subTest(value=value, side=side):
                    actual = np.zeros((1, 1, 3), dtype=np.float32)
                    expected = actual.copy()
                    if side in ("rust", "both"):
                        actual[0, 0, 2] = value
                    if side in ("official", "both"):
                        expected[0, 0, 2] = value
                    with self.assertRaisesRegex(ValueError, "nonfinite pixel samples: 1"):
                        bench.differences(actual, expected)

    def test_mismatched_shapes_types_and_unsupported_channels_are_rejected(self):
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        for expected in (rgb[:, :1], rgb.astype(np.float32)):
            with self.subTest(shape=expected.shape, dtype=expected.dtype):
                with self.assertRaisesRegex(ValueError, "shape/type mismatch"):
                    bench.differences(rgb, expected)
        for shape in ((1, 2), (1, 2, 2), (1, 2, 4), (1, 1, 2, 3)):
            with self.subTest(shape=shape):
                array = np.zeros(shape, dtype=np.uint8)
                with self.assertRaisesRegex(ValueError, "one gray/RGB frame"):
                    bench.differences(array, array)


class PixelBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.argv = []
        self.binaries = {}
        for option, name in (("rust", "rust"), ("djxl", "official")):
            path = self.root / f"{name}.exe"
            path.write_bytes(f"fake {name}, never executable".encode())
            self.binaries[name] = path
            self.argv += [f"--{option}", str(path)]
        self.cases = []
        for label in ("first", "second"):
            jxl, original = self.root / f"{label}.jxl", self.root / f"{label}.jpg"
            jxl.write_bytes(f"fake JXL {label}".encode())
            original.write_bytes(f"fake JPEG {label}".encode())
            self.cases.append(dict(kind="jpeg", label=label, jxl=str(jxl), original=str(original),
                                   jxl_sha256=port.file_sha256(jxl),
                                   original_sha256=port.file_sha256(original),
                                   height=2, width=2, pixels=4, bpp=jxl.stat().st_size * 2))
        self.baseline = self.root / "baseline.json"
        self.write_baseline()
        self.argv += ["--baseline", str(self.baseline)]
        self.calls, self.counts = [], {}
        self.run_count = 0
        self.change = self.failure = self.pixel_mismatch = None
        self.jpeg_mismatch = False
        # Fail closed if a future harness change bypasses either mocked runner.
        guard = patch("subprocess.run", side_effect=AssertionError("Real subprocess forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def write_baseline(self, *, status="passed", cases=None):
        self.baseline.write_text(json.dumps(dict(status=status,
                                                cases=self.cases if cases is None else cases)),
                                 encoding="utf-8")

    def fake_process(self, command, environment, timeout):
        self.calls.append((list(command), dict(environment), timeout))
        name = Path(command[0]).stem
        phase = ("version" if "--version" in command else
                 "timing" if "--disable_output" in command else "verify")
        result = dict(command=list(command), returncode=0, stdout="", stderr="", wall_seconds=100000)
        if self.failure and (phase, name) == self.failure[:2]:
            result.update(self.failure[2])
            return result
        if phase == "version":
            result["stdout"] = f"fake {name} version"
            return result
        source_index = 1 if name == "official" else 2
        source = command[source_index]
        case = next(case for case in self.cases if case["jxl"] == source)
        if phase == "timing":
            key = (source, name)
            count = self.counts.get(key, 0)
            self.counts[key] = count + 1
            result["wall_seconds"] = (10000, 10, 30, 999)[count] * (2 if name == "rust" else 1)
            if self.change and source == self.cases[-1]["jxl"] and name == "rust" and count == 3:
                self.change.write_bytes(b"changed at the end of timing")
            return result
        output = Path(command[source_index + 1])
        if "--reconstruct_jpeg" in command:
            payload = Path(case["original"]).read_bytes()
            output.write_bytes(payload + (b"mismatch" if self.jpeg_mismatch else b""))
        else:
            floating = "--bits_per_sample=32" in command
            values = np.arange(12).reshape(2, 2, 3).astype("<f4" if floating else np.uint8)
            if name == "rust" and self.pixel_mismatch == ("f32" if floating else "u8"):
                values[0, 0, 2] += 1
            if floating:
                output.write_bytes(b"PF\n2 2\n-1.0\n" + values[::-1].tobytes())
            else:
                output.write_bytes(b"P7\nWIDTH 2\nHEIGHT 2\nDEPTH 3\nMAXVAL 255\n"
                                   b"TUPLTYPE RGB\nENDHDR\n" + values.tobytes())
        return result

    def run_harness(self, extra=(), *, error=None):
        self.output = self.root / f"run-{self.run_count}"
        self.run_count += 1
        self.calls.clear()
        self.counts.clear()
        expected = self.assertRaisesRegex(*error) if error else nullcontext()
        with patch.object(bench, "run_process", side_effect=self.fake_process), \
                patch.object(port, "run_process", side_effect=self.fake_process), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()), expected:
            bench.main([*self.argv, "--out", str(self.output), *extra])
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"] == "failed", error is not None)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual({p.name for p in self.output.iterdir()}, {"report.json", "REPORT.md"})
        self.assertIn(report["status"], (self.output / "REPORT.md").read_text(encoding="utf-8"))
        return report

    def test_timed_commands_decode_pixels_at_default_or_requested_depth(self):
        for extra, bits in (([], 32), (["--bits", "32"], 32)):
            with self.subTest(bits=bits, extra=extra):
                report = self.run_harness(extra)
                self.assertEqual(report["status"], "completed_exact")
                self.assertEqual(report["config"]["bits"], bits)
                self.assertEqual(len(report["trials"]), 2 * (6 + 8))
                for trial in report["trials"]:
                    command = trial["command"]
                    rust = trial["decoder"] == "rust"
                    self.assertEqual("decode" in command, rust)
                    self.assertIn("--num_threads=1", command)
                    self.assertNotIn("--pixels_to_jpeg", command)
                    self.assertEqual("--reconstruct_jpeg" in command, trial["phase"] == "verify_jpeg")
                    if trial["phase"] == "timing":
                        self.assertIn(f"--bits_per_sample={bits}", command)
                        self.assertEqual(trial["bits"], bits)
                        self.assertIn("--disable_output", command)
                        self.assertIn("--num_reps=10", command)
                        self.assertTrue(all(arg.startswith("--") for arg in command[3 if rust else 2:]))
                    else:
                        self.assertNotIn("--disable_output", command)
                        self.assertFalse(any(arg.startswith("--num_reps=") for arg in command))
                        if trial["phase"] == "verify_pixels":
                            self.assertIn(f"--bits_per_sample={trial['bits']}", command)
                self.assertEqual({r["bits"] for r in report["trials"] if r["phase"] == "verify_pixels"},
                                 {8, 32})
                sample_type = "uint8" if bits == 8 else "float32"
                self.assertIn(f"Timed: JXL -> {sample_type} pixels", bench.markdown(report))

    def test_dither_is_forwarded_only_to_rust_pixel_decoding(self):
        for mode in (None, "jxl-rs", "libjxl"):
            with self.subTest(mode=mode):
                report = self.run_harness([] if mode is None else ["--dither", mode])
                self.assertEqual(report["config"]["dither"], mode)
                for trial in report["trials"]:
                    command = trial["command"]
                    expected = (mode is not None and trial["decoder"] == "rust"
                                and trial["phase"] != "verify_jpeg")
                    self.assertEqual("--dither" in command, expected)
                    if expected:
                        self.assertEqual(command[command.index("--dither") + 1], mode)
                self.assertIn(mode or "binary default", bench.markdown(report))

    def test_previous_checkpoint_rotates_with_current_and_official(self):
        before = self.root / "before.exe"
        before.write_bytes(b"fake previous Rust")
        report = self.run_harness(["--before", str(before), "--dither", "libjxl"])
        for index, row in enumerate(report["results"]):
            self.assertEqual(set(row["timing"]), {"rust", "official", "before"})
            self.assertTrue(row["before_u8"]["exact"])
            self.assertTrue(row["before_f32"]["exact"])
            for repeat in (-1, 0, 1, 2):
                timed = [r for r in report["trials"] if r["phase"] == "timing"
                         and r["case"] == index and r["repeat"] == repeat]
                names = ["rust", "official", "before"]
                start = (index + repeat) % 3
                self.assertEqual([r["decoder"] for r in timed], names[start:] + names[:start])
        for trial in report["trials"]:
            if trial["decoder"] == "before":
                self.assertIn("decode", trial["command"])
                self.assertNotIn("--dither", trial["command"])
        self.assertIn("Previous Rust ms", bench.markdown(report))

    def test_official_no_output_cannot_be_labeled_u8_timing(self):
        with self.assertRaisesRegex(ValueError, "selects F32"):
            bench.command("official", "djxl", "input.jxl", 1, bits=8)
        args = bench.command("official", "djxl", "input.jxl", 1,
                             output="verified.pam", bits=8)
        self.assertIn("--bits_per_sample=8", args)
        self.assertNotIn("--disable_output", args)
        legacy = self.run_harness()
        legacy["config"]["bits"] = 8
        self.assertIn("official F32; NOT COMPARABLE", bench.markdown(legacy))

    def test_warmups_are_excluded_medians_are_per_decode_and_order_rotates(self):
        with patch.dict(os.environ, {"JXL_TEST": "1", "RS_JXL_PROFILE": "trace"}):
            report = self.run_harness()
        for index, row in enumerate(report["results"]):
            trials = [r for r in report["trials"] if r["case"] == index]
            self.assertTrue(all(r["phase"] != "timing" for r in trials[:6]))
            for repeat in (-1, 0, 1, 2):
                timed = [r for r in trials[6:] if r["repeat"] == repeat]
                names = ["official", "rust"] if (index + repeat) % 2 else ["rust", "official"]
                self.assertEqual([r["decoder"] for r in timed], names)
                self.assertTrue(all(r["warmup"] == (repeat < 0) for r in timed))
            for name, factor in (("rust", 2), ("official", 1)):
                self.assertEqual(row["timing"][name],
                                 dict(seconds=3 * factor, min_seconds=factor, max_seconds=99.9 * factor,
                                      mpix_s=4 / 1e6 / (3 * factor)))
                self.assertTrue(row["jpeg_bytes_exact"])
        for _, environment, _ in self.calls:
            self.assertNotIn("JXL_TEST", environment)
            self.assertNotIn("RS_JXL_PROFILE", environment)

    def test_zero_warmups_single_repeat_and_custom_reps(self):
        report = self.run_harness(["--warmups", "0", "--repeats", "1", "--decode-reps", "4"])
        timed = [r for r in report["trials"] if r["phase"] == "timing"]
        self.assertEqual(len(timed), 4)
        self.assertTrue(all(not r["warmup"] and r["repeat"] == 0 and r["reps"] == 4 for r in timed))
        self.assertTrue(all("--num_reps=4" in r["command"] for r in timed))
        self.assertEqual(report["results"][0]["timing"]["rust"]["seconds"], 5000)

    def test_repeated_case_labels_filter_in_baseline_order_without_duplicate_runs(self):
        report = self.run_harness(["--case-label", "second"])
        self.assertEqual([row["case"]["label"] for row in report["results"]], ["second"])
        self.assertTrue(all(self.cases[1]["jxl"] in r["command"] for r in report["trials"]))
        report = self.run_harness(["--case-label", "second", "--case-label", "first",
                                   "--case-label", "second"])
        self.assertEqual([row["case"]["label"] for row in report["results"]], ["first", "second"])

    def test_old_report_config_without_bits_still_renders_as_float32(self):
        report = self.run_harness()
        del report["config"]["bits"]
        self.assertIn("Timed: JXL -> float32 pixels", bench.markdown(report))

    def test_pixel_differences_remain_visible_for_both_depths(self):
        for key in ("u8", "f32"):
            with self.subTest(key=key):
                self.pixel_mismatch = key
                report = self.run_harness()
                self.assertEqual(report["status"], "completed_with_pixel_differences")
                self.assertFalse(report["results"][0][key]["exact"])
                self.assertEqual(report["results"][0][key]["different_samples"], 1)

    def test_invalid_settings_and_labels_fail_before_processes_or_output_creation(self):
        output = self.root / "invalid"
        for extra in (["--bits", "8"], ["--bits", "16"], ["--bits", "0"], ["--dither", "cpp"], ["--decode-reps", "0"],
                      ["--repeats", "0"], ["--warmups", "-1"], ["--timeout", "0"],
                      ["--timeout=-1"], ["--timeout", "nan"], ["--timeout", "inf"],
                      ["--case-label", "missing"], ["--case-label", "first", "--case-label", "missing"]):
            with self.subTest(extra=extra), patch.object(bench, "binary_info") as probe, \
                    patch.object(bench, "run_process") as process, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main([*self.argv, "--out", str(output), *extra])
                self.assertEqual(error.exception.code, 2)
                probe.assert_not_called()
                process.assert_not_called()
                self.assertFalse(output.exists())

    def test_unverified_baseline_or_missing_jpeg_cases_are_rejected(self):
        for status, cases, extra in (("failed", self.cases, []), ("passed", [], []),
                                    ("passed", [dict(kind="png", label="png")], []),
                                    ("passed", [*self.cases, dict(kind="png", label="png")],
                                     ["--case-label", "png"])):
            with self.subTest(status=status, cases=cases), redirect_stderr(StringIO()):
                self.write_baseline(status=status, cases=cases)
                output = self.root / "invalid-baseline"
                with self.assertRaises(SystemExit):
                    bench.main([*self.argv, "--out", str(output), *extra])
                self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "report.json"
        sentinel.write_bytes(b"accepted report")
        with patch.object(bench, "binary_info") as probe, self.assertRaises(FileExistsError):
            bench.main([*self.argv, "--out", str(output)])
        probe.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"accepted report")

    def test_jpeg_mismatch_stops_before_pixel_timing(self):
        self.jpeg_mismatch = True
        report = self.run_harness(error=(ValueError, "JPEG byte reconstruction mismatch"))
        self.assertEqual(report["results"], [])
        self.assertEqual([r["phase"] for r in report["trials"]], ["verify_jpeg"])

    def test_process_errors_and_timeouts_fail_with_a_clean_report(self):
        for failure in (("version", "official", dict(returncode=7, stderr="bad version")),
                        ("verify", "rust", dict(returncode=4, stderr="bad decode")),
                        ("timing", "official", dict(returncode=None, error="Process timeout"))):
            with self.subTest(failure=failure):
                self.failure = failure
                report = self.run_harness(error=(RuntimeError, "Exit|Process timeout"))
                self.assertEqual(report["results"], [])
                self.assertIn("RuntimeError", report["error"])

    def test_inputs_changed_before_run_are_rejected_before_decoding(self):
        for key in ("jxl", "original"):
            with self.subTest(key=key):
                path = Path(self.cases[0][key])
                original = path.read_bytes()
                try:
                    path.write_bytes(b"changed before run")
                    report = self.run_harness(error=(ValueError, "Input changed since baseline"))
                    self.assertEqual(report["trials"], [])
                    self.assertIn(str(path), report["error"])
                finally:
                    path.write_bytes(original)

    def test_final_hash_checks_reject_late_edits_to_earlier_inputs_and_binaries(self):
        paths = [Path(self.cases[0][key]) for key in ("jxl", "original")]
        paths += list(self.binaries.values())
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                self.change = path
                try:
                    report = self.run_harness(error=(ValueError, "changed during measurement"))
                    self.assertEqual(len(report["results"]), 2)
                    if path in paths[:2]:
                        self.assertIn(str(path), report["error"])
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
