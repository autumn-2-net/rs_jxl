"""Small parser/oracle/control-flow tests; no native DLL calls or builds."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import benchmark_pixel_calls as bench
from compare_cpp_port import Pixels
from test_compare_cpp_port import fixture, make_png


class ParserTests(unittest.TestCase):
    def argv(self, *extra):
        return ["--before", "old.dll", "--after", "new.dll", "--case", "a.png",
                "--out", "unused-pixel-report.json", *extra]

    def test_defaults_and_repeatable_cases(self):
        args = bench.parse_args(self.argv("--case", "b.png"))
        self.assertEqual((args.effort, args.threads, args.repeats, args.warmups), (9, [1, 8], 3, 1))
        self.assertEqual(args.case, [Path("a.png"), Path("b.png")])
        args = bench.parse_args(self.argv("--effort", "11", "--threads", "2", "4",
                                         "--repeats", "1", "--warmups", "0"))
        self.assertEqual((args.effort, args.threads, args.repeats, args.warmups), (11, [2, 4], 1, 0))

    def test_invalid_settings_and_existing_output(self):
        for extra in (("--effort", "0"), ("--effort", "12"), ("--repeats", "0"),
                      ("--warmups", "-1"), ("--threads", "0"), ("--threads", "257"),
                      ("--threads", "1", "1")):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.parse_args(self.argv(*extra))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing.json"
            output.write_text("keep", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.parse_args(self.argv("--out", str(output)))
            self.assertEqual(output.read_text(), "keep")

    def test_native_timer_validation(self):
        self.assertEqual(bench.codec_seconds({"report": {"codec_seconds": 0.125}}), 0.125)
        self.assertEqual(bench.codec_seconds({"report": {"codec_seconds": 0}}), 0)
        for value in (None, True, -1, "0.1", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.codec_seconds({"report": {"codec_seconds": value}})

    def test_native_timer_requires_nested_report_not_top_level(self):
        for meta in ({}, {"codec_seconds": 0.125}, {"report": {}},
                     {"report": None}, {"report": []}, None):
            with self.subTest(meta=meta), self.assertRaisesRegex(ValueError, "report.codec_seconds"):
                bench.codec_seconds(meta)
        self.assertEqual(bench.codec_seconds({"codec_seconds": 99,
                                             "report": {"codec_seconds": 0.125}}), 0.125)

    def test_summary_excludes_warmups_and_uses_median(self):
        samples = [dict(seconds=value, warmup=warmup) for value, warmup in
                   ((99, True), (1, False), (5, False), (3, False))]
        summary = bench.timing_summary(samples, "seconds", 6_000_000)
        self.assertEqual(summary, dict(seconds=3, min_seconds=1, max_seconds=5, mpix_s=2))


class PixelTests(unittest.TestCase):
    def test_structured_oracle_preserves_channels_alpha_and_wide_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original.png"
            for depth in (8, 16):
                a, b, c = (19, 81, 253) if depth == 8 else (259, 513, 65534)
                top = (1 << depth) - 1
                for color, row, expected in (
                    (0, [a, c], [[a], [c]]),
                    (2, [a, b, c, c, b, a], [[a, b, c], [c, b, a]]),
                    (4, [a, 0, c, top], [[a, 0], [c, top]]),
                    (6, [a, b, c, 0, c, b, a, top], [[a, b, c, 0], [c, b, a, top]]),
                ):
                    with self.subTest(depth=depth, color=color):
                        path.write_bytes(make_png(depth, color, [row]))
                        rgba = np.empty((1, 2, 4), dtype=np.uint8 if depth == 8 else np.uint16)
                        rgba[:, :, 3] = top
                        for x, values in enumerate(expected):
                            rgba[0, x, :3] = values[0] if len(values) <= 2 else values[:3]
                            if len(values) in (2, 4):
                                rgba[0, x, 3] = values[-1]
                        # Exercise the structured oracle contract without requiring OpenCV.
                        with patch.object(bench, "png_samples", return_value=Pixels(2, 1, depth, rgba)):
                            case, array = bench.load_case(0, path)
                        self.assertEqual(array.tolist(), [expected])
                        self.assertEqual(array.dtype, np.uint8 if depth == 8 else np.uint16)
                        self.assertTrue(array.flags.c_contiguous)
                        self.assertFalse(array.flags.writeable)
                        self.assertEqual(case["original_sha256"], bench.file_sha256(path))
                        self.assertEqual(case["pixel_sha256"], bench.sha256(array.tobytes()))
                        self.assertEqual(case["channels"], len(expected[0]))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "Existing PNG oracle requires opencv-python")
    def test_real_oracle_palette_transparency_and_wide_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original.png"
            path.write_bytes(make_png(8, 3, [[0, 1]], extra=[
                (b"PLTE", bytes([19, 81, 253, 7, 11, 13])), (b"tRNS", b"\0\xff")]))
            _, array = bench.load_case(0, path)
            self.assertEqual(array.tolist(), [[[19, 81, 253, 0], [7, 11, 13, 255]]])
            path.write_bytes(make_png(8, 0, [[19, 23]], extra=[(b"tRNS", b"\0\x13")]))
            _, array = bench.load_case(0, path)
            self.assertEqual(array.tolist(), [[[19, 0], [23, 255]]])
            for depth, dtype, expected in (
                (8, np.uint8, [[[19, 81, 253, 0], [7, 11, 13, 255]]]),
                (16, np.uint16, [[[259, 513, 65534, 0], [1, 257, 32769, 65535]]]),
            ):
                with self.subTest(depth=depth):
                    path.write_bytes(fixture(depth))
                    case, array = bench.load_case(0, path)
                    self.assertEqual(array.dtype, dtype)
                    self.assertEqual(array.tolist(), expected)
                    self.assertEqual(case["depth"], depth)
                    self.assertEqual(case["pixel_bytes"], 2 * 4 * np.dtype(dtype).itemsize)

    def test_hidden_rgb_low_bits_shape_and_dtype_are_strict(self):
        expected = np.array([[[259, 513, 65534, 0]]], dtype=np.uint16)
        bench.verify_samples(expected, expected.copy())
        wrong = expected.copy()
        wrong[0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "hidden RGB"):
            bench.verify_samples(expected, wrong)
        for wrong in (expected.astype(np.uint8), expected[:, :, :3], expected.tolist()):
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                bench.verify_samples(expected, wrong)

    def test_sample_verification_rejects_widening_and_narrowing(self):
        for dtype, other_dtypes in ((np.uint8, (np.uint16, np.uint32, np.float32)),
                                    (np.uint16, (np.uint8, np.uint32, np.float64))):
            expected = np.array([[[19, 81, 253, 0]]], dtype=dtype)
            bench.verify_samples(expected, expected.copy())
            for other in other_dtypes:
                with self.subTest(original=dtype, decoded=other), self.assertRaisesRegex(ValueError, "shape/dtype"):
                    bench.verify_samples(expected, expected.astype(other))

    def test_jpeg_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not.png"
            path.write_bytes(b"\xff\xd8JPEG")
            with self.assertRaisesRegex(ValueError, "Only PNG"):
                bench.load_case(0, path)


class FakeAPI:
    """Deliberately distinct encodings, with context-bound library selection."""

    def __init__(self, test, paths):
        self.test = test
        self.events = []
        self.contexts = []
        self.payloads = {}
        self.timing = False
        self.clock = 0
        self.corrupt = False
        self.corrupt_timed_decode = False
        self.on_encode = lambda: None
        libs = {str(path.resolve()): SimpleNamespace(name=name) for name, path in paths.items()}
        self.original_library = Mock(side_effect=AssertionError("library selected outside constructor"))
        self.ffi = SimpleNamespace(ffi=SimpleNamespace(dlopen=Mock(side_effect=libs.__getitem__)),
                                   _validate_abi=Mock(), library=self.original_library)
        owner = self

        class Context:
            def __init__(self, *, threads):
                test.assertFalse(owner.timing)
                self.lib = owner.ffi.library()
                self.threads = threads
                self.closed = False
                owner.contexts.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                test.assertFalse(owner.timing)
                self.closed = True

            def event(self, phase):
                test.assertIs(owner.ffi.library, owner.original_library)
                owner.events.append((phase, self.lib.name, self.threads, owner.timing))

        class Encoder(Context):
            def encode(self, array, **options):
                self.event("encode")
                test.assertEqual(options, dict(effort=9, source_search=False, return_info=True))
                test.assertFalse(any(key.upper().startswith(("JXL_", "RS_JXL_")) for key in os.environ))
                owner.on_encode()
                # All streams differ, even repeated outputs from the same DLL.
                data = (self.lib.name.encode() + str(len(owner.payloads)).encode()) * 3
                owner.payloads[data] = (array.copy(), self.lib.name)
                return data, {"operation": "encode", "bytes": len(data), "jpeg_input": False,
                              "report": {"codec_seconds": 0.125 if self.lib.name == "before" else 0.0625}}

        class Decoder(Context):
            def decode(self, data):
                self.event("decode")
                test.assertIsInstance(data, bytes)
                array, encoder = owner.payloads[data]
                result = array.copy()
                if ((owner.corrupt and encoder == "after" and self.lib.name == "after")
                        or (owner.corrupt_timed_decode and owner.timing)):
                    result[0, 0, 0] ^= 1
                owner.events[-1] += (data,)
                return result

        self.module = SimpleNamespace(Encoder=Encoder, Decoder=Decoder, _ffi=self.ffi)

    def perf_counter(self):
        self.timing = not self.timing
        self.clock += 0.25
        return self.clock


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = {name: self.root / (name + ".dll") for name in bench.NAMES}
        for name, path in self.paths.items():
            path.write_bytes(name.encode())
        self.original = self.root / "wide-hidden.png"
        self.original.write_bytes(fixture())
        self.args = bench.parse_args([
            "--before", str(self.paths["before"]), "--after", str(self.paths["after"]),
            "--case", str(self.original), "--threads", "1", "8", "--repeats", "2",
            "--warmups", "1", "--out", str(self.root / "report.json")])
        self.api = FakeAPI(self, self.paths)

    def run_harness(self):
        real_verify, real_hash = bench.verify_samples, bench.file_sha256

        def verify(*args):
            self.assertFalse(self.api.timing)
            return real_verify(*args)

        def file_hash(*args):
            self.assertFalse(self.api.timing)
            return real_hash(*args)

        def oracle(data):
            self.assertFalse(self.api.timing)
            bench.png_payload(data)
            self.assertEqual(data, fixture())
            return Pixels(2, 1, 16, np.array(
                [[[259, 513, 65534, 0], [1, 257, 32769, 65535]]], dtype=np.uint16))

        with patch.dict(sys.modules, {"rs_jxl": self.api.module}), \
                patch.object(sys, "path", sys.path.copy()), \
                patch.dict(os.environ, {"JXL_TEST": "1", "RS_JXL_TEST": "2"}), \
                patch.object(bench.time, "perf_counter", self.api.perf_counter), \
                patch.object(bench, "verify_samples", side_effect=verify), \
                patch.object(bench, "file_sha256", side_effect=file_hash), \
                patch.object(bench, "png_samples", side_effect=oracle), \
                redirect_stdout(StringIO()):
            report = bench.benchmark(self.args)
            self.assertNotIn("JXL_TEST", os.environ)
            self.assertNotIn("RS_JXL_TEST", os.environ)
        self.assertFalse(self.api.timing)
        self.assertTrue(all(context.closed for context in self.api.contexts))
        self.assertEqual(json.loads(self.args.out.read_text())["status"], report["status"])
        return report

    def test_alternation_fixed_baseline_cross_decode_and_report(self):
        report = self.run_harness()
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(len(self.api.contexts), 8)  # One encoder/decoder per DLL/thread setting.
        self.assertEqual(self.api.ffi.ffi.dlopen.call_count, 2)
        self.assertEqual(self.api.ffi._validate_abi.call_count, 2)
        self.api.original_library.assert_not_called()
        baseline_encodes = [event for event in self.api.events if event[0] == "encode" and not event[3]]
        self.assertEqual(baseline_encodes, [("encode", "before", 1, False)])
        timed_decodes = [event for event in self.api.events if event[0] == "decode" and event[3]]
        self.assertEqual(len({event[4] for event in timed_decodes}), 1)
        case = report["cases"][0]
        self.assertEqual(case["baseline"]["codec_seconds"], 0.125)
        for row in report["summaries"]:
            threads = row["threads"]
            calls = [event[1] for event in self.api.events if event[0] == "encode"
                     and event[2] == threads and event[3]]
            expected_order = [name for repeat in (-1, 0, 1) for name in bench.call_order(repeat, 0, threads)]
            self.assertEqual(calls, expected_order)
            self.assertEqual([event[1] for event in timed_decodes if event[2] == threads], expected_order)
            self.assertEqual(row["verified_calls"]["baseline"], {"before": 1, "after": 1})
            self.assertEqual(row["verified_calls"]["decode"], {"before": 3, "after": 3})
            self.assertEqual(row["speedup"]["encode_codec"], 2)
            for name in bench.NAMES:
                self.assertEqual(row["verified_calls"]["cross_decode"][name], {"before": 3, "after": 3})
                entry = row["encode"][name]
                expected_codec_seconds = 0.125 if name == "before" else 0.0625
                self.assertEqual(entry["codec"]["seconds"], expected_codec_seconds)
                self.assertEqual(len(entry["samples"]), 3)
                self.assertEqual([sample["warmup"] for sample in entry["samples"]], [True, False, False])
                self.assertEqual(entry["whole_call"]["seconds"], 0.25)
                self.assertEqual(entry["whole_call"]["mpix_s"], case["pixels"] / 1e6 / 0.25)
                self.assertEqual(len({s["sha256"] for s in entry["samples"]}), 3)
                for sample in entry["samples"]:
                    self.assertEqual(sample["codec_seconds"], expected_codec_seconds)
                    self.assertEqual(sample["bpp"], sample["bytes"] * 8 / case["pixels"])
        self.assertEqual(report["binaries"]["before"]["sha256"], bench.file_sha256(self.paths["before"]))
        self.assertIn("JXL_TEST", report["environment"]["removed_inherited_names"])

    def test_cross_decode_hidden_rgb_failure_writes_partial_report(self):
        self.api.corrupt = True
        report = self.run_harness()
        self.assertEqual(report["status"], "failed")
        self.assertIn("hidden RGB", report["errors"][0])
        row = report["summaries"][0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["phase"], "encode")
        self.assertNotIn("speedup", row)
        self.assertEqual(row["verified_calls"]["cross_decode"]["after"], {"before": 1, "after": 0})

    def test_binary_change_invalidates_report(self):
        self.api.on_encode = lambda: self.paths["after"].write_bytes(b"changed")
        report = self.run_harness()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("Input/library changed" in error for error in report["errors"]))

    def test_each_timed_decode_is_verified(self):
        self.api.corrupt_timed_decode = True
        report = self.run_harness()
        self.assertEqual(report["status"], "failed")
        self.assertIn("hidden RGB", report["errors"][0])
        row = report["summaries"][0]
        self.assertEqual(row["phase"], "decode")
        self.assertEqual(row["verified_calls"]["decode"], {"before": 0, "after": 0})
        self.assertNotIn("speedup", row)

    def test_oracle_failure_records_error_without_codec_calls(self):
        self.original.write_bytes(b"not a png")
        report = self.run_harness()
        self.assertEqual(report["status"], "failed")
        self.assertIn("Only PNG", report["errors"][0])
        self.assertEqual(self.api.events, [])

    def test_existing_report_is_never_overwritten_even_without_parser(self):
        self.args.out.write_text("keep", encoding="utf-8")
        with patch.dict(os.environ), self.assertRaises(FileExistsError):
            bench.benchmark(self.args)
        self.assertEqual(self.args.out.read_text(), "keep")
        self.assertEqual(self.api.events, [])

if __name__ == "__main__":
    unittest.main()
