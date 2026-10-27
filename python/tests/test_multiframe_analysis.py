"""Thin analysis transport and native geometry reports, without encode trials."""

from collections.abc import Mapping
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np

import rs_jxl
from rs_jxl import contexts, multiframe


def _frames():
    first = np.zeros((8, 8, 4), dtype=np.uint8)
    first[..., 3] = 255
    second = first.copy()
    second[1, 1, 0] = 7
    third = first.copy()
    third[6, 6, 1] = 9
    return [first, second, third, third.copy()]


def _png(image, *, linear=False):
    """Small unfiltered RGBA8 fixture independent of the JXL encoder."""
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    height, width, channels = image.shape
    assert image.dtype == np.uint8 and channels == 4
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    color = chunk(b"gAMA", struct.pack(">I", 100000)) if linear else chunk(b"sRGB", b"\x01")
    rows = b"".join(b"\x00" + row.tobytes() for row in image)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + color
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _without_timing(report):
    return {key: value for key, value in report.items() if key != "analysis_seconds"}


class MultiframeAnalysisAdapters(unittest.TestCase):
    def request(self, **options):
        report = {"frames": [], "analysis_seconds": 0.125}
        with patch.object(multiframe, "execute", return_value=(b"", report)) as execute:
            self.assertIs(rs_jxl.analyze_multiframe(_frames(), **options), report)
        return execute.call_args.args[0]

    def test_reference_aliases_and_omitted_native_defaults(self):
        request = self.request()
        self.assertEqual(request["operation"], "analyze_multiframe")
        self.assertEqual(request["analysis_options"], {"reference": "smallest_crop"})
        self.assertEqual(request["options"], {"threads": 1, "strip_metadata": False})
        self.assertNotIn("reference", request)
        self.assertNotIn("analysis_options", request["options"])
        for reference, native in (("auto", "smallest_crop"), ("none", "full"),
                                  ("full", "full"), ("first", "first"),
                                  ("previous", "previous")):
            with self.subTest(reference=reference):
                self.assertEqual(self.request(reference=reference, analysis_options={})
                                 ["analysis_options"], {"reference": native})
        with patch("rs_jxl._options.os.cpu_count", return_value=300):
            self.assertEqual(self.request(threads=0)["options"]["threads"], 256)

    def test_all_options_are_copied_normalized_and_forwarded_at_top_level(self):
        options = {"allow_mask": False, "allow_multi_rect": True,
                   "adaptive_rectangles": False, "tile_size": np.uint32(4),
                   "max_rectangles": np.int64(9), "rectangle_bytes_per_pixel": np.float32(2.5)}
        original = options.copy()
        request = self.request(reference="first", threads=np.int64(2), strip_metadata=True,
                               analysis_options=MappingProxyType(options))
        self.assertEqual(request["analysis_options"], {**options, "reference": "first"})
        self.assertEqual(request["options"], {"threads": 2, "strip_metadata": True})
        self.assertEqual(options, original)
        self.assertIs(type(request["analysis_options"]["tile_size"]), int)
        self.assertIs(type(request["analysis_options"]["rectangle_bytes_per_pixel"]), float)
        json.dumps(request, allow_nan=False)
        for estimate in (0, 1, 0.5, np.int64(2)):
            self.assertEqual(self.request(analysis_options={"rectangle_bytes_per_pixel": estimate})
                             ["analysis_options"]["rectangle_bytes_per_pixel"], float(estimate))
        maximum = {"tile_size": 0xffffffff, "max_rectangles": sys.maxsize * 2 + 1}
        self.assertEqual(self.request(analysis_options=maximum)["analysis_options"],
                         {**maximum, "reference": "smallest_crop"})

    def test_invalid_configuration_fails_before_consuming_input(self):
        class UnreadFrames:
            def __iter__(self):
                raise AssertionError("invalid configuration consumed input")

        invalid = [({"analysis_options": {"reference": "first"}}, ValueError)]
        for name in ("unknown", "prune", "max_frames", "threads"):
            invalid.append(({"analysis_options": {name: 1}}, TypeError))
        for value in ([], [("allow_mask", True)], True, 1, "allow_mask"):
            invalid.append(({"analysis_options": value}, TypeError))
        for name in ("allow_mask", "allow_multi_rect", "adaptive_rectangles"):
            for value in (0, 1, "true", None, np.bool_(True)):
                invalid.append(({"analysis_options": {name: value}}, TypeError))
        for name, maximum in (("tile_size", 0xffffffff), ("max_rectangles", sys.maxsize * 2 + 1)):
            for value in (True, np.bool_(False), 1.5, "1", None):
                invalid.append(({"analysis_options": {name: value}}, TypeError))
            for value in (-1, 0, maximum + 1):
                invalid.append(({"analysis_options": {name: value}}, ValueError))
        for value in (True, np.bool_(False), "1", None, 1j):
            invalid.append(({"analysis_options": {"rectangle_bytes_per_pixel": value}}, TypeError))
        for value in (-0.01, float("nan"), float("inf"), -float("inf"), 10 ** 400):
            invalid.append(({"analysis_options": {"rectangle_bytes_per_pixel": value}}, ValueError))
        for value in ("unknown", "smallest_crop", None, []):
            invalid.append(({"reference": value}, ValueError))
        for value in (True, 1.5, "2", None):
            invalid.append(({"threads": value}, TypeError))
        for value in (-1, 257):
            invalid.append(({"threads": value}, ValueError))
        with patch.object(multiframe, "execute") as execute:
            for options, error in invalid:
                with self.subTest(options=options), self.assertRaises(error):
                    rs_jxl.analyze_multiframe(UnreadFrames(), **options)
            execute.assert_not_called()

    def test_duplicate_mapping_entries_are_rejected(self):
        class Duplicate(Mapping):
            def __iter__(self):
                return iter(("allow_mask", "allow_mask"))

            def __len__(self):
                return 2

            def __getitem__(self, name):
                return True

        with self.assertRaisesRegex(ValueError, "duplicate analysis option"):
            self.request(analysis_options=Duplicate())

    def test_shared_input_transport_context_and_report_are_not_reinterpreted(self):
        frames, extras, context = object(), object(), object()
        payload, specs = SimpleNamespace(data=b"pixels"), [{"format": "pixels"}]
        report = {"frames": [{"previous": None, "first": None, "plan": {}, "candidates": []}],
                  "reference_modes": [], "analysis_seconds": 0.125}
        with patch.object(multiframe, "_images", return_value=(payload, specs)) as images:
            with patch.object(multiframe, "execute", return_value=(b"", report)) as execute:
                actual = rs_jxl.analyze_multiframe(
                    frames, layout="chw", extra_channels=extras, bits_per_sample=12,
                    color_encoding="linear_srgb", icc_profile=None, borrow_inputs=True,
                    allow_device_copy=True, _context=context)
        self.assertIs(actual, report)
        images.assert_called_once_with(
            frames, layout="chw", extra_channels=extras, borrow_inputs=True,
            bits_per_sample=12, color_encoding="linear_srgb", icc_profile=None,
            allow_device_copy=True)
        self.assertIs(execute.call_args.args[0]["images"], specs)
        self.assertEqual(execute.call_args.args[1], b"pixels")
        self.assertIs(execute.call_args.kwargs["context"], context)
        self.assertEqual(self.request(icc_profile=b"profile")["images"][0]["icc"],
                         list(b"profile"))

    def test_input_snapshot_and_explicit_borrow_match_encoding_contract(self):
        for borrow in (False, True):
            source = np.zeros((2, 3, 4), dtype=np.uint8)

            def observe(request, data, **kwargs):
                source[0, 0, 0] = 9
                self.assertEqual(memoryview(data).cast("B")[0], 9 if borrow else 0)
                return b"", {}

            with self.subTest(borrow=borrow), patch.object(multiframe, "execute", side_effect=observe):
                rs_jxl.analyze_multiframe([source], borrow_inputs=borrow)

    def test_context_inherits_threads_and_rejects_override_mismatch_and_closed(self):
        native = SimpleNamespace(threads=2, closed=False)
        with patch.object(contexts, "NativeContext", return_value=native):
            encoder = rs_jxl.Encoder(threads=2)
        with patch.object(multiframe, "execute", return_value=(b"", {})) as execute:
            encoder.analyze_multiframe(_frames())
            self.assertEqual(execute.call_args.args[0]["options"]["threads"], 2)
            self.assertIs(execute.call_args.kwargs["context"], native)
            execute.reset_mock()
            with self.assertRaisesRegex(ValueError, "threads must match"):
                encoder.analyze_multiframe(_frames(), threads=1)
            with self.assertRaisesRegex(TypeError, "cannot override"):
                encoder.analyze_multiframe(_frames(), _context=object())
            native.closed = True
            with self.assertRaisesRegex(RuntimeError, "closed"):
                encoder.analyze_multiframe(_frames())
            execute.assert_not_called()


class NativeMultiframeAnalysis(unittest.TestCase):
    def test_previous_first_identical_statistics_and_geometry_schema(self):
        report = rs_jxl.analyze_multiframe(_frames())
        self.assertEqual((report["width"], report["height"], report["channels"],
                          report["frame_count"]), (8, 8, 4, 4))
        self.assertEqual(report["reference"], "smallest_crop")
        self.assertEqual(report["rectangle_bytes_per_pixel"], 1.0)
        self.assertTrue(report["crop_supported"])
        self.assertGreaterEqual(report["analysis_seconds"], 0)
        full = {"x": 0, "y": 0, "width": 8, "height": 8}
        initial, _, changed, identical = report["frames"]
        self.assertEqual(initial["previous"], full)
        self.assertEqual(initial["first"], full)
        self.assertIsNone(initial["previous_stats"])
        self.assertIsNone(initial["first_stats"])
        self.assertEqual(changed["previous"], {"x": 1, "y": 1, "width": 6, "height": 6})
        self.assertEqual(changed["first"], {"x": 6, "y": 6, "width": 1, "height": 1})
        stats = changed["previous_stats"]
        self.assertEqual(stats["changed_pixels"], 2)
        self.assertEqual(stats["changed_ratio"], 2 / 64)
        self.assertEqual(stats["bbox_pixels"], 36)
        self.assertEqual(stats["bbox_ratio"], 36 / 64)
        self.assertAlmostEqual(stats["bbox_fill_ratio"], 2 / 36)
        self.assertEqual(stats["bounds"], changed["previous"])
        self.assertTrue(stats["mask_eligible"])
        self.assertEqual(changed["first_stats"]["changed_pixels"], 1)
        self.assertEqual(changed["plan"]["reference_index"], 0)
        self.assertEqual(changed["plan"]["source"], 1)
        self.assertEqual(changed["plan"]["region"], changed["first"])
        self.assertEqual(changed["plan"]["changed"], changed["first"])
        self.assertIsNone(identical["previous"])
        self.assertIsNone(identical["previous_stats"]["bounds"])
        self.assertEqual(identical["previous_stats"]["changed_pixels"], 0)
        self.assertEqual(identical["previous_stats"]["bbox_pixels"], 0)
        self.assertEqual(identical["plan"]["reference_index"], 2)
        self.assertIsNone(identical["plan"]["changed"])
        self.assertEqual(identical["plan"]["layer_pixels"], 1)
        for index, frame in enumerate(report["frames"]):
            self.assertEqual(frame["index"], index)
            plan = frame["plan"]
            self.assertEqual(plan["layer_pixels"], plan["region"]["width"] * plan["region"]["height"])
            self.assertEqual(plan["layer_area_ratio"], plan["layer_pixels"] / 64)
            self.assertTrue(frame["candidates"])
            for candidate in frame["candidates"]:
                self.assertTrue({"strategy", "reference_index", "source"} <= candidate.keys())
                self.assertIn(candidate["strategy"], {
                    "full", "crop_previous", "crop_first", "mask_previous", "mask_first",
                    "tiles_previous", "tiles_first", "merged_previous", "merged_first"})
                self.assertNotIn("encoded_bytes", candidate)
                self.assertNotIn("winner_index", frame)
                self.assertEqual(candidate["layer_count"], len(candidate["rectangles"]))
                self.assertEqual(candidate["layer_pixels"],
                                 sum(rect["width"] * rect["height"] for rect in candidate["rectangles"]))
                self.assertEqual(candidate["layer_area_ratio"], candidate["layer_pixels"] / 64)
        modes = {entry["reference"]: entry for entry in report["reference_modes"]}
        self.assertEqual(set(modes), {"full", "previous", "first", "smallest_crop"})
        minimum = min(entry["layer_pixels"] for entry in modes.values())
        self.assertEqual(modes[report["best_geometric_reference"]]["layer_pixels"], minimum)
        for entry in modes.values():
            self.assertTrue({"average_changed_ratio", "crop_frames", "identical_frames"} <= entry.keys())
            self.assertEqual(entry["layer_area_ratio"], entry["layer_pixels"] / (4 * 64))
            self.assertIn(entry["suitability"], {"single_frame", "strong", "moderate", "limited",
                                                "full_frames_required"})

    def test_suitability_uses_subsequent_bbox_area_not_total_area_or_changed_ratio(self):
        first = np.zeros((10, 10), dtype=np.uint8)
        for rows, suitability in ((2, "strong"), (3, "moderate"), (6, "moderate"), (7, "limited")):
            second = first.copy()
            second[0, 0] = second[rows - 1, 9] = 1
            report = rs_jxl.analyze_multiframe([first, second])
            with self.subTest(rows=rows):
                for mode in report["reference_modes"]:
                    if mode["reference"] == "full":
                        self.assertEqual(mode["suitability"], "limited")
                        self.assertEqual(mode["average_changed_ratio"], 1.0)
                    else:
                        self.assertEqual(mode["suitability"], suitability)
                        self.assertEqual(mode["layer_area_ratio"], (100 + rows * 10) / 200)
                        self.assertEqual(mode["average_changed_ratio"], 2 / 100)
                        self.assertEqual(mode["crop_frames"], 1)

    def test_full_replacements_keep_actual_pair_statistics_but_summary_change_is_one(self):
        image = _frames()[0]
        for subsampled in (False, True):
            options = ({"extra_channels": [{"data": np.zeros((4, 4), dtype=np.uint8),
                                            "type": "depth", "dim_shift": 1}]}
                       if subsampled else {"reference": "full"})
            report = rs_jxl.analyze_multiframe([image, image], **options)
            with self.subTest(subsampled=subsampled):
                frame = report["frames"][1]
                for reference in ("previous", "first"):
                    self.assertIsNone(frame[reference])
                    self.assertEqual(frame[reference + "_stats"]["changed_pixels"], 0)
                self.assertIsNone(frame["plan"]["reference_index"])
                self.assertEqual(frame["plan"]["layer_pixels"], 64)
                for mode in report["reference_modes"]:
                    if subsampled or mode["reference"] == "full":
                        self.assertEqual(mode["average_changed_ratio"], 1.0)
                        self.assertEqual(mode["identical_frames"], 0)
                        self.assertEqual(mode["crop_frames"], 0)
                    else:
                        self.assertEqual(mode["average_changed_ratio"], 0.0)
                        self.assertEqual(mode["identical_frames"], 1)

    def test_paths_bytes_streams_and_arrays_have_identical_reports_except_timing(self):
        frames = _frames()
        expected = _without_timing(rs_jxl.analyze_multiframe(frames))
        encoded = [_png(frame) for frame in frames]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, data in enumerate(encoded):
                path = Path(directory) / f"frame-{index}.png"
                path.write_bytes(data)
                paths.append(path)
            variants = [encoded, paths, list(map(str, paths)), list(map(io.BytesIO, encoded)),
                        [paths[0], encoded[1], frames[2], frames[3]]]
            for index, inputs in enumerate(variants):
                with self.subTest(inputs=index):
                    self.assertEqual(_without_timing(rs_jxl.analyze_multiframe(inputs)), expected)

    def test_sparse_mask_eligibility_and_candidate_switches(self):
        first = _frames()[0]
        second = first.copy()
        second[0, 0, 0] = second[7, 7, 0] = 1
        options = {"tile_size": 2, "max_rectangles": 8}
        report = rs_jxl.analyze_multiframe([first, second], analysis_options=options)
        frame = report["frames"][1]
        self.assertTrue(frame["previous_stats"]["mask_eligible"])
        self.assertEqual(frame["previous_stats"]["bbox_fill_ratio"], 2 / 64)
        self.assertTrue(any(candidate["strategy"].startswith("mask_")
                            for candidate in frame["candidates"]))
        self.assertTrue(any(candidate["layer_count"] > 1 for candidate in frame["candidates"]))
        disabled = rs_jxl.analyze_multiframe([first, second], analysis_options={
            **options, "allow_mask": False, "allow_multi_rect": False})["frames"][1]
        self.assertEqual(disabled["previous_stats"], frame["previous_stats"])
        self.assertTrue(disabled["previous_stats"]["mask_eligible"])
        self.assertFalse(any(candidate["strategy"].startswith("mask_")
                             for candidate in disabled["candidates"]))
        self.assertTrue(all(candidate["layer_count"] == 1 for candidate in disabled["candidates"]))
        first[..., 3] = second[..., 3] = 0
        hidden = rs_jxl.analyze_multiframe([first, second])["frames"][1]
        self.assertEqual(hidden["previous_stats"]["changed_pixels"], 2)
        self.assertFalse(hidden["previous_stats"]["mask_eligible"])
        self.assertFalse(any(candidate["strategy"].startswith("mask_")
                             for candidate in hidden["candidates"]))

    def test_torch_chw_and_native_sample_metadata_match_numpy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        frames = [frame.astype(np.uint16) for frame in _frames()]
        options = {"bits_per_sample": 12, "color_encoding": "linear_srgb"}
        expected = rs_jxl.analyze_multiframe(frames, **options)
        tensors = [torch.from_numpy(frame).permute(2, 0, 1) for frame in frames]
        actual = rs_jxl.analyze_multiframe(tensors, layout="chw", **options)
        self.assertEqual(_without_timing(actual), _without_timing(expected))

    def test_float_comparison_uses_words_including_signed_zero_and_nan_payload(self):
        for word_dtype, dtype, nan_word, negative_zero in (
                (np.uint16, np.float16, 0x7e01, 0x8000),
                (np.uint32, np.float32, 0x7fc00001, 0x80000000)):
            with self.subTest(dtype=dtype):
                first = np.array([[0, nan_word, 0, 0]], dtype=word_dtype)
                second = first.copy()
                second[0, 0] = negative_zero
                second[0, 1] = nan_word + 1
                frames = [first.view(dtype), second.view(dtype), second.copy().view(dtype)]
                report = rs_jxl.analyze_multiframe(frames)
                self.assertEqual(report["frames"][1]["previous_stats"]["changed_pixels"], 2)
                self.assertEqual(report["frames"][2]["previous_stats"]["changed_pixels"], 0)
                self.assertIsNone(report["frames"][2]["previous"])

    def test_extra_channel_changes_and_subsampled_full_frame_fallback(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        for shift in (0, 1):
            size = 8 >> shift
            first = np.zeros((size, size), dtype=np.uint16)
            second = first.copy()
            second[1, 1] = 9
            report = rs_jxl.analyze_multiframe([image, image], extra_channels=[{
                "data": [first, second], "type": "depth", "dim_shift": shift}])
            with self.subTest(dim_shift=shift):
                self.assertEqual(report["frames"][1]["previous_stats"]["changed_pixels"], 1 << (2 * shift))
                self.assertEqual(report["crop_supported"], shift == 0)
                if shift:
                    self.assertEqual(report["frames"][1]["plan"]["layer_pixels"], 64)
                    self.assertTrue(all(entry["suitability"] == "full_frames_required"
                                        for entry in report["reference_modes"]))

    def test_geometry_sample_depth_and_color_metadata_mismatches_are_rejected(self):
        image = _frames()[0]
        mismatches = [image[:7], image[:, :7], image[..., :3], image.astype(np.uint16),
                      _png(image, linear=True)]
        for other in mismatches:
            for reference in ("auto", "full"):
                with self.subTest(other=type(other).__name__, reference=reference):
                    with self.assertRaises(rs_jxl.CodecError) as caught:
                        rs_jxl.analyze_multiframe([image, other], reference=reference)
                    self.assertEqual(caught.exception.kind, "invalid_input")
        with self.assertRaises(ValueError):
            rs_jxl.analyze_multiframe([])

    def test_context_reuse_reference_modes_and_single_frame(self):
        frames = _frames()
        with rs_jxl.Encoder(threads=2) as encoder:
            for reference in ("auto", "previous", "first", "full"):
                expected = rs_jxl.analyze_multiframe(frames, reference=reference, threads=2)
                actual = encoder.analyze_multiframe(frames, reference=reference)
                self.assertEqual(_without_timing(actual), _without_timing(expected))
                if reference == "full":
                    self.assertTrue(all(frame["plan"]["layer_pixels"] == 64 for frame in actual["frames"]))
            with self.assertRaisesRegex(ValueError, "threads must match"):
                encoder.analyze_multiframe(frames, threads=1)
            with self.assertRaises(rs_jxl.CodecError):
                encoder.analyze_multiframe([frames[0], frames[0][:7]])
            single = encoder.analyze_multiframe(frames[:1], threads=2)
            self.assertTrue(all(entry["suitability"] == "single_frame" for entry in single["reference_modes"]))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            encoder.analyze_multiframe(frames)


if __name__ == "__main__":
    unittest.main()
