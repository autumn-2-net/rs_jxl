"""Automatic-search adapters and native reports; no Python image analysis."""

from collections.abc import Mapping
import json
import sys
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

import rs_jxl
from rs_jxl import multiframe


class MultiframeOptionAdapters(unittest.TestCase):
    def test_extra_grids_forward_without_mutating_input_and_validate_early(self):
        grids = [np.uint32(8), 16, 32, 8, 64]
        request = self.request(strategy="auto", auto_options={"extra_tile_sizes": grids})
        self.assertEqual(request["auto_options"]["extra_tile_sizes"], grids)
        self.assertIsNot(request["auto_options"]["extra_tile_sizes"], grids)
        json.dumps(request)
        for value in (None, True, "8,16", 8, [True], [1.5], ["8"]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.request(strategy="auto", auto_options={"extra_tile_sizes": value})
        for value in ([0], [-1], [1 << 32]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.request(strategy="auto", auto_options={"extra_tile_sizes": value})

    def setUp(self):
        self.frames = [np.zeros((2, 3, 4), dtype=np.uint8)]

    def request(self, **options):
        with patch.object(multiframe, "execute", return_value=(b"encoded", {})) as execute:
            self.assertEqual(rs_jxl.encode_multiframe(self.frames, **options), b"encoded")
        return execute.call_args.args[0]

    def test_omission_preserves_native_defaults_without_injecting_values(self):
        for strategy in ("crop", "auto"):
            with self.subTest(strategy=strategy):
                request = self.request(strategy=strategy)
                self.assertNotIn("auto_options", request)
                self.assertNotIn("auto_options", request["options"])
                self.assertEqual(request["auto"], strategy == "auto")
        self.assertNotIn("auto_options", self.request(auto_options=None))
        self.assertEqual(self.request(strategy="auto", auto_options={})["auto_options"], {})

    def test_all_options_are_forwarded_at_top_level_without_mutating_mapping(self):
        options = {"allow_mask": False, "allow_multi_rect": True,
                   "tile_size": np.uint32(8), "max_rectangles": np.int64(4097),
                   "prune": False, "adaptive_rectangles": False,
                   "max_frames": 12, "max_total_trials": 30, "max_trial_pixels": 4000}
        original = options.copy()
        request = self.request(strategy="auto", reference="first",
                               auto_options=MappingProxyType(options), encoder_options={"rct": 0})
        self.assertEqual(request["operation"], "encode_multiframe")
        self.assertEqual(request["auto_options"], options)
        self.assertEqual(request["reference"], "first")
        self.assertEqual(request["options"]["rct"], 0)
        self.assertNotIn("auto_options", request["options"])
        self.assertEqual(options, original)
        self.assertIsNot(request["auto_options"], options)
        self.assertIs(type(request["auto_options"]["tile_size"]), int)
        json.dumps(request)

    def test_budgets_accept_zero_and_native_integer_maxima(self):
        for value in (0, sys.maxsize * 2 + 1):
            options = {"max_frames": value, "max_total_trials": value}
            self.assertEqual(self.request(strategy="auto", auto_options=options)["auto_options"], options)
        for value in (0, 0xffffffffffffffff):
            self.assertEqual(self.request(strategy="auto", auto_options={"max_trial_pixels": value})
                             ["auto_options"]["max_trial_pixels"], value)

    def test_invalid_options_fail_before_input_consumption_or_native_execution(self):
        class UnreadFrames:
            def __iter__(self):
                raise AssertionError("invalid options must fail before consuming frames")

        invalid = [({"strategy": "crop", "auto_options": {}}, ValueError),
                   ({"auto_options": {"reference": "auto"}}, ValueError),
                   ({"auto_options": {"unknown": True}}, TypeError)]
        for value in ([], [("prune", False)], True, 1, "prune"):
            invalid.append(({"auto_options": value}, TypeError))
        for name in ("allow_mask", "allow_multi_rect", "prune", "adaptive_rectangles"):
            for value in (0, 1, "true", None, np.bool_(True)):
                invalid.append(({"auto_options": {name: value}}, TypeError))
        for name in ("tile_size", "max_rectangles", "max_frames", "max_total_trials", "max_trial_pixels"):
            for value in (True, np.bool_(False), 1.5, "1", None):
                invalid.append(({"auto_options": {name: value}}, TypeError))
            invalid.append(({"auto_options": {name: -1}}, ValueError))
        for name, value in (("tile_size", 0), ("max_rectangles", 0),
                            ("tile_size", 1 << 32), ("max_rectangles", sys.maxsize * 2 + 2),
                            ("max_frames", sys.maxsize * 2 + 2),
                            ("max_total_trials", sys.maxsize * 2 + 2),
                            ("max_trial_pixels", 1 << 64)):
            invalid.append(({"auto_options": {name: value}}, ValueError))
        with patch.object(multiframe, "execute") as execute:
            for options, error in invalid:
                with self.subTest(options=options), self.assertRaises(error):
                    rs_jxl.encode_multiframe(UnreadFrames(), **{"strategy": "auto", **options})
            execute.assert_not_called()

    def test_duplicate_mapping_entries_are_rejected(self):
        class Duplicate(Mapping):
            def __iter__(self):
                return iter(("prune", "prune"))

            def __len__(self):
                return 2

            def __getitem__(self, name):
                return False

        with self.assertRaisesRegex(ValueError, "duplicate auto option"):
            self.request(strategy="auto", auto_options=Duplicate())

    def test_report_and_context_are_forwarded_without_python_recomputation(self):
        report = {"frames": [{
            "winner_index": 0, "analysis_seconds": 0.125,
            "differences": [{"reference": 0, "changed_pixels": 0, "bounds": None}],
            "candidates": [{"strategy": "Full", "encoded_bytes": 10,
                            "encoded_frames": 1, "trial_pixels": 6, "elapsed_seconds": 0.25}],
            "pruned": [{"strategy": "MergedFirst", "lower_bound": 11,
                        "encoded_frames": 0, "planned_frames": 2,
                        "trial_pixels": 3, "elapsed_seconds": 0.5}],
        }]}
        context = object()
        with patch.object(multiframe, "execute", return_value=(b"encoded", report)) as execute:
            data, actual = rs_jxl.encode_multiframe(
                self.frames, strategy="auto", auto_options={"prune": True},
                return_info=True, _context=context)
        self.assertEqual(data, b"encoded")
        self.assertIs(actual, report)
        self.assertIs(execute.call_args.kwargs["context"], context)
        self.assertNotIn("encoded_bytes", actual["frames"][0]["pruned"][0])


class NativeMultiframeOptions(unittest.TestCase):
    def test_reports_and_exact_samples_with_and_without_pruning(self):
        for dtype in (np.uint8, np.uint16):
            first = np.arange(8 * 8 * 4, dtype=dtype).reshape(8, 8, 4)
            first[..., 3] = 0
            second = first.copy()
            second[2, 3, 0] ^= 7
            frames = [first, second, second.copy()]
            for prune in (False, True):
                with self.subTest(dtype=dtype, prune=prune):
                    data, report = rs_jxl.encode_multiframe(
                        frames, strategy="auto", reference="previous", durations=[3, 4, 5],
                        effort=3, return_info=True,
                        auto_options={"prune": prune, "adaptive_rectangles": False})
                    decoded = rs_jxl.decode(data, frame=None)
                    self.assertEqual(len(decoded), len(frames))
                    for actual, expected in zip(decoded, frames):
                        np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(len(report["frames"]), 3)
                    for frame in report["frames"]:
                        self.assertGreaterEqual(frame["analysis_seconds"], 0)
                        completed = frame["candidates"]
                        self.assertTrue(completed)
                        winner = frame["winner_index"]
                        self.assertEqual(winner, min(range(len(completed)),
                                                    key=lambda i: completed[i]["encoded_bytes"]))
                        for candidate in completed:
                            self.assertGreater(candidate["encoded_bytes"], 0)
                            self.assertGreater(candidate["encoded_frames"], 0)
                            self.assertGreater(candidate["trial_pixels"], 0)
                            self.assertGreaterEqual(candidate["elapsed_seconds"], 0)
                            self.assertNotIn("lower_bound", candidate)
                            self.assertFalse(candidate["strategy"].startswith("Merged"))
                        if not prune:
                            self.assertEqual(frame["pruned"], [])
                        for candidate in frame["pruned"]:
                            self.assertNotIn("encoded_bytes", candidate)
                            self.assertGreaterEqual(candidate["lower_bound"], completed[winner]["encoded_bytes"])
                            self.assertLessEqual(candidate["encoded_frames"], candidate["planned_frames"])
                            self.assertGreaterEqual(candidate["elapsed_seconds"], 0)
                            self.assertGreaterEqual(candidate["trial_pixels"], 0)
                    self.assertEqual(report["frames"][1]["differences"], [{
                        "reference": 0, "source": 1, "changed_pixels": 1,
                        "bounds": {"x": 3, "y": 2, "width": 1, "height": 1},
                    }])
                    self.assertEqual(report["frames"][2]["differences"], [{
                        "reference": 1, "source": 2, "changed_pixels": 0, "bounds": None,
                    }])

    def test_zero_budgets_fail_explicitly_in_native_code(self):
        frame = np.zeros((2, 3), dtype=np.uint8)
        for name in ("max_frames", "max_total_trials", "max_trial_pixels"):
            with self.subTest(name=name):
                with self.assertRaises(rs_jxl.CodecError) as caught:
                    rs_jxl.encode_multiframe([frame], strategy="auto", effort=3,
                                             auto_options={name: 0})
                self.assertEqual(caught.exception.kind, "resource_limit")


if __name__ == "__main__":
    unittest.main()
