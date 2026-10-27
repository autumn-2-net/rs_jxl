"""Opt-in retained-reference transport and exact native sequence reconstruction."""

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

import rs_jxl
from rs_jxl import _ffi, multiframe
import test_bindings as fixtures


def recurrence(dtype=np.uint8):
    """X, A, B, A2: the useful background is neither first nor previous."""
    random = np.random.default_rng(38013)
    maximum = np.iinfo(dtype).max
    frames = [random.integers(0, maximum + 1, (32, 32, 4), dtype=dtype)
              for _ in range(3)]
    for frame in frames:
        frame[..., 3] = maximum
        frame[::2, ::2, 3] = 0
    frames.append(frames[1].copy())
    frames[-1][10, 12, 0] ^= 7  # Change invisible RGB, not merely displayed color.
    return frames


def encode_retained(frames, **kwargs):
    options = {"strategy": "auto", "reference": "auto", "effort": 3,
               "threads": 1, "return_info": True,
               "auto_options": {"retained_reference": True, "allow_mask": False,
                                "allow_multi_rect": False}}
    options.update(kwargs)
    return rs_jxl.encode_multiframe(frames, **options)


def lookahead_recurrence(dtype=np.uint8):
    """A, B, B2, A2: retaining A requires overwriting B after blending B2."""
    first, second = recurrence(dtype)[:2]
    frames = [first, second, second.copy(), first.copy()]
    frames[2][2, 2, 0] ^= 7
    frames[3][10, 12, 0] ^= 7
    return frames


def encode_lookahead(frames, **kwargs):
    auto_options = {"retained_reference": True, "reference_lookahead": True,
                    "allow_mask": False, "allow_multi_rect": False}
    auto_options.update(kwargs.pop("auto_options", {}))
    return encode_retained(frames, auto_options=auto_options, **kwargs)


class RetainedOptionAdapters(unittest.TestCase):
    def test_boolean_forwarding_does_not_inject_search_defaults(self):
        frames = [np.zeros((2, 3), dtype=np.uint8)]
        for enabled in (False, True):
            options = MappingProxyType({"retained_reference": enabled})
            with self.subTest(enabled=enabled), patch.object(
                    multiframe, "execute", return_value=(b"jxl", {})) as execute:
                data = rs_jxl.encode_multiframe(frames, strategy="auto", auto_options=options)
                self.assertEqual(data, b"jxl")
                request = execute.call_args.args[0]
                self.assertEqual(request["auto_options"], {"retained_reference": enabled})
                self.assertIsNot(request["auto_options"], options)
                self.assertNotIn("retained_reference", request["options"])
                self.assertNotIn("reference_lookahead", request["auto_options"])

    def test_lookahead_forwarding_does_not_mutate_or_inject_defaults(self):
        for values in ({"reference_lookahead": False},
                       {"reference_lookahead": True, "retained_reference": True}):
            options = MappingProxyType(values)
            with self.subTest(options=options), patch.object(
                    multiframe, "execute", return_value=(b"jxl", {})) as execute:
                rs_jxl.encode_multiframe([np.zeros((2, 3), dtype=np.uint8)],
                                        strategy="auto", auto_options=options)
                request = execute.call_args.args[0]
                self.assertEqual(request["auto_options"], values)
                self.assertIsNot(request["auto_options"], options)
                self.assertNotIn("reference_lookahead", request["options"])

    def test_validation_precedes_generator_consumption(self):
        class UnreadFrames:
            def __iter__(self):
                raise AssertionError("invalid options consumed the input")

        cases = []
        for field in ("retained_reference", "reference_lookahead"):
            for value in (0, 1, 1.0, "true", None, np.bool_(True), [], {}):
                cases.append(({"auto_options": {field: value}}, TypeError))
        for options in ({"reference_lookahead": True},
                        {"reference_lookahead": True, "retained_reference": False}):
            cases.append(({"auto_options": options}, ValueError))
        for reference in ("first", "previous", "full", "none"):
            cases.append(({"reference": reference,
                           "auto_options": {"retained_reference": True}}, ValueError))
            cases.append(({"reference": reference, "auto_options": {
                "retained_reference": True, "reference_lookahead": True}}, ValueError))
        for enabled in (False, True):
            cases.append(({"strategy": "crop",
                           "auto_options": {"retained_reference": enabled}}, ValueError))
            cases.append(({"strategy": "crop", "auto_options": {
                "retained_reference": True, "reference_lookahead": enabled}}, ValueError))
        with patch.object(multiframe, "execute") as execute:
            for kwargs, error in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(error):
                    rs_jxl.encode_multiframe(UnreadFrames(), **{"strategy": "auto", **kwargs})
            execute.assert_not_called()

    def test_false_preserves_all_existing_reference_choices(self):
        for reference in ("auto", "previous", "first", "full", "none"):
            with self.subTest(reference=reference), patch.object(
                    multiframe, "execute", return_value=(b"jxl", {})):
                data = rs_jxl.encode_multiframe(
                    [np.zeros((2, 3), dtype=np.uint8)], strategy="auto", reference=reference,
                    auto_options={"retained_reference": False, "reference_lookahead": False})
                self.assertEqual(data, b"jxl")

    def test_duplicate_policy_keys_rejected(self):
        class Duplicate(Mapping):
            def __init__(self, key):
                self.key = key

            def __len__(self):
                return 2

            def __iter__(self):
                return iter((self.key, self.key))

            def __getitem__(self, key):
                return True

        for field in ("retained_reference", "reference_lookahead"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "duplicate auto option"):
                rs_jxl.encode_multiframe([], strategy="auto", auto_options=Duplicate(field))

    def test_static_analyzer_does_not_accept_dynamic_policy(self):
        for field in ("retained_reference", "reference_lookahead"):
            with self.subTest(field=field), self.assertRaisesRegex(TypeError, "unknown analysis option"):
                rs_jxl.analyze_multiframe([], analysis_options={field: True})


class NativeRetainedReferences(fixtures.PixelAssertions, unittest.TestCase):
    def assertReports(self, report):
        frames = report["frames"]
        for index, frame in enumerate(frames):
            self.assertTrue(frame["retained_reference"])
            self.assertFalse(frame["reference_lookahead"])
            self.assertEqual(frame["lookahead_comparisons"], 0)
            before, after = frame["references_before"], frame["references_after"]
            self.assertEqual(len(before), 4)
            self.assertEqual(len(after), 4)
            self.assertIsNone(before[3])
            self.assertIsNone(after[3])
            if index == 0:
                self.assertEqual(before, [None] * 4)
            else:
                self.assertEqual(before, frames[index - 1]["references_after"])
            for entry in before:
                if entry is not None:
                    self.assertTrue(0 <= entry < index)
            for entry in after:
                if entry is not None:
                    self.assertTrue(0 <= entry <= index)
            for difference in frame["differences"]:
                self.assertEqual(before[difference["source"]], difference["reference"])
            for candidate in frame["candidates"] + frame["pruned"]:
                self.assertIn(candidate["source"], range(4))
                if candidate["reference_index"] is not None:
                    self.assertEqual(before[candidate["source"]], candidate["reference_index"])
                if index == len(frames) - 1:
                    self.assertEqual(candidate["save_as_reference"], 0)
                else:
                    self.assertIn(candidate["save_as_reference"], (1, 2))
        final = frames[-1]
        winner = final["candidates"][final["winner_index"]]
        self.assertEqual(winner["strategy"], "CropRetained")
        self.assertEqual(winner["reference_index"], 1)
        self.assertEqual(final["references_before"][winner["source"]], 1)

    def test_recurrence_retains_older_canvas_and_exact_hidden_rgb8_rgb16(self):
        for dtype in (np.uint8, np.uint16):
            frames = recurrence(dtype)
            self.assertTrue(np.any(frames[1][..., :3][frames[1][..., 3] == 0]))
            for threads in (1, 2):
                with self.subTest(dtype=dtype, threads=threads):
                    data, report = encode_retained(frames, durations=[2, 3, 4, 5], threads=threads)
                    self.assertReports(report)
                    actual = rs_jxl.decode(data, frame=None)
                    self.assertEqual(len(actual), len(frames))
                    for index, (decoded, expected) in enumerate(zip(actual, frames)):
                        self.assertPixels(decoded, expected)
                        self.assertPixels(rs_jxl.decode(data, frame=index), expected)

    def test_zero_duration_reference_can_be_reused_after_a_later_frame(self):
        frames = recurrence()
        for durations, visible in (([0, 2, 0, 3], [1, 3]), ([0, 0, 0, 0], [3])):
            with self.subTest(durations=durations):
                data, report = encode_retained(frames, durations=durations)
                self.assertReports(report)
                actual = rs_jxl.decode(data, frame=None)
                self.assertEqual(len(actual), len(visible))
                for decoded, index in zip(actual, visible):
                    self.assertPixels(decoded, frames[index])

    def test_disabled_mode_roundtrips_and_reports_false(self):
        frames = recurrence()
        for auto_options in ({}, {"retained_reference": False}):
            with self.subTest(auto_options=auto_options):
                data, report = encode_retained(frames, auto_options=auto_options)
                self.assertTrue(all(not frame["retained_reference"] for frame in report["frames"]))
                actual = rs_jxl.decode(data, frame=None)
                self.assertEqual(len(actual), len(frames))
                for decoded, expected in zip(actual, frames):
                    self.assertPixels(decoded, expected)

    def test_direct_wire_rejects_bad_types_and_reference_conflicts(self):
        payload, images = multiframe._images([np.zeros((2, 3), dtype=np.uint8)])
        base = {"operation": "encode_multiframe", "images": images,
                "options": {"effort": 3}, "auto": True, "reference": "smallest_crop"}
        for value in (None, 1, 0, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(rs_jxl.CodecError):
                _ffi.execute({**base, "auto_options": {"retained_reference": value}}, payload.data)
        for reference in (None, "previous", "first", "full"):
            request = {**base, "auto_options": {"retained_reference": True}}
            if reference is None:
                del request["reference"]
            else:
                request["reference"] = reference
            with self.subTest(reference=reference), self.assertRaises(rs_jxl.CodecError):
                _ffi.execute(request, payload.data)

    def test_context_is_reusable_after_errors_and_does_not_leak_reference_state(self):
        options = {"strategy": "auto", "reference": "auto", "effort": 3,
                   "auto_options": {"retained_reference": True,
                                    "allow_mask": False, "allow_multi_rect": False}}
        with rs_jxl.Encoder(threads=2) as encoder:
            frames = recurrence()
            data, report = encoder.encode_multiframe(frames, return_info=True, **options)
            self.assertReports(report)
            self.assertPixels(rs_jxl.decode(data, frame=3), frames[-1])
            with self.assertRaises(ValueError):
                encoder.encode_multiframe(frames, **{**options, "reference": "first"})
            with self.assertRaises(rs_jxl.CodecError):
                encoder.encode_multiframe([frames[0], frames[1][:2]], **options)
            with self.assertRaises(rs_jxl.CodecError):
                encoder.encode_multiframe(frames, **{**options, "auto_options": {
                    "retained_reference": True, "max_trial_pixels": 0}})
            fresh = np.full((4, 5, 3), 17, dtype=np.uint8)
            data, report = encoder.encode_multiframe([fresh], return_info=True, **options)
            self.assertEqual(report["frames"][0]["references_before"], [None] * 4)
            self.assertPixels(rs_jxl.decode(data), fresh)
        with self.assertRaises(RuntimeError):
            encoder.encode_multiframe([fresh], **options)


class NativeReferenceLookahead(fixtures.PixelAssertions, unittest.TestCase):
    def assertCounters(self, report):
        for frame in report["frames"]:
            self.assertIs(frame["reference_lookahead"], True)
            self.assertIs(frame["retained_reference"], True)
            for field in ("lookahead_comparisons", "reused_payloads", "reused_payload_pixels"):
                self.assertIs(type(frame[field]), int)
                self.assertGreaterEqual(frame[field], 0)
            self.assertLessEqual(frame["lookahead_comparisons"], 3)
            submitted = sum(candidate["trial_pixels"]
                            for candidate in frame["candidates"] + frame["pruned"])
            self.assertLessEqual(frame["reused_payload_pixels"], submitted)
        self.assertEqual(report["frames"][-1]["lookahead_comparisons"], 0)

    def test_next_input_changes_retention_and_can_overwrite_delta_source(self):
        for dtype in (np.uint8, np.uint16):
            frames = lookahead_recurrence(dtype)
            for threads in (1, 2):
                with self.subTest(dtype=dtype, threads=threads):
                    _, old = encode_retained(frames, threads=threads)
                    data, report = encode_lookahead(frames, threads=threads)
                    self.assertCounters(report)
                    self.assertGreater(sum(f["lookahead_comparisons"] for f in report["frames"]), 0)
                    current = report["frames"][2]
                    self.assertNotIn(0, old["frames"][2]["references_after"])
                    self.assertIn(0, current["references_after"])
                    self.assertIn(2, current["references_after"])
                    winner = current["candidates"][current["winner_index"]]
                    self.assertEqual(winner["reference_index"], 1)
                    self.assertEqual(winner["source"], winner["save_as_reference"])
                    final = report["frames"][3]
                    self.assertEqual(final["candidates"][final["winner_index"]]["reference_index"], 0)
                    actual = rs_jxl.decode(data, frame=None)
                    self.assertEqual(len(actual), len(frames))
                    for index, expected in enumerate(frames):
                        self.assertPixels(actual[index], expected)
                        self.assertPixels(rs_jxl.decode(data, frame=index), expected)

    def test_multi_rectangle_source_overwrite_and_zero_durations_roundtrip(self):
        frames = lookahead_recurrence()
        frames[2][26, 26, 1] ^= 31
        for durations, visible in (([1, 2, 3, 4], [0, 1, 2, 3]), ([0, 2, 0, 3], [1, 3])):
            with self.subTest(durations=durations):
                data, report = encode_lookahead(
                    frames, durations=durations, patches=False,
                    auto_options={"allow_multi_rect": True, "tile_size": 4, "prune": False})
                self.assertCounters(report)
                current = report["frames"][2]
                winner = current["candidates"][current["winner_index"]]
                self.assertGreater(winner["encoded_frames"], 1)
                self.assertEqual(winner["source"], winner["save_as_reference"])
                self.assertIn(0, current["references_after"])
                actual = rs_jxl.decode(data, frame=None)
                self.assertEqual(len(actual), len(visible))
                for decoded, index in zip(actual, visible):
                    self.assertPixels(decoded, frames[index])

    def test_omitted_and_false_keep_old_policy_and_short_sequences_succeed(self):
        frames = lookahead_recurrence()
        for retained in (False, True):
            options = {"retained_reference": retained, "allow_mask": False, "allow_multi_rect": False}
            data, old = encode_retained(frames, auto_options=options)
            disabled, report = encode_retained(
                frames, auto_options={**options, "reference_lookahead": False})
            self.assertEqual(len(disabled), len(data))
            self.assertPixels(rs_jxl.decode(disabled, frame=3), frames[-1])
            for before, after in zip(old["frames"], report["frames"]):
                self.assertFalse(after["reference_lookahead"])
                self.assertEqual(after["lookahead_comparisons"], 0)
                for key in ("winner_index", "references_before", "references_after", "differences"):
                    self.assertEqual(before[key], after[key])
        for count in (1, 2):
            data, report = encode_lookahead(frames[:count])
            self.assertCounters(report)
            actual = rs_jxl.decode(data, frame=None)
            self.assertEqual(len(actual), count)
            for decoded, expected in zip(actual, frames[:count]):
                self.assertPixels(decoded, expected)

    def test_equal_future_geometry_keeps_previous_retention_policy(self):
        frames = lookahead_recurrence()
        # Invert both old canvases' red samples so every pixel differs equally.
        frames[-1] = frames[0].copy()
        frames[-1][..., 0] = frames[0][..., 0] ^ 255
        collision = frames[-1][..., 0] == frames[1][..., 0]
        frames[-1][..., 0][collision] ^= 1
        _, old = encode_retained(frames)
        data, report = encode_lookahead(frames)
        self.assertEqual(report["frames"][2]["references_after"],
                         old["frames"][2]["references_after"])
        self.assertPixels(rs_jxl.decode(data, frame=3), frames[-1])

    def test_exact_extra_only_changes_and_float_words_with_metadata(self):
        for dtype in (np.float16, np.float32):
            values = np.array([0.0, -0.0, 0.125, 1.0, -1.0, 2.0,
                               np.finfo(dtype).tiny, np.nextafter(dtype(0), dtype(1))], dtype=dtype)
            image = np.tile(values, (8, 1))
            frames = [image.copy() for _ in range(4)]
            frames[2][0, 0] = dtype(-0.0)
            depths = [np.full((8, 8), value, dtype=np.uint16) for value in (17, 60000, 60000, 17)]
            depths[2][2, 2] ^= 1
            depths[3][4, 4] ^= 1
            with self.subTest(dtype=dtype):
                data, report = encode_lookahead(
                    frames, color_encoding="linear_srgb", tps=(30000, 1001), loops=2,
                    extra_channels=[{"data": depths, "type": "depth", "name": "distance"}])
                self.assertCounters(report)
                actual, metadata = rs_jxl.decode(data, frame=None, return_extra_channels=True)
                self.assertEqual(len(actual), len(frames))
                for index, decoded in enumerate(actual):
                    expected = frames[index].astype(np.float32)[..., None]
                    self.assertPixels(decoded, expected)
                    np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
                    extra = metadata["frames"][index]["extra_channels"][0]
                    self.assertEqual((extra["type"], extra["name"]), ("depth", "distance"))
                    self.assertPixels(extra["data"], depths[index])
                header = rs_jxl.info(data)
                self.assertEqual(header["bits_per_sample"], np.dtype(dtype).itemsize * 8)
                self.assertEqual(header["exponent_bits_per_sample"], 5 if dtype == np.float16 else 8)
                self.assertEqual(header["animation"], {
                    "tps_numerator": 30000, "tps_denominator": 1001,
                    "num_loops": 2, "have_timecodes": False})

    def test_automatic_payload_reuse_keeps_trial_pixel_accounting(self):
        random = np.random.default_rng(81023)
        base = random.integers(0, 256, (16, 16), dtype=np.uint8)
        frames = [base, base.copy(), base.copy()]
        frames[1][2:14, 2:14] ^= np.uint8(17)
        frames[2][2:14, 2:14] ^= np.uint8(33)
        data, report = encode_lookahead(frames, patches=False, auto_options={"prune": False})
        self.assertCounters(report)
        final = report["frames"][-1]
        self.assertGreater(final["reused_payloads"], 0)
        self.assertEqual(final["reused_payload_pixels"], final["reused_payloads"] * 12 * 12)
        for candidate in final["candidates"]:
            expected_pixels = 16 * 16 if candidate["strategy"] == "Full" else 12 * 12
            self.assertEqual(candidate["trial_pixels"], expected_pixels)
        actual = rs_jxl.decode(data, frame=None)
        self.assertEqual(len(actual), len(frames))
        for decoded, expected in zip(actual, frames):
            self.assertPixels(decoded, expected)

    def test_direct_wire_validation_precedes_payload_access(self):
        base = {"operation": "encode_multiframe", "images": [{"length": 100}],
                "auto": True, "reference": "smallest_crop"}
        for value in (None, 0, 1, 1.0, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(rs_jxl.CodecError, "boolean"):
                _ffi.execute({**base, "auto_options": {"reference_lookahead": value}}, b"")
        for options in ({"reference_lookahead": True},
                        {"reference_lookahead": True, "retained_reference": False}):
            with self.subTest(options=options), self.assertRaisesRegex(
                    rs_jxl.CodecError, "reference_lookahead requires retained_reference"):
                _ffi.execute({**base, "auto_options": options}, b"")
        options = {"reference_lookahead": True, "retained_reference": True}
        for reference in (None, "first", "previous", "full"):
            request = {**base, "auto_options": options}
            if reference is None:
                del request["reference"]
            else:
                request["reference"] = reference
            with self.subTest(reference=reference), self.assertRaisesRegex(
                    rs_jxl.CodecError, "retained_reference requires reference=smallest_crop"):
                _ffi.execute(request, b"")
        with self.assertRaisesRegex(rs_jxl.CodecError, "auto_options require auto=true"):
            _ffi.execute({**base, "auto": False, "auto_options": options}, b"")

    def test_context_recovers_and_does_not_reuse_maps_or_payloads_across_calls(self):
        frames = lookahead_recurrence()
        options = {"strategy": "auto", "reference": "auto", "effort": 3,
                   "auto_options": {"retained_reference": True, "reference_lookahead": True,
                                    "allow_mask": False, "allow_multi_rect": False}}
        with rs_jxl.Encoder(threads=2) as encoder:
            data, report = encoder.encode_multiframe(frames, return_info=True, **options)
            self.assertCounters(report)
            self.assertPixels(rs_jxl.decode(data, frame=3), frames[-1])
            with self.assertRaises(ValueError):
                encoder.encode_multiframe(frames, **{**options, "auto_options": {"reference_lookahead": True}})
            with self.assertRaises(rs_jxl.CodecError):
                encoder.encode_multiframe([frames[0], frames[1][:2]], **options)
            fresh = np.full((4, 5, 3), 71, dtype=np.uint8)
            data, report = encoder.encode_multiframe([fresh], return_info=True, **options)
            self.assertCounters(report)
            first = report["frames"][0]
            self.assertEqual(first["references_before"], [None] * 4)
            self.assertEqual(first["reused_payloads"], 0)
            self.assertEqual(first["reused_payload_pixels"], 0)
            self.assertPixels(rs_jxl.decode(data), fresh)


@unittest.skipUnless("RS_JXL_DJXL" in os.environ, "official djxl is opt-in via RS_JXL_DJXL")
class OfficialRetainedReferences(fixtures.PixelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OfficialDecoderTests.setUpClass.__func__(cls)

    def test_recurring_and_mixed_duration_hidden_rgba(self):
        self.assertOfficialFrames(recurrence, encode_retained)

    def test_lookahead_and_mixed_duration_hidden_rgba(self):
        self.assertOfficialFrames(lookahead_recurrence, encode_lookahead)

    def assertOfficialFrames(self, make_frames, encode):
        for dtype in (np.uint8, np.uint16):
            frames = make_frames(dtype)
            for durations, visible in (([1, 2, 3, 4], [0, 1, 2, 3]), ([0, 2, 0, 3], [1, 3])):
                data, _ = encode(frames, durations=durations)
                expected = np.stack([frames[index] for index in visible])
                for decoder in self.decoders:
                    with self.subTest(dtype=dtype, durations=durations, decoder=decoder), \
                            tempfile.TemporaryDirectory() as directory:
                        destination = Path(directory) / "frames.npy"
                        fixtures.OfficialDecoderTests.runDecoder(
                            self, decoder, data, destination, directory)
                        actual = np.load(destination, allow_pickle=False)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.dtype, np.dtype("<f4"))
                        self.assertTrue(np.isfinite(actual).all())
                        self.assertTrue(np.all((actual >= 0) & (actual <= 1)))
                        maximum = np.iinfo(dtype).max
                        recovered = np.rint(actual.astype(np.float64) * maximum).astype(dtype)
                        self.assertPixels(recovered, expected)


if __name__ == "__main__":
    unittest.main()
