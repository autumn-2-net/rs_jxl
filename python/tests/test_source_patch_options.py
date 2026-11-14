"""Source Patch parallel gather uses presence-aware encoder options."""

import unittest

import numpy as np
import rs_jxl
from rs_jxl import _options
from rs_jxl._ffi import execute


OPTION = "source_patch_parallel_gather"


class SourcePatchOptions(unittest.TestCase):
    def test_mapping_preserves_bools_and_previous_options(self):
        for value in (False, True):
            mapping = {OPTION: value, "source_ma_defaults": False, "patches": True}
            for actual in (
                _options.encoder(9, 2, True, mapping, {}),
                _options.encoder(9, 2, True, None, mapping),
            ):
                self.assertIs(actual[OPTION], value)
                self.assertIs(actual["source_ma_defaults"], False)
                self.assertIs(actual["patches"], True)
                self.assertIs(actual["source_search"], True)
                self.assertEqual(actual["threads"], 2)
            self.assertEqual(mapping, {OPTION: value, "source_ma_defaults": False,
                                       "patches": True})

    def test_omission_and_duplicate_precedence(self):
        self.assertNotIn(OPTION, _options.encoder(7, 1, False, None, {}))
        with self.assertRaisesRegex(ValueError, "duplicate encoder option"):
            _options.encoder(9, 2, False, {OPTION: False}, {OPTION: True})


class NativeSourcePatchOptions(unittest.TestCase):
    def test_still_animation_and_layers_preserve_integer_samples(self):
        for dtype in (np.uint8, np.uint16):
            words = np.arange(17 * 19 * 4, dtype=np.uint32).reshape(17, 19, 4)
            original = (words * 137 + 19).astype(dtype)
            original[::2, :, 3] = 0
            changed = original.copy()
            changed[3:7, 4:9, :3] ^= np.iinfo(dtype).max
            images = [original, changed]
            for enabled in (None, False, True):
                options = {} if enabled is None else {OPTION: enabled}
                with self.subTest(dtype=dtype, enabled=enabled):
                    still = rs_jxl.encode(original, effort=9, threads=2, **options)
                    np.testing.assert_array_equal(rs_jxl.decode(still), original)
                    sequence = rs_jxl.encode_multiframe(
                        images, effort=9, threads=2, strategy="crop", **options)
                    actual = rs_jxl.decode(sequence, frame=None)
                    self.assertEqual(len(actual), 2)
                    for decoded, expected in zip(actual, images):
                        np.testing.assert_array_equal(decoded, expected)
                    layers = rs_jxl.encode_layers(images, effort=9, threads=2, **options)
                    for index, expected in enumerate(images):
                        decoded, _ = rs_jxl.decode_layer(layers, layer=index)
                        np.testing.assert_array_equal(decoded, expected)

    def test_wrong_types_and_null_are_rejected_by_native_json(self):
        original = np.zeros((1, 1, 1), dtype=np.uint8)
        for value in (None, 0, 1, 1.0, "true", "false", [], {}):
            with self.subTest(value=value), self.assertRaises(rs_jxl.CodecError):
                rs_jxl.encode(original, encoder_options={OPTION: value})

    def test_explicit_bools_are_rejected_for_irrelevant_operations(self):
        for operation in ("transcode_jpeg", "reconstruct_jpeg", "decode",
                          "decode_to_png", "info", "read_image", "analyze_multiframe"):
            for value in (False, True):
                request = {"operation": operation, "options": {OPTION: value}}
                if operation == "analyze_multiframe":
                    request["images"] = [{"length": 0}]
                with self.subTest(operation=operation, value=value), \
                        self.assertRaisesRegex(rs_jxl.CodecError, "unsupported"):
                    execute(request, b"")

    def test_jpeg_encode_dispatch_rejects_pixel_only_option(self):
        for value in (False, True):
            with self.subTest(value=value), \
                    self.assertRaisesRegex(rs_jxl.CodecError, "unsupported"):
                rs_jxl.encode(b"\xff\xd8", encoder_options={OPTION: value})
