"""Source MA defaults use the existing encoder options mapping."""

import unittest

import numpy as np
import rs_jxl
from rs_jxl import _options


class SourceMaOptions(unittest.TestCase):
    def test_mapping_preserves_false_and_source_search(self):
        for value in (False, True):
            mapping = {"source_ma_defaults": value}
            actual = _options.encoder(9, 1, True, mapping, {})
            self.assertIs(actual["source_ma_defaults"], value)
            self.assertIs(actual["source_search"], True)
            self.assertEqual(mapping, {"source_ma_defaults": value})
            direct = _options.encoder(9, 1, False, None, mapping)
            self.assertIs(direct["source_ma_defaults"], value)

    def test_omission_and_duplicate_precedence(self):
        self.assertNotIn("source_ma_defaults", _options.encoder(7, 1, False, None, {}))
        with self.assertRaisesRegex(ValueError, "duplicate encoder option"):
            _options.encoder(9, 1, False, {"source_ma_defaults": False},
                             {"source_ma_defaults": True})


class NativeSourceMaDefaults(unittest.TestCase):
    def test_e9_still_animation_and_layers_preserve_integer_samples(self):
        for dtype in (np.uint8, np.uint16):
            words = np.arange(17 * 19 * 4, dtype=np.uint32).reshape(17, 19, 4)
            original = (words * 137 + 19).astype(dtype)
            original[::2, :, 3] = 0
            changed = original.copy()
            changed[3:7, 4:9, :3] ^= np.iinfo(dtype).max
            images = [original, changed]
            for enabled in (None, False, True):
                options = {} if enabled is None else {"source_ma_defaults": enabled}
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
