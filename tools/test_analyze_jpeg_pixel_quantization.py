"""Quantizer diagnostics do not invoke a codec in these tests."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import analyze_jpeg_pixel_quantization as diagnostic


class QuantizationTests(unittest.TestCase):
    def test_clipping_and_nearest_even(self):
        samples = np.array([-4, 0, .5, 1.5, 2.5, 254.5, 255, 300], np.float32)
        pixels = (samples / np.float32(255)).reshape(1, -1, 1)
        actual = diagnostic.quantize(pixels)
        np.testing.assert_array_equal(actual.ravel(), [0, 0, 0, 2, 2, 254, 255, 255])

    def test_channel_offsets_and_wraparound(self):
        table = (np.arange(1024, dtype=np.float32).reshape(32, 32) / 1024 - .5)
        pixels = np.full((35, 37, 3), np.float32(.5))
        actual = diagnostic.quantize(pixels, table)
        for y, x in [(0, 0), (0, 31), (1, 32), (32, 34), (34, 36)]:
            for c in range(3):
                expected = 127 if table[(y + c * 13) % 32, (x + c * 23) % 32] < 0 else 128
                self.assertEqual(actual[y, x, c], expected)

    def test_reference_table_removes_only_cpp_padding(self):
        values = np.arange(1024, dtype=np.float32).reshape(32, 32)
        padded = np.concatenate([values, values[:, :16]], axis=1)
        source = "const float kDither[48 * 32] = {" + ",".join(map(str, padded.ravel())) + ",};"
        with patch.object(diagnostic.subprocess, "run", return_value=SimpleNamespace(stdout=source)):
            table, digest = diagnostic.reference_table("unused")
        np.testing.assert_array_equal(table, values)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
