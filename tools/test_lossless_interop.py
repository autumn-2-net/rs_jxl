"""Fixture/oracle checks need no native library or external CLI executables."""

import io
import struct
import unittest
import zlib

import numpy as np

from lossless_interop_cases import exact_pixels, jpeg_cases, pfm_bytes, pixel_cases, png_bytes
from verify_lossless_interop import jobs_for


class LosslessInteropTests(unittest.TestCase):
    def test_png_fixtures_preserve_all_sample_bytes_and_chunk_crcs(self):
        for case in pixel_cases():
            if case.pixels.dtype == np.float32:
                continue
            with self.subTest(case=case.name):
                data = png_bytes(case.pixels)
                chunks = {}
                offset = 8
                while offset < len(data):
                    size, kind = struct.unpack_from(">I4s", data, offset)
                    payload = data[offset + 8:offset + 8 + size]
                    crc, = struct.unpack_from(">I", data, offset + 8 + size)
                    self.assertEqual(zlib.crc32(kind + payload), crc)
                    chunks[kind] = payload
                    offset += size + 12
                self.assertEqual(offset, len(data))
                height, width, channels = case.pixels.shape
                row_bytes = width * channels * case.pixels.dtype.itemsize
                rows = zlib.decompress(chunks[b"IDAT"])
                recovered = b"".join(rows[y * (row_bytes + 1) + 1:(y + 1) * (row_bytes + 1)] for y in range(height))
                self.assertEqual(recovered, case.pixels.astype(">u2" if case.pixels.dtype == np.uint16 else "u1").tobytes())

    def test_float_oracle_detects_signed_zero_changes(self):
        expected = np.array([-0., 0., -2., 1e-10], dtype=np.float32).reshape(2, 2, 1)
        exact_pixels(expected[None], expected, normalized=True)
        wrong = expected.copy()
        wrong[0, 0, 0] = 0.
        with self.assertRaises(AssertionError):
            exact_pixels(wrong[None], expected, normalized=True)
        pfm = pfm_bytes(expected)
        self.assertTrue(pfm.startswith(b"Pf\n2 2\n-1.0\n"))
        self.assertEqual(pfm.split(b"\n", 3)[3], expected[::-1].tobytes())

    def test_u16_oracle_detects_single_low_bit_and_hidden_color_change(self):
        expected = next(case.pixels for case in pixel_cases() if case.name == "rgba16_edges")
        normalized = expected.astype(np.float32)[None] / np.float32(65535)
        exact_pixels(normalized, expected, normalized=True)
        wrong = expected.copy()
        wrong[0, 0, 0] ^= 1
        with self.assertRaises(AssertionError):
            exact_pixels(wrong.astype(np.float32)[None] / np.float32(65535), expected, normalized=True)
        with self.assertRaises(AssertionError):
            exact_pixels(normalized[..., :3], expected, normalized=True)
        with self.assertRaises(ValueError):
            exact_pixels(np.zeros((1, 1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1), dtype=np.uint32), normalized=True)

    def test_generated_jpegs_are_real_files_with_preserved_ancillary_bytes(self):
        from PIL import Image
        for name, data in jpeg_cases():
            with self.subTest(name=name):
                self.assertTrue(data.endswith(b"\0trailing archival bytes\xff"))
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    self.assertEqual(image.size, (35, 33))

    def test_effort_matrix_does_not_replace_low_efforts_with_nine(self):
        case = {"case": "example", "kind": "pixels", "floating": False}
        jobs = jobs_for([case], ["official-0", "official-1"], range(1, 11), False)
        self.assertEqual(len(jobs), 30)
        self.assertEqual({job["effort"] for job in jobs}, set(range(1, 11)))
        self.assertEqual({job["threads"] for job in jobs}, {1, 4})
        self.assertTrue(all(job["variant"] == "default" for job in jobs))

    def test_uint16_small_groups_cross_both_group_boundaries(self):
        case = next(case for case in pixel_cases() if case.name == "rgba16_groups")
        self.assertEqual(case.pixels.dtype, np.uint16)
        self.assertGreater(case.pixels.shape[0], 128)
        self.assertGreater(case.pixels.shape[1], 128)
        jobs = jobs_for([{"case": case.name, "floating": False}],
                        ["official-0", "official-1"], [9], True)
        variants = [job for job in jobs if job["variant"] == "weighted_small_groups"]
        self.assertEqual(len(variants), 3)
        self.assertTrue(all(job["options"]["group_size_shift"] == 0 for job in variants))


if __name__ == "__main__":
    unittest.main()
