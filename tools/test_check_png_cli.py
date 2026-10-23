"""Checks for the optional Pillow-based acceptance helper, not the codec."""

from io import BytesIO
import struct
import unittest
import zlib

from PIL import Image

from check_png_cli import png_samples


def png16(color_type, samples):
    def chunk(kind, content):
        return (struct.pack(">I", len(content)) + kind + content
                + struct.pack(">I", zlib.crc32(kind + content)))

    header = struct.pack(">IIBBBBB", 1, 1, 16, color_type, 0, 0, 0)
    scanline = b"\0" + struct.pack(">" + "H" * len(samples), *samples)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(scanline)) + chunk(b"IEND", b""))


class PngSamplesTests(unittest.TestCase):
    def test_hidden_rgb_is_not_discarded(self):
        expected = bytes((19, 81, 253, 0, 7, 11, 13, 255))
        image = Image.frombytes("RGBA", (2, 1), expected)
        data = BytesIO()
        image.save(data, format="PNG")
        self.assertEqual(png_samples(data.getvalue()), ((2, 1), "RGBA", expected))

    def test_rejects_silent_pillow_sixteen_bit_narrowing(self):
        for kind, samples in [(0, [257]), (2, [257, 513, 769]),
                              (4, [257, 1025]), (6, [257, 513, 769, 1025])]:
            with self.subTest(color_type=kind), self.assertRaisesRegex(ValueError, "16-bit"):
                png_samples(png16(kind, samples))

    def test_rejects_missing_header(self):
        for data in (b"", b"\x89PNG\r\n\x1a\n", b"not a PNG"):
            with self.assertRaises(ValueError):
                png_samples(data)


if __name__ == "__main__":
    unittest.main()
