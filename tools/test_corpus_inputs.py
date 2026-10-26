"""Synthetic input preparation tests only; no builds, encoders or benchmarks."""

from io import BytesIO
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from PIL import Image, JpegImagePlugin

import compare_cpp_port as port
import corpus_inputs as inputs


def chunk(kind, body):
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body)))


def png(depth, color, rows, width, extra=(), interlace=False, filtering=0):
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    header = struct.pack(">IIBBBBB", width, len(rows), depth, color, 0, 0, int(interlace))
    passes = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
              (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)) if interlace else ((0, 0, 1, 1),)
    raw = bytearray()
    for x0, y0, dx, dy in passes:
        if x0 >= width:
            continue
        previous = None
        for y in range(y0, len(rows), dy):
            samples = [value for x in range(x0, width, dx)
                       for value in rows[y][x * channels:(x + 1) * channels]]
            if depth < 8:
                row = bytearray((len(samples) * depth + 7) // 8)
                for index, value in enumerate(samples):
                    row[index * depth // 8] |= value << (8 - depth - index * depth % 8)
            else:
                row = struct.pack(">" + ("B" if depth == 8 else "H") * len(samples), *samples)
            encoded = bytearray()
            bpp = max(1, channels * depth // 8)
            for index, value in enumerate(row):
                left = row[index - bpp] if index >= bpp else 0
                above = previous[index] if previous is not None else 0
                corner = previous[index - bpp] if previous is not None and index >= bpp else 0
                prediction = left + above - corner
                distances = [abs(prediction - candidate) for candidate in (left, above, corner)]
                paeth = (left, above, corner)[distances.index(min(distances))]
                predictor = (0, left, above, (left + above) // 2, paeth)[filtering]
                encoded.append((value - predictor) & 255)
            raw.extend(bytes([filtering]) + encoded)
            previous = row
    return (port.PNG_SIGNATURE + chunk(b"IHDR", header)
            + b"".join(chunk(kind, body) for kind, body in extra)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class CorpusInputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="corpus-input-test-", dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "original.dat"
        self.output = self.root / "snapshots"
        self.output.mkdir()

    def prepare(self, data):
        self.source.write_bytes(data)
        info, snapshot, expected = inputs.prepare(self.source, self.output)
        self.assertEqual(self.source.read_bytes(), data)
        self.assertEqual(snapshot.parent, self.output.resolve())
        self.assertNotEqual(snapshot.resolve(), self.source.resolve())
        self.assertEqual(info["input_sha256"], port.sha256(data))
        self.assertEqual(info["payload_sha256"], port.file_sha256(snapshot))
        self.assertEqual(info["input_bytes"], len(data))
        self.assertEqual(info["pixels"], info["width"] * info["height"])
        if info["kind"] == "png":
            self.assertIsInstance(expected, port.Pixels)
            self.assertEqual(info["depth"], expected.depth)
            port.compare_pixels(expected, port.png_samples(snapshot.read_bytes()))
        return info, snapshot, expected

    def reject(self, data, error=ValueError, reason=None):
        self.source.write_bytes(data)
        before = set(self.output.iterdir())
        with self.assertRaises(error) as caught:
            inputs.prepare(self.source, self.output)
        if error is ValueError:
            self.assertNotIsInstance(caught.exception, inputs.UnsupportedInput)
        if reason:
            self.assertIn(reason, str(caught.exception))
        self.assertEqual(set(self.output.iterdir()), before)
        self.assertEqual(self.source.read_bytes(), data)

    def test_native_png_all_channels_and_depths_no_pillow(self):
        for depth in (8, 16):
            top = (1 << depth) - 1
            a, b, c = (19, 81, 253) if depth == 8 else (259, 513, 65534)
            for color, samples, expected in (
                (0, [a, c], [[a, a, a, top], [c, c, c, top]]),
                (2, [a, b, c, c, b, a], [[a, b, c, top], [c, b, a, top]]),
                (4, [a, 0, c, top], [[a, a, a, 0], [c, c, c, top]]),
                (6, [a, b, c, 0, c, b, a, top], [[a, b, c, 0], [c, b, a, top]]),
            ):
                with self.subTest(depth=depth, color=color), patch.object(
                        Image, "open", side_effect=AssertionError("Pillow must not read normal PNG")):
                    data = png(depth, color, [samples], 2)
                    _, snapshot, actual = self.prepare(data)
                    self.assertEqual(snapshot.read_bytes(), data)
                    self.assertEqual(actual.rgba.tolist(), [expected])

    def test_metadata_removed_but_idat_and_samples_preserved(self):
        rows = [[259, 513, 65534, 0, 1, 257, 32769, 65535]]
        extra = [(b"gAMA", struct.pack(">I", 45455)), (b"tEXt", b"key\0value"),
                 (b"sBIT", bytes([12, 12, 12, 12]))]
        info, snapshot, _ = self.prepare(png(16, 6, rows, 2, extra))
        self.assertEqual(info["removed_metadata_chunks"], ["gAMA", "tEXt", "sBIT"])
        self.assertEqual(snapshot.read_bytes(), png(16, 6, rows, 2))
        self.assertNotEqual(info["input_sha256"], info["payload_sha256"])

    def test_packed_gray_depths_filters_adam7_and_transparency(self):
        for depth in (1, 2, 4):
            top = (1 << depth) - 1
            rows = [[(x + y) % (top + 1) for x in range(9)] for y in range(5)]
            for interlace in (False, True):
                for filtering in range(5):
                    with self.subTest(depth=depth, interlace=interlace, filtering=filtering):
                        data = png(depth, 0, rows, 9, [(b"tRNS", struct.pack(">H", 1)),
                                   (b"tEXt", b"key\0value")], interlace, filtering)
                        info, _, expected = self.prepare(data)
                        self.assertEqual(info["depth"], 8)
                        self.assertIn(f"packed{depth}", info["conversion"])
                        self.assertEqual(info["removed_metadata_chunks"], ["tEXt"])
                        self.assertEqual(expected.rgba.tolist(), [
                            [[v * 255 // top] * 3 + [0 if v == 1 else 255] for v in row] for row in rows])

    def test_palette_depths_keep_hidden_rgb_and_partial_trns(self):
        palette = bytes([19, 81, 253, 7, 11, 13])
        for depth in (1, 2, 4, 8):
            for interlace in (False, True):
                with self.subTest(depth=depth, interlace=interlace):
                    _, _, expected = self.prepare(png(depth, 3, [[0, 1, 0]], 3,
                        [(b"PLTE", palette), (b"tRNS", b"\0")], interlace))
                    self.assertEqual(expected.rgba.tolist(),
                        [[[19, 81, 253, 0], [7, 11, 13, 255], [19, 81, 253, 0]]])

    def test_tiny_one_pixel_packed_png_regression(self):
        for depth in (1, 2, 4):
            for color in (0, 3):
                with self.subTest(depth=depth, color=color):
                    extra = [(b"PLTE", bytes([19, 81, 253]))] if color == 3 else []
                    info, _, expected = self.prepare(png(depth, color, [[0]], 1, extra))
                    self.assertEqual((info["width"], info["height"], info["depth"]), (1, 1, 8))
                    self.assertEqual(expected.rgba.tolist(),
                                     [[[19, 81, 253, 255]]] if color == 3 else [[[0, 0, 0, 255]]])

    def test_png_transparency_keys_at_native_depth(self):
        for depth, value in ((8, 19), (16, 259)):
            for color, row, key in ((0, [value, 0], [value]),
                                    (2, [value, 1, 2, 0, 0, 0], [value, 1, 2])):
                with self.subTest(depth=depth, color=color):
                    _, _, expected = self.prepare(png(depth, color, [row], 2,
                        [(b"tRNS", struct.pack(">" + "H" * len(key), *key))]))
                    self.assertEqual(int(expected.rgba[0, 0, 0]), value)
                    self.assertEqual(int(expected.rgba[0, 0, 3]), 0)

    def test_jpeg_bytes_dimensions_without_decode(self):
        for progressive in (False, True):
            buffer = BytesIO()
            Image.new("RGB", (3, 2), (19, 81, 253)).save(buffer, "JPEG", progressive=progressive)
            data = buffer.getvalue() + b"preserved trailing bytes"
            with patch.object(JpegImagePlugin.JpegImageFile, "load", side_effect=AssertionError("No JPEG pixels")), \
                    patch.object(port, "pixel_libraries", side_effect=AssertionError("No pixel libraries")):
                info, snapshot, expected = self.prepare(data)
            self.assertEqual(expected, data)
            self.assertEqual(snapshot.read_bytes(), data)
            self.assertEqual((info["kind"], info["width"], info["height"], info["depth"]), ("jpeg", 3, 2, None))
            self.assertEqual(info["input_sha256"], info["payload_sha256"])
            self.assertEqual(info["removed_metadata_chunks"], [])

    def test_truncated_jpeg(self):
        buffer = BytesIO()
        Image.new("RGB", (2, 1)).save(buffer, "JPEG")
        data = buffer.getvalue()
        for bad in (data[:-2], data[:30], b"\xff\xd8\xff\xe0\x00\x01", b"\xff\xd8\xff\xd9"):
            with self.subTest(size=len(bad)):
                self.reject(bad)

    def test_jpeg_dimension_limits_without_pixel_decode(self):
        buffer = BytesIO()
        Image.new("RGB", (1, 1)).save(buffer, "JPEG")
        original = buffer.getvalue()
        sof = original.index(b"\xff\xc0")
        for width, height in ((16385, 1), (8001, 8000), (65535, 65535)):
            data = bytearray(original)
            data[sof + 5:sof + 9] = struct.pack(">HH", height, width)
            with self.subTest(width=width, height=height), patch.object(
                    JpegImagePlugin.JpegImageFile, "load", side_effect=AssertionError("No JPEG pixels")):
                self.reject(bytes(data), inputs.UnsupportedInput, "limits")

    def test_p5_p6_p7_every_integer_channel_and_depth(self):
        _, np = port.pixel_libraries()
        for depth in (8, 16):
            top = (1 << depth) - 1
            for channels in (1, 2, 3, 4):
                samples = np.array([[[19, 81, 253, 0][:channels], [7, 11, 13, top][:channels]]],
                                   dtype=np.uint8 if depth == 8 else np.uint16)
                if depth == 16:
                    samples[0, 0, 0] = 259
                    samples[0, 1, 0] = 32769
                if channels == 2:
                    samples[0, 0, 1] = 0
                raster = samples.astype("u1" if depth == 8 else ">u2").tobytes()
                tuple_name = ("GRAYSCALE", "GRAYSCALE_ALPHA", "RGB", "RGB_ALPHA")[channels - 1]
                headers = [f"P7\n# comment\nWIDTH 2\nHEIGHT 1\nDEPTH {channels}\nMAXVAL {top}\nTUPLTYPE {tuple_name}\nENDHDR\n"]
                if channels in (1, 3):
                    headers.append(f"P{5 if channels == 1 else 6}\n# comment\n2\t1\n{top}\n")
                for header in headers:
                    with self.subTest(depth=depth, channels=channels, magic=header[:2]):
                        info, _, expected = self.prepare(header.encode() + raster)
                        self.assertEqual(expected.depth, depth)
                        self.assertEqual(expected.rgba[:, :, 0].tolist(), samples[:, :, 0].tolist())
                        if channels <= 2:
                            self.assertTrue(np.all(expected.rgba[:, :, :3] == samples[:, :, :1]))
                        else:
                            self.assertTrue(np.array_equal(expected.rgba[:, :, :3], samples[:, :, :3]))
                        alpha = samples[:, :, -1] if channels in (2, 4) else top
                        self.assertTrue(np.all(expected.rgba[:, :, 3] == alpha))
                        if channels == 2:
                            self.assertIn("grayalpha-to-rgba", info["conversion"])

    def test_netpbm_first_raster_byte_is_not_skipped(self):
        for separator in (b"\n", b"\r\n", b" "):
            for first in (0, 9, 10, 13, 32, 35, 255):
                with self.subTest(separator=separator, first=first):
                    _, _, expected = self.prepare(b"P5\n1 1\n255" + separator + bytes([first]))
                    self.assertEqual(int(expected.rgba[0, 0, 0]), first)

    def test_unsupported_formats_and_limits_are_not_success(self):
        for data, reason in ((b"PF\n1 1\n-1.0\n", "PFM"),
                             (b"P5\n1 1\n100\n\0", "MAXVAL"),
                             (b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 5\nMAXVAL 255\nENDHDR\n", "channels"),
                             (b"P5\n16385 1\n255\n", "Dimensions"),
                             (b"P5\n8001 8000\n255\n", "Dimensions"),
                             (b"P3\n1 1\n255\n0 0 0", "format")):
            with self.subTest(reason=reason):
                self.reject(data, inputs.UnsupportedInput, reason)
        valid = png(1, 0, [[0]], 1)
        apng = valid[:33] + chunk(b"acTL", struct.pack(">II", 1, 0)) + valid[33:]
        self.reject(apng, inputs.UnsupportedInput, "APNG")
        oversized = (port.PNG_SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 16385, 1, 1, 0, 0, 0, 0))
                     + valid[33:])
        self.reject(oversized, inputs.UnsupportedInput, "Dimensions")
        with patch.object(inputs, "MAX_INPUT_BYTES", 10):
            self.reject(valid, inputs.UnsupportedInput, "512 MiB")

    def test_bad_png_framing_never_reaches_pillow(self):
        valid = png(2, 0, [[0, 1]], 2)
        broken_crc = bytearray(valid)
        broken_crc[29] ^= 1
        header = valid[:33]
        tail = valid[33:]
        bad = [b"", valid[:5], valid[:-1], valid + b"extra", bytes(broken_crc), header + header[8:] + tail,
               header + chunk(b"tRNS", b"\0\1") + chunk(b"tRNS", b"\0\1") + tail,
               header + chunk(b"ABCD", b"") + tail,
               port.PNG_SIGNATURE + tail,
               png(2, 3, [[0, 1]], 2),
               png(2, 3, [[0, 1]], 2, [(b"tRNS", b"\0"), (b"PLTE", b"\0" * 6)]),
               png(2, 3, [[0, 1]], 2, [(b"PLTE", b"\0" * 4)]),
               header + chunk(b"IDAT", b"") + chunk(b"tEXt", b"k\0v") + tail,
               header + chunk(b"IDAT", b"broken zlib") + chunk(b"IEND", b"")]
        with patch.object(Image, "open", side_effect=AssertionError("Framing must be checked first")):
            for data in bad:
                with self.subTest(data=data[:40]):
                    self.reject(data)

    def test_bad_netpbm_header_and_raster(self):
        for data in (b"P5\n1 1\n255\n", b"P6\n1 1\n65535\n\0",
                     b"P5\n1 1\n255\n\0trailing", b"P5\n0 1\n255\n", b"P5\n1 x\n255\n",
                     b"P5\n1 1\n0\n", b"P5\n1 1\n65536\n", b"P7\nWIDTH 1\n",
                     b"P7\nWIDTH 1\nWIDTH 2\nENDHDR\n",
                     b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 1\nMAXVAL 255\nTUPLTYPE RGB\nENDHDR\n\0"):
            with self.subTest(data=data):
                self.reject(data)

    def test_converted_samples_checked_against_original(self):
        cv2, np = port.pixel_libraries()
        ok, wrong = cv2.imencode(".png", np.array([[18]], dtype=np.uint8))
        self.assertTrue(ok)
        with patch.object(cv2, "imencode", return_value=(True, wrong)):
            self.reject(b"P5\n1 1\n255\n\x13", reason="Pixel mismatch")
        wrong_png = png(8, 6, [[0, 0, 0, 0]], 1)
        with patch.object(Image.Image, "save", side_effect=lambda output, **kwargs: output.write(wrong_png)):
            self.reject(png(1, 0, [[1]], 1, [(b"tRNS", b"\0\1")]), reason="Pixel mismatch")

    def test_snapshots_unique_even_when_input_is_in_output_directory(self):
        self.source = self.output / "input.png"
        data = png(8, 0, [[19]], 1)
        _, first, _ = self.prepare(data)
        _, second, _ = self.prepare(data)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
