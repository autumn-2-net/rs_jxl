"""PNG export must inspect metadata beyond the selected pixel frame."""

from contextlib import ExitStack
import io
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np
import rs_jxl


PREFIX = bytes.fromhex("0000000c4a584c200d0a870a00000014667479706a786c20000000006a786c20")
XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'
# Brotli-compressed XMP above, quality 4. No optional Brotli test dependency.
BROTLI_XMP = bytes.fromhex(
    "1b24000044e7963aa1d03f34f53b1f513709944440859ad1e87ca9e28a4a3d8cfb2ee17ca48a6e3e01")
EXIF = b"\0\0\0\0II*\0\x08\0\0\0\0\0\0\0\0\0"


def _box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _split_container(encoded):
    if encoded.startswith(b"\xff\x0a"):
        return PREFIX, encoded
    header, chunks = [], []
    position = 0
    while position < len(encoded):
        size, kind = struct.unpack_from(">I4s", encoded, position)
        if not 8 <= size <= len(encoded) - position:
            raise AssertionError("expected small, complete boxes in generated fixture")
        payload = encoded[position + 8:position + size]
        if kind == b"jxlc":
            chunks.append(payload)
        elif kind == b"jxlp":
            chunks.append(payload[4:])
        else:
            header.append(encoded[position:position + size])
        position += size
    return b"".join(header), b"".join(chunks)


def _with_metadata(encoded, metadata, placement="after"):
    header, raw = _split_container(encoded)
    if placement == "before":
        return header + metadata + _box(b"jxlc", raw)
    if placement == "between":
        split = len(raw) // 2
        return (header + _box(b"jxlp", struct.pack(">I", 0) + raw[:split])
                + metadata + _box(b"jxlp", struct.pack(">I", 0x80000001) + raw[split:]))
    return header + _box(b"jxlc", raw) + metadata


def _png_icc(png):
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("expected PNG")
    position = 8
    while position < len(png):
        size, kind = struct.unpack_from(">I4s", png, position)
        payload = png[position + 8:position + 8 + size]
        if kind == b"iCCP":
            _, compressed = payload.split(b"\0", 1)
            if compressed[0] != 0:
                raise AssertionError("expected PNG zlib compression")
            return zlib.decompress(compressed[1:])
        position += size + 12
    raise AssertionError("missing output ICC")


class PngMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pixels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        cls.encoded = rs_jxl.encode(cls.pixels, effort=2, threads=1)

    def assertMetadataRejected(self, export, source, **kwargs):
        with self.assertRaisesRegex(rs_jxl.CodecError, "ancillary JXL boxes"):
            export(source, **kwargs)

    def test_known_boxes_before_between_and_after_codestream(self):
        for kind, payload in ((b"xml ", XMP), (b"Exif", EXIF),
                              (b"jumb", b"opaque JUMBF"), (b"jhgm", b"gain map")):
            for placement in ("before", "between", "after"):
                with self.subTest(kind=kind, placement=placement):
                    data = _with_metadata(self.encoded, _box(kind, payload), placement)
                    self.assertEqual(rs_jxl.info(data)["boxes"],
                                     [{"type": kind.decode("ascii"), "data": list(payload)}])
                    self.assertMetadataRejected(rs_jxl.decode_to_png, data)

    def test_compressed_extended_unbounded_and_empty_xmp(self):
        variants = [
            (_box(b"brob", b"xml " + BROTLI_XMP), XMP),
            (struct.pack(">I4sQ", 1, b"xml ", len(XMP) + 16) + XMP, XMP),
            (struct.pack(">I4s", 0, b"xml ") + XMP, XMP),
            (_box(b"xml ", b""), b""),
        ]
        for metadata, payload in variants:
            with self.subTest(metadata=metadata):
                data = _with_metadata(self.encoded, metadata)
                self.assertEqual(rs_jxl.info(data)["boxes"],
                                 [{"type": "xml ", "data": list(payload)}])
                self.assertMetadataRejected(rs_jxl.decode_to_png, data)

    def test_metadata_budget_is_cumulative_and_decompressed(self):
        data = _with_metadata(self.encoded, _box(b"brob", b"xml " + BROTLI_XMP)
                              + _box(b"xml ", XMP))
        with self.assertRaises(rs_jxl.CodecError) as error:
            rs_jxl.decode_to_png(data, limits={"max_metadata_bytes": len(XMP) * 2 - 1})
        self.assertEqual(error.exception.kind, "resource_limit")
        self.assertMetadataRejected(rs_jxl.decode_to_png, data,
                                    limits={"max_metadata_bytes": len(XMP) * 2})

    def test_truncated_tail_fails_export_but_targeted_pixels_stay_fast(self):
        for tail in (b"\0\0\0\x20xml <x/>",
                     b"\0\0\0\x10brobxml \xff\xff\xff\xff",
                     b"\0\0\0\x07xml "):
            with self.subTest(tail=tail):
                data = _with_metadata(self.encoded, tail)
                pixels, meta = rs_jxl.decode(data, return_info=True)
                np.testing.assert_array_equal(pixels, self.pixels)
                self.assertEqual(meta["boxes"], [])
                with self.assertRaises(rs_jxl.CodecError):
                    rs_jxl.decode_to_png(data)

    def test_all_input_forms_and_output_destinations_share_the_guard(self):
        data = _with_metadata(self.encoded, _box(b"xml ", XMP))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source = Path(directory) / "source.jxl"
            source.write_bytes(data)
            decoder = stack.enter_context(rs_jxl.Decoder(threads=2))
            for export in (rs_jxl.decode_to_png, decoder.decode_to_png):
                for make_input in (lambda: data, lambda: bytearray(data),
                                   lambda: memoryview(data), lambda: io.BytesIO(data),
                                   lambda: source):
                    with self.subTest(export=export.__qualname__, source=type(make_input())):
                        self.assertMetadataRejected(export, make_input())
                destination = Path(directory) / "existing.png"
                destination.write_bytes(b"keep original file")
                self.assertMetadataRejected(export, source, output=destination)
                self.assertEqual(destination.read_bytes(), b"keep original file")
                missing = Path(directory) / "not-created.png"
                self.assertMetadataRejected(export, source, output=missing)
                self.assertFalse(missing.exists())
                stream = io.BytesIO(b"keep original stream")
                self.assertMetadataRejected(export, data, output=stream)
                self.assertEqual(stream.getvalue(), b"keep original stream")
                self.assertEqual(stream.tell(), 0)
                restored = rs_jxl.decode(export(self.encoded))
                np.testing.assert_array_equal(restored, self.pixels)

    def test_selected_later_frame_and_raw_layer_cannot_hide_metadata(self):
        frames = [self.pixels, np.bitwise_xor(self.pixels, 7)]
        encoded = rs_jxl.encode_multiframe(frames, effort=2, reference="full")
        data = _with_metadata(encoded, _box(b"xml ", XMP))
        with rs_jxl.Decoder(threads=2) as decoder:
            for coalesced in (False, True):
                for export in (rs_jxl.decode_to_png, decoder.decode_to_png):
                    with self.subTest(export=export.__qualname__, coalesced=coalesced):
                        self.assertMetadataRejected(export, data, frame=1, coalesced=coalesced)
                        png = export(encoded, frame=1, coalesced=coalesced)
                        np.testing.assert_array_equal(rs_jxl.decode(png), frames[1])

    def test_metadata_free_png_preserves_icc_and_integer_samples(self):
        for dtype in (np.uint8, np.uint16):
            for channels in (1, 2, 3, 4):
                with self.subTest(dtype=dtype, channels=channels):
                    pixels = np.arange(6 * channels, dtype=dtype).reshape(2, 3, channels)
                    pixels = pixels * (257 if dtype == np.uint16 else 1)
                    encoded = rs_jxl.encode(pixels, effort=2)
                    info = rs_jxl.info(encoded)
                    png = rs_jxl.decode_to_png(encoded)
                    self.assertEqual(_png_icc(png), bytes(info["output_icc"]))
                    np.testing.assert_array_equal(rs_jxl.decode(png), pixels)


if __name__ == "__main__":
    unittest.main()
