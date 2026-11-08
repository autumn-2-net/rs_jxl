"""Real-native decode failures must not poison reusable decoder state."""

import struct
import unittest

import numpy as np
import rs_jxl


PREFIX = bytes.fromhex(
    "0000000c4a584c200d0a870a00000014667479706a786c20000000006a786c20")
XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'
# Same fixed quality-4 Brotli fixture as test_png_metadata; no optional dependency.
BROTLI_XMP = bytes.fromhex(
    "1b24000044e7963aa1d03f34f53b1f513709944440859ad1e87ca9e28a4a3d8cfb2ee17ca48a6e3e01")


def _box(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _codestream(encoded):
    if encoded.startswith(b"\xff\x0a"):
        return encoded
    chunks = []
    position = 0
    while position < len(encoded):
        size, kind = struct.unpack_from(">I4s", encoded, position)
        if not 8 <= size <= len(encoded) - position:
            raise AssertionError("expected bounded small boxes in encoded fixture")
        payload = encoded[position + 8:position + size]
        if kind == b"jxlc":
            chunks.append(payload)
        elif kind == b"jxlp":
            chunks.append(payload[4:])
        position += size
    raw = b"".join(chunks)
    if not raw.startswith(b"\xff\x0a"):
        raise AssertionError("missing generated codestream")
    return raw


class DecodeRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Do not skip a missing DLL: every substantive operation uses the codec.
        base = (np.arange(36, dtype=np.uint16) * 1543).reshape(3, 4, 3)
        base[0, 0] = [0, 32768, 65535]
        cls.frames = [base.copy() for _ in range(3)]
        cls.frames[1][1, 2] = [65535, 1, 32767]
        cls.frames[2] = cls.frames[1].copy()
        cls.frames[2][2, 3] = [32768, 65534, 0]
        depth = (np.arange(12, dtype=np.uint16) * 4097).reshape(3, 4)
        cls.depths = [depth.copy() for _ in cls.frames]
        cls.depths[1][1, 2] = 65535
        cls.depths[2] = cls.depths[1].copy()
        cls.depths[2][0, 1] = 32768
        cls.sequence = rs_jxl.encode_multiframe(
            cls.frames, durations=[2, 3, 5], reference="previous", strategy="crop",
            extra_channels=[{"data": cls.depths, "type": "depth", "name": "distance"}],
            effort=3, threads=1)
        cls.still = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        cls.encoded_still = rs_jxl.encode(cls.still, effort=3, threads=1)

    def assertPixels(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.flags.writeable)
        self.assertTrue(actual.dtype.isnative)

    def assertCodecFailure(self, kind, operation):
        with self.assertRaises(rs_jxl.CodecError) as failure:
            operation()
        self.assertEqual(failure.exception.kind, kind)

    def assertSequence(self, decoder):
        frames, metadata = decoder.decode(
            self.sequence, frame=None, return_extra_channels=True)
        self.assertTrue(metadata["complete"])
        self.assertEqual(len(frames), 3)
        self.assertEqual(len(metadata["frames"]), 3)
        self.assertEqual([entry["duration"] for entry in metadata["frames"]], [2, 3, 5])
        retained = []
        for index, (actual, entry) in enumerate(zip(frames, metadata["frames"])):
            self.assertTrue(entry["complete"])
            self.assertPixels(actual, self.frames[index])
            self.assertEqual(len(entry["extra_channels"]), 1)
            extra = entry["extra_channels"][0]
            self.assertEqual((extra["type"], extra["name"]), ("depth", "distance"))
            expected_depth = self.depths[index][..., None]
            self.assertPixels(extra["data"], expected_depth)
            retained.extend([(actual, self.frames[index]), (extra["data"], expected_depth)])
        for index, (array, _) in enumerate(retained):
            for other, _ in retained[index + 1:]:
                self.assertFalse(np.shares_memory(array, other))
        return retained

    def assertReusable(self, decoder, retained):
        self.assertFalse(decoder.closed)
        self.assertPixels(decoder.decode(self.encoded_still), self.still)
        current = self.assertSequence(decoder)
        for old, expected in retained:
            self.assertPixels(old, expected)
            for new, _ in current:
                self.assertFalse(np.shares_memory(old, new))

    def assertRetained(self, retained):
        for actual, expected in retained:
            self.assertPixels(actual, expected)

    def test_truncated_pixels_and_decode_into_are_recoverable(self):
        for threads in (1, 2):
            with self.subTest(threads=threads), rs_jxl.Decoder(threads=threads) as decoder:
                retained = self.assertSequence(decoder)
                # Keep the original container and its declared profile level.
                for encoded, frame, expected in ((self.encoded_still, 0, self.still),
                                                 (self.sequence, 2, self.frames[2])):
                    truncated = encoded[:len(encoded) // 2]
                    for selection in (None, frame):
                        with self.subTest(frame=selection, size=len(encoded)):
                            self.assertCodecFailure(
                                "decoding", lambda: decoder.decode(truncated, frame=selection))
                            self.assertReusable(decoder, retained)
                    destination = np.full_like(expected, 71)
                    self.assertCodecFailure(
                        "decoding", lambda: decoder.decode_into(
                            truncated, destination, frame=frame))
                    np.testing.assert_array_equal(destination, np.full_like(expected, 71))
                    self.assertIs(decoder.decode_into(encoded, destination, frame=frame), destination)
                    self.assertPixels(destination, expected)
                    self.assertReusable(decoder, retained)
            self.assertRetained(retained)

    def test_unavailable_frame_and_layer_selection_are_recoverable(self):
        for threads in (1, 2):
            with self.subTest(threads=threads), rs_jxl.Decoder(threads=threads) as decoder:
                retained = self.assertSequence(decoder)
                for operation in (
                        lambda: decoder.decode(self.sequence, frame=3),
                        lambda: decoder.decode_layer(self.sequence, layer=3)):
                    self.assertCodecFailure("decoding", operation)
                    self.assertReusable(decoder, retained)
                selected, metadata = decoder.decode(
                    self.sequence, frame=2, return_extra_channels=True)
                self.assertPixels(selected, self.frames[2])
                extra = metadata["extra_channels"][0]["data"]
                retained.extend([(selected, self.frames[2]),
                                 (extra, self.depths[2][..., None])])
                self.assertReusable(decoder, retained)
            self.assertRetained(retained)

    def test_metadata_failures_and_intentionally_early_selected_decode(self):
        raw = _codestream(self.encoded_still)
        container = PREFIX + _box(b"jxlc", raw)
        # Valid pixels followed by a bounded XMP box whose payload is truncated.
        truncated_tail = container + _box(b"xml ", XMP)[:-1]
        compressed = container + _box(b"brob", b"xml " + BROTLI_XMP)
        for threads in (1, 2):
            with self.subTest(threads=threads), rs_jxl.Decoder(threads=threads) as decoder:
                retained = self.assertSequence(decoder)
                for operation in (
                        lambda: decoder.info(truncated_tail),
                        lambda: decoder.decode(truncated_tail, frame=None),
                        lambda: decoder.decode_to_png(truncated_tail)):
                    self.assertCodecFailure("decoding", operation)
                    self.assertReusable(decoder, retained)
                # A complete selected frame is not validation of later boxes.
                selected, metadata = decoder.decode(truncated_tail, return_info=True)
                self.assertPixels(selected, self.still)
                self.assertTrue(metadata["complete"])
                self.assertEqual(metadata["boxes"], [])
                for operation in (
                        lambda: decoder.info(compressed, limits={"max_metadata_bytes": len(XMP) - 1}),
                        lambda: decoder.decode(compressed, frame=None,
                                               limits={"max_metadata_bytes": len(XMP) - 1})):
                    self.assertCodecFailure("resource_limit", operation)
                    self.assertReusable(decoder, retained)
                expected_boxes = [{"type": "xml ", "data": list(XMP)}]
                self.assertEqual(decoder.info(compressed)["boxes"], expected_boxes)
                decoded, metadata = decoder.decode(compressed, frame=None, return_info=True)
                self.assertTrue(metadata["complete"])
                self.assertEqual(metadata["boxes"], expected_boxes)
                self.assertEqual(len(decoded), 1)
                self.assertPixels(decoded[0], self.still)
                self.assertPixels(selected, self.still)
            self.assertRetained(retained)
            self.assertPixels(selected, self.still)
            self.assertPixels(decoded[0], self.still)


if __name__ == "__main__":
    unittest.main()
