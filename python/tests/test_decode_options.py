"""Decode recovery option transport and validation without a native library."""

from contextlib import contextmanager
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import rs_jxl
from rs_jxl import contexts


decode_module = importlib.import_module("rs_jxl.decode")
JXL = b"\xff\x0a"
ENTRIES = ("decode", "decode_into", "decode_layer", "decode_extra_channel", "decode_to_png")


@contextmanager
def pixels(*args, **kwargs):
    plane = {"dtype": "u8", "width": 1, "height": 1, "channels": 1,
             "offset": 0, "length": 1}
    extra = {**plane, "offset": 1, "type": "depth"}
    yield b"\x07\x09", {"frames": [{**plane, "extra_channels": [extra]}]}


class DecodeOptions(unittest.TestCase):
    def invoke(self, api, name, src=JXL, **options):
        operation = getattr(api, name)
        if name == "decode_into":
            destination = np.full((1, 1, 1), 255, dtype=np.uint8)
            result = operation(src, destination, **options)
            self.assertIs(result, destination)
            self.assertEqual(destination.item(), 7)
        elif name == "decode_to_png":
            self.assertEqual(operation(src, **options), b"png")
        else:
            result = operation(src, out="bytes", **options)
            if name == "decode_layer":
                result = result[0]
            if options.get("frame", 0) is None:
                result = result[0]
            self.assertEqual(result, b"\x09" if name == "decode_extra_channel" else b"\x07")

    def test_defaults_and_explicit_bools_reach_every_decode_entry(self):
        with patch.object(decode_module, "execute_buffer", side_effect=pixels) as buffer, \
                patch.object(decode_module, "execute", return_value=(b"png", {})) as execute:
            for name in ENTRIES:
                for options, expected in (({}, False), ({"legacy_gradient": False}, False),
                                          ({"legacy_gradient": True}, True)):
                    with self.subTest(entry=name, options=options):
                        self.invoke(rs_jxl, name, **options)
                        call = (execute if name == "decode_to_png" else buffer).call_args
                        request = call.args[0]
                        self.assertIs(request["options"]["legacy_gradient"], expected)
                        self.assertEqual(request["operation"],
                                         "decode_to_png" if name == "decode_to_png" else "decode")
                        if name == "decode_layer":
                            self.assertIs(request["options"]["coalescing"], False)
                        if name == "decode_extra_channel":
                            self.assertIs(request["options"]["include_extra_channels"], True)
            self.invoke(rs_jxl, "decode", frame=None, legacy_gradient=True)
            self.assertIsNone(buffer.call_args.args[0]["options"]["frame_index"])
            self.assertIs(buffer.call_args.args[0]["options"]["legacy_gradient"], True)

    def test_context_calls_forward_the_flag_without_retaining_it(self):
        native = SimpleNamespace(threads=2, closed=False)
        with patch.object(contexts, "NativeContext", return_value=native), \
                patch.object(decode_module, "execute_buffer", side_effect=pixels) as buffer, \
                patch.object(decode_module, "execute", return_value=(b"png", {})) as execute:
            decoder = rs_jxl.Decoder(threads=2)
            for name in ENTRIES:
                for options, expected in (({"legacy_gradient": True}, True), ({}, False),
                                          ({"legacy_gradient": False}, False)):
                    with self.subTest(entry=name, options=options):
                        self.invoke(decoder, name, **options)
                        call = (execute if name == "decode_to_png" else buffer).call_args
                        self.assertIs(call.kwargs["context"], native)
                        self.assertEqual(call.args[0]["options"]["threads"], 2)
                        self.assertIs(call.args[0]["options"]["legacy_gradient"], expected)

    def test_wrong_types_fail_before_reading_input_or_calling_native(self):
        native = SimpleNamespace(threads=1, closed=False)
        with patch.object(contexts, "NativeContext", return_value=native), \
                patch.object(decode_module, "read_bytes") as read, \
                patch.object(decode_module, "execute_buffer") as buffer, \
                patch.object(decode_module, "execute") as execute:
            for api in (rs_jxl, rs_jxl.Decoder()):
                for name in ENTRIES:
                    for value in (None, 0, 1, 1.0, "true", "false", [], {}, np.bool_(True)):
                        with self.subTest(entry=name, value=value), \
                                self.assertRaisesRegex(TypeError, "legacy_gradient must be a boolean"):
                            self.invoke(api, name, object(), legacy_gradient=value)
            read.assert_not_called()
            buffer.assert_not_called()
            execute.assert_not_called()

    def test_recovery_is_rejected_for_png_and_jpeg_pixel_input(self):
        with patch.object(decode_module, "execute_buffer", side_effect=pixels) as buffer:
            for source in (b"\x89PNG\r\n\x1a\n", b"\xff\xd8"):
                for name in ("decode", "decode_into", "decode_extra_channel"):
                    with self.subTest(entry=name, source=source), \
                            self.assertRaisesRegex(ValueError, "JXL-specific"):
                        self.invoke(rs_jxl, name, source, legacy_gradient=True)
                buffer.assert_not_called()
                self.invoke(rs_jxl, "decode", source, legacy_gradient=False)
                self.assertEqual(buffer.call_args.args[0]["operation"], "read_image")
                self.assertNotIn("legacy_gradient", buffer.call_args.args[0]["options"])
                buffer.reset_mock()

    def test_decode_failure_never_retries_with_a_different_interpretation(self):
        failure = rs_jxl.CodecError("decode failed", kind="decoding")
        with patch.object(decode_module, "execute_buffer", side_effect=failure) as buffer, \
                patch.object(decode_module, "execute", side_effect=failure) as execute:
            for name in ENTRIES:
                for options, expected in (({}, False), ({"legacy_gradient": True}, True)):
                    with self.subTest(entry=name, options=options), \
                            self.assertRaises(rs_jxl.CodecError) as raised:
                        self.invoke(rs_jxl, name, **options)
                    self.assertIs(raised.exception, failure)
                    called = execute if name == "decode_to_png" else buffer
                    self.assertEqual(called.call_count, 1)
                    self.assertIs(called.call_args.args[0]["options"]["legacy_gradient"], expected)
                    buffer.reset_mock()
                    execute.reset_mock()

    def test_recovery_is_not_an_encoding_or_inspection_option(self):
        source = np.zeros((1, 1, 1), dtype=np.uint8)
        for value in (False, True, None):
            for operation in (rs_jxl.encode, rs_jxl.encode_layers, rs_jxl.encode_multiframe):
                for options in ({"legacy_gradient": value},
                                {"encoder_options": {"legacy_gradient": value}}):
                    with self.subTest(entry=operation.__name__, options=options), \
                            self.assertRaisesRegex(ValueError, "decode-only recovery"):
                        operation(source if operation is rs_jxl.encode else [source], **options)
            for operation in (rs_jxl.info, rs_jxl.reconstruct_jpeg, rs_jxl.transcode_jpeg):
                with self.subTest(entry=operation.__name__), self.assertRaises(TypeError):
                    operation(object(), legacy_gradient=value)


if __name__ == "__main__":
    unittest.main()
