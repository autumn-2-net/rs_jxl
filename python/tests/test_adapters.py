import io
import gc
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import rs_jxl
from rs_jxl import _arrays, _ffi, _io, _options


class ArrayAdapters(unittest.TestCase):
    def test_layout_dtype_and_endian(self):
        source = np.arange(24, dtype=">u2").reshape(3, 2, 4)[:, :, ::-1]
        result, kind = _arrays.as_array(source, layout="chw")
        self.assertEqual(kind, "u16")
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, np.moveaxis(source, 0, -1))
        self.assertEqual(result.dtype.str, "<u2")

    def test_no_lossy_dtype_coercion(self):
        for dtype in (np.int8, np.int16, np.int32, np.uint64, np.float64, np.bool_):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                _arrays.as_array(np.zeros((2, 3), dtype=dtype))

    def test_metadata_does_not_mutate_caller(self):
        extra = {"data": np.zeros((2, 3), np.uint16), "type": "depth"}
        payload = _arrays.Payload()
        image = payload.image(np.zeros((2, 3, 4), np.uint8), extra_channels=[extra])
        self.assertIn("data", extra)
        self.assertEqual(image["extra_channels"][0]["length"], 12)
        self.assertEqual(len(payload.data), 36)

    def test_subsampled_extra_geometry(self):
        image = np.zeros((3, 5, 3), np.uint8)
        _arrays.Payload().image(image, extra_channels=[{
            "data": np.zeros((2, 3), np.uint8), "dim_shift": 1,
        }])
        with self.assertRaises(ValueError):
            _arrays.Payload().image(image, extra_channels=[{
                "data": np.zeros((3, 5), np.uint8), "dim_shift": 1,
            }])

    def test_native_descriptor_bounds(self):
        spec = {"width": 1, "height": 1, "channels": 3, "dtype": "u8", "offset": 0, "length": 3}
        value = _arrays.unpack_plane(b"\x01\x02\x03", spec)
        self.assertTrue(value.flags.writeable)
        np.testing.assert_array_equal(value, [[[1, 2, 3]]])
        for update in ({"length": 4}, {"offset": 1}, {"height": -1}, {"dtype": "i8"}):
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                _arrays.unpack_plane(b"\x01\x02\x03", {**spec, **update})

    def test_duplicate_option_rejected(self):
        with self.assertRaises(ValueError):
            _options.encoder(7, 1, False, {"rct": 0}, {"modular_colorspace": 1})
        with self.assertRaises(ValueError):
            _options.encoder(7, 1, False, {"effort": 8}, {})
        with self.assertRaises(ValueError):
            _options.encoder(7, 1, False, {"strip_metadata": True}, {})
        with self.assertRaises(TypeError):
            _options.threads(True)


class IoAdapters(unittest.TestCase):
    def test_encoded_inputs_borrow_only_when_explicitly_requested(self):
        for signature in (b"\xff\xd8", b"\x89PNG"):
            for borrow in (False, True):
                source = bytearray(signature + b"sample")
                snapshot = bytes(source)

                def observe(request, data, **kwargs):
                    self.assertEqual(bytes(data), snapshot)
                    self.assertEqual(request["operation"],
                                     "transcode_jpeg" if signature == b"\xff\xd8" else "encode")
                    source[-1] ^= 1
                    self.assertEqual(bytes(data), bytes(source) if borrow else snapshot)
                    return b"encoded", {}

                with self.subTest(signature=signature, borrow=borrow):
                    with patch.object(importlib.import_module("rs_jxl.encode"),
                                      "execute", side_effect=observe):
                        self.assertEqual(rs_jxl.encode(source, borrow_inputs=borrow), b"encoded")

    def test_bytes_paths_streams(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.bin"
            self.assertEqual(_io.write_or_return(b"abcdef", path), path)
            self.assertEqual(_io.read_bytes(path), b"abcdef")
        self.assertEqual(_io.read_bytes(io.BytesIO(b"abc")), b"abc")
        self.assertEqual(_io.read_bytes(memoryview(b"abc")), b"abc")
        with self.assertRaises(TypeError):
            _io.read_bytes(io.StringIO("abc"))

    def test_short_writes_are_completed(self):
        class Short(io.BytesIO):
            def write(self, data):
                return super().write(data[:2])
        sink = Short()
        _io.write_or_return(b"abcdef", sink)
        self.assertEqual(sink.getvalue(), b"abcdef")
        with self.assertRaises(OSError):
            _io.write_or_return(b"abc", Mock(write=Mock(return_value=0)))


class AbiAdapters(unittest.TestCase):
    def test_derived_view_keeps_native_result_alive_after_scope_exit(self):
        result = _ffi.ffi.cast("RsJxlResult *", 1)
        json_data = _ffi.ffi.new("uint8_t[]", b"{}")
        samples = _ffi.ffi.new("uint8_t[]", [1, 2, 3, 4])

        def getter(data, count):
            def get(_, length):
                length[0] = count
                return data
            return get

        lib = Mock()
        lib.rs_jxl_result_json.side_effect = getter(json_data, 2)
        lib.rs_jxl_result_data.side_effect = getter(samples, 4)
        lib.rs_jxl_result_status.return_value = 0
        with _ffi._result_buffer(lib, result) as (view, _):
            derived = np.frombuffer(view, dtype=np.uint8)
        gc.collect()
        lib.rs_jxl_result_free.assert_not_called()
        np.testing.assert_array_equal(derived, [1, 2, 3, 4])
        del derived
        gc.collect()
        lib.rs_jxl_result_free.assert_called_once_with(result)

    def test_incomplete_abi_fails_before_creating_a_native_result(self):
        lib = SimpleNamespace(rs_jxl_abi_version=lambda: 1,
                              rs_jxl_request_size=lambda: _ffi.ffi.sizeof("RsJxlRequest"))
        with self.assertRaisesRegex(ImportError, "Missing rs_jxl ABI symbols"):
            _ffi._validate_abi(lib, Path("incomplete.dll"))

    def test_wrong_abi_and_structure_fail_before_execute(self):
        lib = Mock()
        lib.rs_jxl_abi_version.return_value = 99
        lib.rs_jxl_request_size.return_value = _ffi.ffi.sizeof("RsJxlRequest")
        with self.assertRaises(ImportError):
            _ffi._validate_abi(lib, Path("bad.dll"))
        lib.rs_jxl_abi_version.return_value = 1
        lib.rs_jxl_request_size.return_value = 8
        with self.assertRaises(ImportError):
            _ffi._validate_abi(lib, Path("bad.dll"))
        lib.rs_jxl_execute.assert_not_called()

    def test_failed_result_is_always_freed(self):
        result = _ffi.ffi.cast("RsJxlResult *", 1)
        error = _ffi.ffi.new("uint8_t[]", b'{"error":"failure"}')

        def json_buffer(_, length):
            length[0] = len(b'{"error":"failure"}')
            return error

        lib = Mock()
        lib.rs_jxl_execute.return_value = result
        lib.rs_jxl_result_json.side_effect = json_buffer
        lib.rs_jxl_result_status.return_value = 1
        with patch.object(_ffi, "library", return_value=lib):
            with self.assertRaisesRegex(_ffi.CodecError, "failure"):
                _ffi.execute({"operation": "decode"}, b"bad")
        lib.rs_jxl_result_free.assert_called_once_with(result)


if __name__ == "__main__":
    unittest.main()
