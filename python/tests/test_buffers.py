"""Buffer ownership tests; no native codec build or library is needed."""

import gc
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from rs_jxl import _arrays, _ffi


def descriptor(array, kind, offset=0):
    h, w, channels = array.shape
    return {"height": h, "width": w, "channels": channels, "dtype": kind,
            "offset": offset, "length": array.nbytes}


class PayloadBuffers(unittest.TestCase):
    def test_mutable_array_snapshot_is_default(self):
        for readonly in (False, True):
            with self.subTest(readonly=readonly):
                source = np.arange(18, dtype="<u2").reshape(2, 3, 3).copy()
                alias = source.view()
                alias.flags.writeable = not readonly
                expected = source.tobytes()
                payload = _arrays.Payload()
                payload.image(alias)
                self.assertFalse(np.shares_memory(np.frombuffer(payload.data, "u1"), source))
                source[:] = 123
                self.assertEqual(bytes(payload.data), expected)
                self.assertEqual(alias.flags.writeable, not readonly)

    def test_mutable_encoded_snapshot_is_default(self):
        for readonly_view in (False, True):
            with self.subTest(readonly_view=readonly_view):
                source = bytearray(b"encoded")
                value = memoryview(source).toreadonly() if readonly_view else source
                payload = _arrays.Payload()
                spec = payload.image(value)
                source[:] = b"changed"
                self.assertEqual(spec, {"format": "encoded", "offset": 0, "length": 7})
                self.assertEqual(bytes(payload.data), b"encoded")

    def test_immutable_encoded_buffers_are_borrowed_by_default(self):
        source = b"prefix-encoded-suffix"
        for value in (source, memoryview(source)[7:14]):
            with self.subTest(value=value):
                payload = _arrays.Payload()
                payload.image(value)
                self.assertIsInstance(payload.data, memoryview)
                self.assertIs(payload.data.obj, source)
                self.assertTrue(payload.data.readonly)
                self.assertEqual(bytes(payload.data), bytes(value))

    def test_contiguous_images_are_borrowed_only_when_requested(self):
        for kind, dtype in _arrays.DTYPES.items():
            for readonly in (False, True):
                with self.subTest(kind=kind, readonly=readonly):
                    source = np.arange(18, dtype=dtype).reshape(2, 3, 3).copy()
                    source.flags.writeable = not readonly
                    payload = _arrays.Payload(borrow_inputs=True)
                    spec = payload.image(source)
                    self.assertEqual(spec["dtype"], kind)
                    self.assertIsInstance(payload.data, memoryview)
                    self.assertEqual(payload.data.readonly, readonly)
                    self.assertTrue(np.shares_memory(np.frombuffer(payload.data, dtype), source))
                    self.assertEqual(bytes(payload.data), source.tobytes())

    def test_gray_and_contiguous_slice_keep_original_storage(self):
        owner = np.arange(36, dtype="<u2").reshape(6, 6)
        source = owner[1:3, :]
        payload = _arrays.Payload(borrow_inputs=True)
        spec = payload.image(source)
        self.assertEqual((spec["height"], spec["width"], spec["channels"]), (2, 6, 1))
        self.assertTrue(np.shares_memory(np.frombuffer(payload.data, "<u2"), owner))
        self.assertEqual(bytes(payload.data), source.tobytes())

    def test_borrowed_mutations_are_visible_before_execution(self):
        source = np.zeros((2, 3, 1), dtype="<u4")
        payload = _arrays.Payload(borrow_inputs=True)
        payload.image(source)
        source[0, 0, 0] = 2**30 + 1
        self.assertEqual(np.frombuffer(payload.data, "<u4")[0], 2**30 + 1)
        encoded = bytearray(b"abc")
        payload = _arrays.Payload(borrow_inputs=True)
        payload.image(memoryview(encoded).toreadonly())
        encoded[0] = ord("z")
        self.assertEqual(bytes(payload.data), b"zbc")

    def test_encoded_bytearray_borrowing_bypasses_read_bytes(self):
        source = bytearray(b"encoded")
        payload = _arrays.Payload(borrow_inputs=True)
        with patch.object(_arrays, "read_bytes", side_effect=AssertionError("unexpected copy")):
            payload.image(source)
        self.assertIs(payload.data.obj, source)

    def test_normalization_copies_only_when_needed(self):
        base = np.arange(24, dtype="<u2").reshape(2, 4, 3)
        cases = [(base[:, ::-1, :], "hwc"), (np.asfortranarray(base), "hwc"),
                 (base.astype(">u2"), "hwc"), (base.astype(">f4"), "hwc"),
                 (np.moveaxis(base, -1, 0).copy(), "chw")]
        for source, layout in cases:
            with self.subTest(dtype=source.dtype, strides=source.strides, layout=layout):
                payload = _arrays.Payload(borrow_inputs=True)
                spec = payload.image(source, layout=layout)
                dtype = _arrays.DTYPES[spec["dtype"]]
                values = np.frombuffer(payload.data, dtype=dtype).reshape(2, 4, 3)
                expected = np.moveaxis(source, 0, -1) if layout == "chw" else source
                self.assertFalse(np.shares_memory(values, source))
                np.testing.assert_array_equal(values, expected)
                self.assertTrue(values.flags.c_contiguous)
                self.assertEqual(values.dtype, dtype)
                before = bytes(payload.data)
                source[:] = 0
                self.assertEqual(bytes(payload.data), before)

    def test_noncontiguous_encoded_view_is_normalized(self):
        for borrow in (False, True):
            with self.subTest(borrow=borrow):
                source = bytearray(b"a0b0c0")
                payload = _arrays.Payload(borrow_inputs=borrow)
                payload.image(memoryview(source)[::2])
                source[:] = b"z0z0z0"
                self.assertEqual(bytes(payload.data), b"abc")
                self.assertTrue(payload.data.c_contiguous)

    def test_path_and_stream_inputs_remain_owned_bytes(self):
        for borrow in (False, True):
            with self.subTest(borrow=borrow):
                payload = _arrays.Payload(borrow_inputs=borrow)
                payload.image(io.BytesIO(b"stream"))
                self.assertEqual(bytes(payload.data), b"stream")
                with patch.object(_arrays, "read_bytes", return_value=b"path") as read:
                    payload = _arrays.Payload(borrow_inputs=borrow)
                    payload.image("input.png")
                read.assert_called_once_with("input.png")
                self.assertEqual(bytes(payload.data), b"path")

    def test_second_append_consolidates_and_freezes_borrowed_contents(self):
        first = bytearray(b"abc")
        second = np.array([0x1234, 0x5678], dtype="<u2")
        payload = _arrays.Payload(borrow_inputs=True)
        self.assertEqual(payload.add(first), {"offset": 0, "length": 3})
        self.assertIs(payload.data.obj, first)
        first[0] = ord("z")
        self.assertEqual(payload.add(second), {"offset": 3, "length": 4})
        self.assertIsInstance(payload.data, bytearray)
        consolidated = payload.data
        first[:] = b"xxx"
        second[:] = 0
        self.assertEqual(bytes(payload.data), b"zbc\x34\x12\x78\x56")
        self.assertEqual(payload.add(b"last"), {"offset": 7, "length": 4})
        self.assertIs(payload.data, consolidated)
        self.assertEqual(bytes(payload.data), b"zbc\x34\x12\x78\x56last")

    def test_default_snapshot_survives_later_consolidation(self):
        first, second = bytearray(b"abc"), bytearray(b"def")
        payload = _arrays.Payload()
        payload.add(first)
        first[:] = b"xxx"
        payload.add(second)
        second[:] = b"yyy"
        self.assertEqual(bytes(payload.data), b"abcdef")

    def test_owned_snapshot_can_grow_without_retaining_original_exporter(self):
        source = np.arange(6, dtype="u1").reshape(2, 3, 1).copy()
        ref = weakref.ref(source)
        payload = _arrays.Payload()
        payload.image(source)
        snapshot = payload.data
        del source
        gc.collect()
        self.assertIsNone(ref())
        payload.add(bytearray(b"more"))
        self.assertIs(payload.data, snapshot)
        self.assertEqual(bytes(payload.data), bytes(range(6)) + b"more")

    def test_readonly_inputs_support_const_cffi_export(self):
        source = np.arange(6, dtype="<u2").reshape(2, 3, 1)
        source.flags.writeable = False
        for borrow in (False, True):
            for value in (source, source.tobytes()):
                with self.subTest(borrow=borrow, source_type=type(value)):
                    payload = _arrays.Payload(borrow_inputs=borrow)
                    payload.image(value)
                    exported = _ffi.ffi.from_buffer("const uint8_t[]", payload.data)
                    self.assertEqual(bytes(_ffi.ffi.buffer(exported)), source.tobytes())

    def test_empty_buffers_have_correct_offsets(self):
        payload = _arrays.Payload(borrow_inputs=True)
        for value, offset, length in [(b"", 0, 0), (b"ab", 0, 2), (b"", 2, 0), (b"c", 2, 1)]:
            self.assertEqual(payload.add(value), {"offset": offset, "length": length})
        self.assertEqual(bytes(payload.data), b"abc")

    def test_extra_channels_and_frames_have_byte_offsets(self):
        for borrow in (False, True):
            with self.subTest(borrow=borrow):
                rgb = np.arange(18, dtype="<u2").reshape(2, 3, 3)
                extra = np.arange(6, dtype="<f4").reshape(2, 3)
                next_frame = np.arange(6, dtype="u1").reshape(2, 3)
                payload = _arrays.Payload(borrow_inputs=borrow)
                first = payload.image(rgb, extra_channels=[extra])
                second = payload.image(next_frame)
                self.assertEqual((first["offset"], first["length"]), (0, 36))
                plane = first["extra_channels"][0]
                self.assertEqual((plane["offset"], plane["length"]), (36, 24))
                self.assertEqual((second["offset"], second["length"]), (60, 6))
                self.assertEqual(bytes(payload.data), rgb.tobytes() + extra.tobytes() + next_frame.tobytes())

    def test_releasing_callers_view_does_not_release_borrow(self):
        source = bytearray(b"abc")
        view = memoryview(source)
        payload = _arrays.Payload(borrow_inputs=True)
        payload.add(view)
        view.release()
        self.assertEqual(bytes(payload.data), b"abc")
        payload.add(b"def")
        with self.assertRaises(BufferError):
            source.extend(b"g")
        del payload
        source.extend(b"g")
        self.assertEqual(source, b"abcg")

    def test_temporary_exporters_survive_execute_and_consolidation(self):
        for count in (1, 3):
            with self.subTest(count=count):
                payload = _arrays.Payload(borrow_inputs=True)
                refs = []
                for index in range(count):
                    source = np.full((2, 3, 1), index, dtype="u1")
                    refs.append(weakref.ref(source))
                    payload.image(source)
                    del source
                gc.collect()
                expected = bytes(payload.data)
                metadata = _ffi.ffi.new("uint8_t[]", b"{}")
                lib = Mock()

                def execute(request):
                    gc.collect()
                    self.assertTrue(all(ref() is not None for ref in refs))
                    self.assertEqual(bytes(_ffi.ffi.buffer(request.data, request.data_len)), expected)
                    return object()

                def json_buffer(result, length):
                    length[0] = 2
                    return metadata

                lib.rs_jxl_execute.side_effect = execute
                lib.rs_jxl_result_json.side_effect = json_buffer
                lib.rs_jxl_result_status.return_value = 0
                lib.rs_jxl_result_data.return_value = _ffi.ffi.NULL
                with patch.object(_ffi, "library", return_value=lib):
                    self.assertEqual(_ffi.execute({"operation": "encode"}, payload.data), (b"", {}))
                lib.rs_jxl_result_free.assert_called_once()
                self.assertTrue(all(ref() is not None for ref in refs))
                del payload
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))

    def test_normalized_temporary_is_retained_after_original_is_deleted(self):
        source = np.arange(12, dtype=">u2").reshape(2, 2, 3)
        expected = source.astype("<u2").tobytes()
        payload = _arrays.Payload(borrow_inputs=True)
        payload.image(source)
        del source
        gc.collect()
        self.assertEqual(bytes(payload.data), expected)

    def test_invalid_add_does_not_change_payload(self):
        payload = _arrays.Payload(borrow_inputs=True)
        payload.add(b"abc")
        original = payload.data
        with self.assertRaises(TypeError):
            payload.add(object())
        self.assertIs(payload.data, original)
        self.assertEqual(payload.add(b"d"), {"offset": 3, "length": 1})

    def test_unsupported_dtypes_are_not_narrowed(self):
        for dtype in (np.int8, np.int16, np.int32, np.uint64, np.float64, np.bool_):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                _arrays.Payload(borrow_inputs=True).image(np.zeros((2, 3, 1), dtype=dtype))

    def test_device_transfer_remains_explicit_and_torch_layout_is_preserved(self):
        class Tensor:
            __module__ = "torch"

            def __init__(self, device):
                self.device = SimpleNamespace(type=device)
                self.array = np.arange(18, dtype="<u2").reshape(3, 2, 3)
                self.transfers = 0

            def detach(self):
                return self

            def cpu(self):
                self.transfers += 1
                return self

            def numpy(self):
                return self.array

        tensor = Tensor("cuda")
        with self.assertRaisesRegex(ValueError, "allow_device_copy=True"):
            _arrays.Payload(borrow_inputs=True).image(tensor)
        self.assertEqual(tensor.transfers, 0)
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                tensor = Tensor(device)
                payload = _arrays.Payload(borrow_inputs=True)
                payload.image(tensor, allow_device_copy=device != "cpu")
                values = np.frombuffer(payload.data, "<u2").reshape(2, 3, 3)
                np.testing.assert_array_equal(values, np.moveaxis(tensor.array, 0, -1))
                self.assertEqual(tensor.transfers, 1)


class CopyIntoBuffers(unittest.TestCase):
    def test_exact_dtypes_offsets_and_destination_identity(self):
        for kind, dtype in _arrays.DTYPES.items():
            for channels in (1, 2, 3, 4):
                with self.subTest(kind=kind, channels=channels):
                    source = np.arange(6 * channels, dtype=dtype).reshape(2, 3, channels)
                    if kind == "u32":
                        source += 2**30
                    raw = b"pad" + source.tobytes() + b"tail"
                    destination = np.empty(source.shape, dtype=dtype.newbyteorder("="))
                    pointer = destination.__array_interface__["data"][0]
                    result = _arrays.copy_into(raw, descriptor(source, kind, 3), destination)
                    self.assertIs(result, destination)
                    self.assertEqual(destination.__array_interface__["data"][0], pointer)
                    np.testing.assert_array_equal(destination, source)
                    self.assertTrue(destination.dtype.isnative)

    def test_float_bit_patterns_survive_copy(self):
        for kind, word_dtype, words in (
            ("f16", "<u2", [0, 0x8000, 0x7E01, 0x7C00, 0xFC00, 1]),
            ("f32", "<u4", [0, 0x80000000, 0x7FC00001, 0x7F800000, 0xFF800000, 1]),
        ):
            with self.subTest(kind=kind):
                source = np.array(words, dtype=word_dtype).view(_arrays.DTYPES[kind]).reshape(2, 3, 1)
                destination = np.empty(source.shape, dtype=source.dtype.newbyteorder("="))
                _arrays.copy_into(memoryview(source), descriptor(source, kind), destination)
                actual = destination.view(np.dtype(word_dtype).newbyteorder("=")).reshape(-1)
                np.testing.assert_array_equal(actual, words)

    def test_buffer_protocol_sources_and_nonowning_destination(self):
        source = np.arange(12, dtype="<u2").reshape(2, 2, 3)
        spec = descriptor(source, "u16")
        for data in (source.tobytes(), bytearray(source.tobytes()), memoryview(source),
                     memoryview(source).toreadonly()):
            with self.subTest(source_type=type(data)):
                owner = np.full((4, 2, 3), 77, dtype="=u2")
                destination = owner[1:3]
                _arrays.copy_into(data, spec, destination)
                np.testing.assert_array_equal(destination, source)
                np.testing.assert_array_equal(owner[[0, 3]], 77)

    def test_copy_uses_source_view_without_an_intermediate_plane(self):
        source = np.arange(12, dtype="<u2").reshape(2, 2, 3)
        destination = np.empty_like(source, dtype="=u2")
        with patch.object(_arrays.np, "copyto", wraps=np.copyto) as copy:
            _arrays.copy_into(memoryview(source), descriptor(source, "u16"), destination)
        target, values = copy.call_args.args
        self.assertIs(target, destination)
        self.assertTrue(np.shares_memory(values, source))
        self.assertEqual(copy.call_args.kwargs, {"casting": "equiv"})

    def test_overlapping_source_and_destination_are_supported(self):
        owner = np.arange(9, dtype="u1")
        expected = owner[:6].reshape(2, 3, 1).copy()
        destination = owner[3:].reshape(2, 3, 1)
        _arrays.copy_into(owner, descriptor(expected, "u8"), destination)
        np.testing.assert_array_equal(destination, expected)

    def test_missing_channels_defaults_to_gray(self):
        source = np.arange(6, dtype="u1").reshape(2, 3, 1)
        spec = descriptor(source, "u8")
        del spec["channels"]
        destination = np.empty_like(source)
        _arrays.copy_into(source, spec, destination)
        np.testing.assert_array_equal(destination, source)

    def test_invalid_descriptors_never_modify_destination(self):
        source = np.arange(6, dtype="<u2").reshape(2, 3, 1)
        spec = descriptor(source, "u16")
        invalid = [{**spec, key: value} for key, value in (
            ("dtype", "i16"), ("dtype", []), ("height", 0), ("height", True),
            ("width", -1), ("channels", 5), ("channels", 0), ("offset", -1),
            ("offset", 1), ("offset", 10**100), ("length", 11), ("length", 13),
            ("length", 1.5), ("length", True), ("height", 10**100),
        )]
        invalid.extend({key: value for key, value in spec.items() if key != missing}
                       for missing in ("dtype", "height", "width", "offset", "length"))
        for bad_spec in invalid:
            with self.subTest(spec=bad_spec):
                destination = np.full_like(source, 123, dtype="=u2")
                before = destination.tobytes()
                with self.assertRaises(RuntimeError):
                    _arrays.copy_into(source, bad_spec, destination)
                self.assertEqual(destination.tobytes(), before)

    def test_invalid_destinations_never_modify_storage(self):
        source = np.arange(12, dtype="<u2").reshape(2, 2, 3)
        wrong_endian = np.dtype("=u2").newbyteorder("S")
        readonly = np.full(source.shape, 123, dtype="=u2")
        readonly.flags.writeable = False
        destinations = [
            np.full(source.shape, 123, dtype=dtype)
            for dtype in ("u1", "=u4", "=i2", "=f2", "=f4", wrong_endian, object)
        ]
        destinations.extend([
            np.full((2, 6), 123, dtype="=u2"),
            np.full((3, 2, 2), 123, dtype="=u2"),
            np.full((2, 4, 3), 123, dtype="=u2")[:, ::2],
            np.full(source.shape, 123, dtype="=u2")[:, :, ::-1],
            np.full(source.shape, 123, dtype="=u2", order="F"), readonly,
        ])
        for destination in destinations:
            with self.subTest(dtype=destination.dtype, shape=destination.shape,
                              strides=destination.strides, writable=destination.flags.writeable):
                before = destination.tobytes()
                with self.assertRaises((TypeError, ValueError)):
                    _arrays.copy_into(source, descriptor(source, "u16"), destination)
                self.assertEqual(destination.tobytes(), before)
        with self.assertRaises(TypeError):
            _arrays.copy_into(source, descriptor(source, "u16"), memoryview(source))

    def test_invalid_source_buffers_never_modify_destination(self):
        source = np.arange(6, dtype="<u2").reshape(2, 3, 1)
        released = memoryview(source)
        released.release()
        for data in (source.tobytes()[:-1], memoryview(source)[::2], released, object()):
            with self.subTest(source_type=type(data)):
                destination = np.full_like(source, 123, dtype="=u2")
                before = destination.tobytes()
                with self.assertRaises(RuntimeError):
                    _arrays.copy_into(data, descriptor(source, "u16"), destination)
                self.assertEqual(destination.tobytes(), before)

    def test_unpack_plane_still_returns_independent_writable_native_array(self):
        source = np.arange(6, dtype="<u2").reshape(2, 3, 1)
        raw = bytearray(source.tobytes())
        result = _arrays.unpack_plane(raw, descriptor(source, "u16"))
        self.assertTrue(result.flags.writeable)
        self.assertTrue(result.dtype.isnative)
        self.assertFalse(np.shares_memory(result, np.frombuffer(raw, "<u2")))
        raw[:] = b"\0" * len(raw)
        np.testing.assert_array_equal(result, source)

    def test_unpack_plane_bytes_always_returns_owned_bytes(self):
        source = np.arange(6, dtype="<u2").reshape(2, 3, 1)
        for input_type in (bytes, bytearray, memoryview):
            with self.subTest(input_type=input_type):
                storage = bytearray(b"pad" + source.tobytes() + b"tail")
                data = input_type(storage)
                result = _arrays.unpack_plane(data, descriptor(source, "u16", 3), out="bytes")
                self.assertIs(type(result), bytes)
                if isinstance(data, memoryview):
                    data.release()
                storage[:] = b"\0" * len(storage)
                self.assertEqual(result, source.tobytes())

    def test_outputs_survive_scoped_native_result_release(self):
        source = np.arange(6, dtype="<u2").reshape(2, 3, 1)
        raw = b"pad" + source.tobytes() + b"tail"
        spec = descriptor(source, "u16", 3)
        for mode in ("bytes", "numpy", "into"):
            with self.subTest(mode=mode):
                native = _ffi.ffi.new("uint8_t[]", raw)
                metadata = _ffi.ffi.new("uint8_t[]", b"{}")
                lib = Mock()
                lib.rs_jxl_execute.return_value = object()
                lib.rs_jxl_result_status.return_value = 0

                def json_buffer(result, length):
                    length[0] = 2
                    return metadata

                def pixel_buffer(result, length):
                    length[0] = len(raw)
                    return native

                def free_result(result):
                    _ffi.ffi.buffer(native, len(raw))[:] = b"\xff" * len(raw)

                lib.rs_jxl_result_json.side_effect = json_buffer
                lib.rs_jxl_result_data.side_effect = pixel_buffer
                lib.rs_jxl_result_free.side_effect = free_result
                with patch.object(_ffi, "library", return_value=lib):
                    with _ffi.execute_buffer({"operation": "decode"}) as (buffer, _):
                        if mode == "into":
                            destination = np.empty_like(source, dtype="=u2")
                            result = _arrays.copy_into(buffer, spec, destination)
                            self.assertIs(result, destination)
                        else:
                            result = _arrays.unpack_plane(buffer, spec, out=mode)
                        lib.rs_jxl_result_free.assert_not_called()
                lib.rs_jxl_result_free.assert_called_once()
                with self.assertRaises(ValueError):
                    bytes(buffer)
                if mode == "bytes":
                    self.assertIs(type(result), bytes)
                    self.assertEqual(result, source.tobytes())
                else:
                    np.testing.assert_array_equal(result, source)
                    self.assertTrue(result.flags.writeable)
                    self.assertTrue(result.dtype.isnative)


if __name__ == "__main__":
    unittest.main()
