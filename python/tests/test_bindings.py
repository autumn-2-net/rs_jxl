"""End-to-end tests against an installed/PYTHONPATH rs_jxl and its native library.

No Pillow is needed. RS_JXL_DJXL optionally names official decoder executables
using os.pathsep; configured oracles must succeed. RS_JXL_TEST_CUDA=1 opts into
GPU transfer tests, while the codec itself remains CPU-only.
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import io
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import unittest

import numpy as np
import rs_jxl


def _rgba(dtype=np.uint8, height=5, width=7):
    maximum = np.iinfo(dtype).max
    values = np.arange(height * width * 4, dtype=np.uint64)
    image = ((values * 173 + 37) % (maximum + 1)).astype(dtype)
    image = image.reshape(height, width, 4)
    image[..., 3] = maximum
    image[::2, :, 3] = 0
    image[1::2, ::2, 3] = maximum // 2
    return image


def _frames():
    first = _rgba()
    second = first.copy()
    second[0, 1, 0] ^= 127  # A real difference under zero alpha.
    third = first.copy()
    third[2, 4, 1] ^= 63
    return [first, second, third, third.copy()]


def _tiny_jpeg():
    # Port of jpeg/archival.rs::tests::tiny_jpeg: one gray 1x1 baseline block.
    # DC zero and EOB each use code 0; six padding bits complete the scan byte.
    data = bytearray([0xFF, 0xD8, 0xFF, 0xDB, 0, 67, 0])
    data.extend([16] * 64)
    data.extend([0xFF, 0xC0, 0, 11, 8, 0, 1, 0, 1, 1, 1, 0x11, 0])
    data.extend([0xFF, 0xC4, 0, 38])
    for selector in (0, 0x10):
        data.extend([selector, 1])
        data.extend([0] * 16)
    data.extend([0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 63, 0, 0x3F, 0xFF, 0xD9])
    return bytes(data)


def setUpModule():
    # Deliberately do not catch import/ABI/load errors or skip native coverage.
    pixel = np.array([[73]], dtype=np.uint8)
    encoded = rs_jxl.encode(pixel, effort=3, threads=1)
    np.testing.assert_array_equal(rs_jxl.decode(encoded, threads=1), pixel[..., None])


class PixelAssertions:
    def assertPixels(self, actual, expected):
        if expected.ndim == 2:
            expected = expected[..., None]
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype.newbyteorder("="))
        np.testing.assert_array_equal(actual, expected)

    def roundTrip(self, source, expected=None, **options):
        encoded = rs_jxl.encode(source, effort=3, threads=1, **options)
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 2)
        self.assertPixels(rs_jxl.decode(encoded, threads=1), source if expected is None else expected)
        return encoded


class BindingsTests(PixelAssertions, unittest.TestCase):
    def test_hidden_rgb_rgba8_and_rgba16(self):
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                source = _rgba(dtype)
                hidden = source[..., 3] == 0
                self.assertTrue(np.any(source[..., :3][hidden] != 0))
                encoded = self.roundTrip(source)
                decoded, details = rs_jxl.decode(encoded, return_info=True, threads=1)
                self.assertPixels(decoded, source)
                header = rs_jxl.info(encoded, threads=1)
                for metadata in (details, header):
                    self.assertEqual((metadata["height"], metadata["width"]), source.shape[:2])
                    self.assertEqual(metadata["bits_per_sample"], source.dtype.itemsize * 8)
                self.assertEqual(len(header["frames"]), 1)

    def test_gray_gray_alpha_and_rgb_shapes(self):
        gray = np.arange(30, dtype=np.uint8).reshape(5, 6)
        for source in (gray, gray[..., None], np.stack([gray, 255 - gray], axis=-1),
                       np.stack([gray, 31 + gray, 63 + gray], axis=-1)):
            with self.subTest(shape=source.shape):
                self.roundTrip(source)

    def test_noncontiguous_endian_and_explicit_chw(self):
        source = _rgba(np.uint16, height=6, width=8)
        cases = [source[::2, ::2], source[::-1, ::-1], np.asfortranarray(source),
                 source.astype(">u2")[:, ::-1]]
        for array in cases:
            with self.subTest(strides=array.strides, dtype=str(array.dtype)):
                self.roundTrip(array)
        self.roundTrip(np.moveaxis(source, -1, 0), expected=source, layout="chw")

    def test_uint31_preserves_bits_beyond_float_precision(self):
        source = np.array([0, 1, 65535, 65536, 16777215, 16777216, 16777217,
                           1073741823, 1073741824, 2147483645, 2147483646,
                           2147483647], dtype=np.uint32).reshape(3, 4)
        encoded = self.roundTrip(source, bits_per_sample=31)
        self.assertEqual(rs_jxl.info(encoded, threads=1)["bits_per_sample"], 31)
        with self.assertRaises((ValueError, rs_jxl.CodecError)):
            rs_jxl.encode(np.array([[1 << 31]], dtype=np.uint32), effort=3, threads=1)

    def test_finite_float16_and_float32_exact_words(self):
        for dtype, bits, exponent in ((np.float16, 16, 5), (np.float32, 32, 8)):
            with self.subTest(dtype=dtype):
                source = np.array([0.0, -0.0, 0.125, 0.5, 1.0, -1.0,
                                   2.0, -2.0, np.finfo(dtype).tiny,
                                   np.nextafter(dtype(0), dtype(1)), 0.75, 16.0],
                                  dtype=dtype).reshape(3, 4)
                self.assertTrue(np.isfinite(source).all())
                encoded = rs_jxl.encode(source, color_encoding="linear_srgb", effort=3, threads=1)
                decoded = rs_jxl.decode(encoded, threads=1)
                # Native decoding widens F16 to F32 without losing finite values.
                expected = source.astype(np.float32)[..., None]
                self.assertPixels(decoded, expected)
                np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
                metadata = rs_jxl.info(encoded, threads=1)
                self.assertEqual(metadata["bits_per_sample"], bits)
                self.assertEqual(metadata["exponent_bits_per_sample"], exponent)

    def test_signed_and_narrowing_dtypes_are_rejected(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint64,
                      np.float64, np.complex64, np.bool_, object):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    rs_jxl.encode(np.ones((2, 3, 3), dtype=dtype), effort=3, threads=1)

    def test_lossy_and_invalid_options_are_rejected(self):
        source = _rgba(height=2, width=3)
        for options in ({"lossless": False}, {"distance": 1.0}, {"effort": 0},
                        {"threads": -1}, {"layout": "invalid"}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    rs_jxl.encode(source, **options)
        with self.assertRaises((TypeError, ValueError, rs_jxl.CodecError)):
            rs_jxl.encode(source, effort=3, threads=1, unrecognized_option=True)

    def test_duplicate_strip_metadata_encoder_option_is_rejected(self):
        source = _rgba(height=2, width=3)
        for encode, images in ((rs_jxl.encode, source),
                               (rs_jxl.encode_multiframe, [source])):
            for strip_metadata in (False, True):
                with self.subTest(encoder=encode.__name__, strip_metadata=strip_metadata):
                    with self.assertRaisesRegex(ValueError, "strip_metadata"):
                        encode(images, effort=3, threads=1, strip_metadata=strip_metadata,
                               encoder_options={"strip_metadata": True})

    def test_paths_bytes_and_streams(self):
        source = _rgba()
        encoded = self.roundTrip(source)
        with tempfile.TemporaryDirectory(prefix="rs-jxl-python-") as directory:
            path = Path(directory) / "image.jxl"
            for destination in (path, str(path)):
                result = rs_jxl.encode(source, destination, effort=3, threads=1)
                self.assertEqual(Path(result), path)
                self.assertPixels(rs_jxl.decode(destination, threads=1), source)
            stream = io.BytesIO()
            self.assertIsNone(rs_jxl.encode(source, stream, effort=3, threads=1))
            self.assertFalse(stream.closed)
            for data in (encoded, bytearray(encoded), memoryview(encoded),
                         io.BytesIO(stream.getvalue())):
                with self.subTest(source_type=type(data).__name__):
                    self.assertPixels(rs_jxl.decode(data, threads=1), source)

    def test_bytes_output_and_arrays_outlive_native_results(self):
        source = _rgba(np.uint16)
        encoded = self.roundTrip(source)
        first = rs_jxl.decode(encoded, threads=1)
        second = rs_jxl.decode(encoded, threads=1)
        raw, metadata = rs_jxl.decode(encoded, out="bytes", return_info=True, threads=1)
        self.assertIsInstance(raw, bytes)
        self.assertEqual(raw, source.astype("<u2").tobytes())
        self.assertEqual(metadata["dtype"], "u16")
        del encoded
        gc.collect()
        replacement = rs_jxl.encode(np.zeros_like(source), effort=3, threads=1)
        for _ in range(3):
            self.assertPixels(rs_jxl.decode(replacement, threads=1), np.zeros_like(source))
        self.assertFalse(np.shares_memory(first, second))
        self.assertTrue(first.flags.writeable)
        self.assertPixels(first, source)
        first.fill(0)
        self.assertPixels(second, source)
        self.assertEqual(raw, source.astype("<u2").tobytes())

    def test_extra_planes_and_alpha_filter(self):
        source = _rgba()
        depth = (np.arange(35, dtype=np.uint16).reshape(5, 7) * 1301)[:, ::-1]
        selection = (np.arange(35, dtype=np.uint8).reshape(5, 7) % 2) * 255
        extras = [{"data": depth, "type": "depth", "name": "distance"},
                  {"data": selection, "type": "selection_mask", "name": "selected"}]
        encoded = self.roundTrip(source, extra_channels=extras)
        image, details = rs_jxl.decode(encoded, return_extra_channels=True, threads=1)
        self.assertPixels(image, source)
        self.assertEqual([e["type"] for e in details["extra_channels"]],
                         ["depth", "selection_mask"])
        for actual, expected in zip(details["extra_channels"], extras):
            self.assertEqual(actual["name"], expected["name"])
            self.assertPixels(actual["data"], expected["data"])
        _, all_details = rs_jxl.decode(encoded, return_extra_channels=True,
                                       include_alpha_extra=True, threads=1)
        self.assertEqual(len(all_details["extra_channels"]), 3)
        alpha = all_details["extra_channels"][0]
        self.assertEqual(alpha["type"], "alpha")
        self.assertPixels(alpha["data"], source[..., 3])
        details["extra_channels"][0]["data"].fill(0)
        self.assertPixels(all_details["extra_channels"][1]["data"], depth)
        header = rs_jxl.info(encoded, threads=1)
        self.assertEqual([e["type"] for e in header["extra_channels"]],
                         ["alpha", "depth", "selection_mask"])
        raw, byte_details = rs_jxl.decode(encoded, out="bytes", return_extra_channels=True,
                                          threads=1)
        self.assertEqual(raw, source.tobytes())
        self.assertEqual(byte_details["extra_channels"][0]["data"], depth.astype("<u2").tobytes())
        self.assertEqual(byte_details["extra_channels"][1]["data"], selection.tobytes())

    def test_extra_plane_geometry_is_checked(self):
        with self.assertRaises(ValueError):
            rs_jxl.encode(_rgba(), extra_channels=[{
                "data": np.zeros((2, 2), dtype=np.uint8), "type": "depth"}],
                effort=3, threads=1)

    def test_decode_to_png_without_pillow(self):
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                source = _rgba(dtype)
                encoded = self.roundTrip(source)
                png = rs_jxl.decode_to_png(encoded, threads=1)
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(png[12:16], b"IHDR")
                self.assertEqual(struct.unpack(">IIB", png[16:25]),
                                 (source.shape[1], source.shape[0], source.dtype.itemsize * 8))
                self.assertPixels(rs_jxl.decode(png, threads=1), source)
                self.roundTrip(png, expected=source)
                path = Path(directory) / "image.png"
                rs_jxl.decode_to_png(encoded, path, threads=1)
                self.assertPixels(rs_jxl.decode(path, threads=1), source)
                stream = io.BytesIO()
                self.assertIsNone(rs_jxl.decode_to_png(encoded, stream, threads=1))
                self.assertPixels(rs_jxl.decode(io.BytesIO(stream.getvalue()), threads=1), source)

    def test_invalid_input_frames_and_decode_limits(self):
        for data in (b"", b"not an image", b"\xff\x0a\x00"):
            with self.subTest(data=data):
                with self.assertRaises(rs_jxl.CodecError):
                    rs_jxl.decode(data, threads=1)
        encoded = self.roundTrip(_rgba())
        with self.assertRaises(ValueError):
            rs_jxl.decode(encoded, frame=-1, threads=1)
        with self.assertRaises((IndexError, rs_jxl.CodecError)):
            rs_jxl.decode(encoded, frame=20, threads=1)
        with self.assertRaises(rs_jxl.CodecError):
            rs_jxl.decode(encoded, max_pixels=1, threads=1)
        with self.assertRaises(ValueError):
            rs_jxl.decode(encoded, out="unsupported", threads=1)
        with self.assertRaises(rs_jxl.CodecError):
            rs_jxl.reconstruct_jpeg(encoded, threads=1)

    def test_eight_concurrent_cffi_roundtrips(self):
        barrier = threading.Barrier(8)

        def worker(index):
            source = _rgba(np.uint16, height=3, width=5)
            source[..., 0] += index
            barrier.wait(timeout=20)
            encoded = rs_jxl.encode(source, effort=3, threads=1)
            return source, rs_jxl.decode(encoded, threads=1)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(8)]
            for future in futures:
                source, decoded = future.result(timeout=60)
                self.assertPixels(decoded, source)

    def test_auto_threads_and_small_effort9_lossless(self):
        source = _rgba(height=2, width=3)
        encoded = rs_jxl.encode(source, effort=9, threads=0)
        self.assertPixels(rs_jxl.decode(encoded, threads=0), source)

    def test_screenshot_recipe_effort9_is_lossless(self):
        source = np.arange(32 * 48 * 3, dtype=np.uint16).reshape(32, 48, 3)
        source = (source % 251).astype(np.uint8)
        encoded = rs_jxl.encode(source, effort=9, threads=2, modular_group_size=3,
                                iterations=100, modular_predictor=0,
                                modular_palette_colors=10000, patches=False,
                                modular_colorspace=0)
        self.assertIsInstance(encoded, bytes)
        self.assertPixels(rs_jxl.decode(encoded, threads=2), source)


class MultiframeTests(PixelAssertions, unittest.TestCase):
    def test_difference_bounds_include_hidden_rgb(self):
        report = rs_jxl.analyze_multiframe(_frames())
        frames = report["frames"]
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[1]["previous"], {"x": 1, "y": 0, "width": 1, "height": 1})
        self.assertEqual(frames[2]["previous"], {"x": 1, "y": 0, "width": 4, "height": 3})
        self.assertEqual(frames[2]["first"], {"x": 4, "y": 2, "width": 1, "height": 1})
        self.assertIsNone(frames[3]["previous"])

    def test_reference_modes_crop_geometry_and_owned_frames(self):
        frames = _frames()
        for reference in ("auto", "first", "previous", "full"):
            with self.subTest(reference=reference):
                encoded = rs_jxl.encode_multiframe(frames, durations=[2, 3, 5, 7],
                                                   reference=reference, strategy="crop",
                                                   effort=3, threads=1)
                decoded, metadata = rs_jxl.decode(encoded, frame=None, return_info=True, threads=1)
                self.assertIsInstance(decoded, list)
                self.assertEqual(len(decoded), len(frames))
                self.assertEqual([f["duration"] for f in metadata["frames"]], [2, 3, 5, 7])
                for actual, expected in zip(decoded, frames):
                    self.assertPixels(actual, expected)
                self.assertPixels(rs_jxl.decode(encoded, frame=2, threads=1), frames[2])
                layer, details = rs_jxl.decode_layer(encoded, layer=2, threads=1)
                layout = details["header"]["layer_info"]
                x, y, width, height, source = {
                    "auto": (4, 2, 1, 1, 1), "first": (4, 2, 1, 1, 1),
                    "previous": (1, 0, 4, 3, 2), "full": (0, 0, 7, 5, 0),
                }[reference]
                self.assertEqual(layout["crop_origin"], [x, y])
                self.assertEqual((layout["width"], layout["height"]), (width, height))
                self.assertEqual(layout["blend_info"]["source"], source)
                self.assertPixels(layer, frames[2][y:y + height, x:x + width])
                self.assertFalse(np.shares_memory(decoded[2], decoded[3]))
                decoded[2].fill(0)
                self.assertPixels(decoded[3], frames[3])

    def test_zero_duration_layers_are_not_display_frames(self):
        frames = _frames()[:3]
        encoded = rs_jxl.encode_multiframe(frames, durations=[0, 0, 9],
                                           reference="previous", effort=3, threads=1)
        headers = rs_jxl.info(encoded, coalesced=False, threads=1)["frames"]
        self.assertEqual([header["duration"] for header in headers], [0, 0, 9])
        displayed = rs_jxl.decode(encoded, frame=None, threads=1)
        self.assertEqual(len(displayed), 1)
        self.assertPixels(displayed[0], frames[2])
        self.assertPixels(rs_jxl.decode(encoded, threads=1), frames[2])
        layers, metadata = rs_jxl.decode(encoded, frame=None, coalesced=False,
                                         return_info=True, threads=1)
        self.assertEqual(len(layers), 3)
        self.assertEqual([frame["duration"] for frame in metadata["frames"]], [0, 0, 9])
        first, details = rs_jxl.decode_layer(encoded, layer=0, return_info=True, threads=1)
        self.assertPixels(first, frames[0])
        self.assertEqual(details["duration"], 0)
        second, details = rs_jxl.decode_layer(encoded, layer=1, threads=1)
        self.assertPixels(second, frames[1][0:1, 1:2])
        self.assertEqual(details["duration"], 0)
        self.assertEqual(details["header"]["layer_info"]["crop_origin"], [1, 0])

    def test_auto_strategy_and_scalar_duration(self):
        frames = [frame[:3, :4, :3] for frame in _frames()[:2]]
        encoded = rs_jxl.encode_multiframe(frames, durations=2, strategy="auto",
                                           effort=3, threads=1)
        decoded, metadata = rs_jxl.decode(encoded, frame=None, return_info=True, threads=1)
        self.assertEqual(len(decoded), 2)
        for actual, expected in zip(decoded, frames):
            self.assertPixels(actual, expected)
        self.assertEqual([f["duration"] for f in metadata["frames"]], [2, 2])

    def test_all_zero_durations_remain_zero(self):
        frames = _frames()[:2]
        encoded = rs_jxl.encode_multiframe(frames, durations=0, reference="previous",
                                           effort=3, threads=1)
        headers = rs_jxl.info(encoded, coalesced=False, threads=1)["frames"]
        self.assertEqual([header["duration"] for header in headers], [0, 0])
        visible = rs_jxl.decode(encoded, frame=None, threads=1)
        self.assertEqual(len(visible), 1)
        self.assertPixels(visible[0], frames[-1])
        layer, metadata = rs_jxl.decode_layer(encoded, layer=1, threads=1)
        self.assertEqual(metadata["duration"], 0)
        self.assertPixels(layer, frames[1][0:1, 1:2])

    def test_per_frame_extra_planes_affect_difference_bounds(self):
        image = _rgba()
        depths = [np.zeros((5, 7), dtype=np.uint16)]
        depths.append(depths[0].copy())
        depths[1][2, 3] = 65535
        extras = [{"data": depths, "type": "depth", "name": "distance"}]
        report = rs_jxl.analyze_multiframe([image, image], extra_channels=extras)
        self.assertEqual(report["frames"][1]["previous"],
                         {"x": 3, "y": 2, "width": 1, "height": 1})
        encoded = rs_jxl.encode_multiframe([image, image], extra_channels=extras,
                                           reference="previous", effort=3, threads=1)
        decoded, metadata = rs_jxl.decode(encoded, frame=None, return_extra_channels=True,
                                          threads=1)
        self.assertEqual(len(decoded), 2)
        self.assertEqual(len(metadata["frames"]), 2)
        for index, frame in enumerate(metadata["frames"]):
            self.assertPixels(decoded[index], image)
            self.assertEqual(len(frame["extra_channels"]), 1)
            self.assertEqual(frame["extra_channels"][0]["name"], "distance")
            self.assertPixels(frame["extra_channels"][0]["data"], depths[index])

    def test_invalid_sequence_parameters(self):
        for frames, options in (([], {}), (_frames(), {"durations": [1]}),
                                (_frames(), {"durations": -1}),
                                (_frames(), {"reference": "unknown"}),
                                (_frames(), {"strategy": "unknown"})):
            with self.subTest(options=options), self.assertRaises(ValueError):
                rs_jxl.encode_multiframe(frames, effort=3, threads=1, **options)


class JpegTests(PixelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jpeg = _tiny_jpeg()
        cls.encoded = rs_jxl.transcode_jpeg(cls.jpeg, threads=1)  # Public effort=9 default.

    def test_default_transcode_recovers_original_bytes(self):
        self.assertEqual(rs_jxl.reconstruct_jpeg(self.encoded, threads=1), self.jpeg)
        self.assertPixels(rs_jxl.decode(self.jpeg, threads=1), np.array([[128]], dtype=np.uint8))
        self.assertPixels(rs_jxl.decode(self.encoded, threads=1), np.array([[128]], dtype=np.uint8))

    def test_encode_jpeg_lossless_default_and_explicit_pixel_route(self):
        encoded = rs_jxl.encode(io.BytesIO(self.jpeg), effort=3, threads=1)
        self.assertEqual(rs_jxl.reconstruct_jpeg(encoded, threads=1), self.jpeg)
        pixels = rs_jxl.encode(self.jpeg, lossless_jpeg=False, effort=3, threads=1)
        self.assertPixels(rs_jxl.decode(pixels, threads=1), np.array([[128]], dtype=np.uint8))
        with self.assertRaises(rs_jxl.CodecError):
            rs_jxl.reconstruct_jpeg(pixels, threads=1)

    def test_jpeg_path_and_stream_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.jpg"
            source.write_bytes(self.jpeg)
            destination = Path(directory) / "archival.jxl"
            rs_jxl.transcode_jpeg(source, destination, effort=3, threads=1)
            restored = Path(directory) / "restored.jpg"
            rs_jxl.reconstruct_jpeg(destination, restored, threads=1)
            self.assertEqual(restored.read_bytes(), self.jpeg)
            stream = io.BytesIO()
            self.assertIsNone(rs_jxl.reconstruct_jpeg(destination, stream, threads=1))
            self.assertEqual(stream.getvalue(), self.jpeg)


class TorchTests(PixelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(f"optional Torch import unavailable: {error}") from error
        cls.torch = torch

    def test_cpu_auto_chw_and_hwc_owned_output(self):
        source = _rgba()
        tensor = self.torch.from_numpy(source.copy()).permute(2, 0, 1)
        self.assertFalse(tensor.is_contiguous())
        encoded = rs_jxl.encode(tensor, effort=3, threads=1)
        self.assertPixels(rs_jxl.decode(encoded, threads=1), source)
        first = rs_jxl.decode(encoded, out="torch", threads=1)
        second = rs_jxl.decode(encoded, out="torch", threads=1)
        self.assertEqual(first.device.type, "cpu")
        self.assertEqual(first.dtype, self.torch.uint8)
        self.assertEqual(tuple(first.shape), source.shape)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        del encoded
        gc.collect()
        self.assertPixels(first.numpy(), source)
        first.zero_()
        self.assertPixels(second.numpy(), source)

    def test_cpu_gray_and_explicit_hwc(self):
        gray = np.arange(20, dtype=np.uint8).reshape(4, 5)
        for source, layout in ((gray, "auto"), (_rgba(), "hwc")):
            with self.subTest(layout=layout):
                tensor = self.torch.from_numpy(source.copy())
                encoded = rs_jxl.encode(tensor, layout=layout, effort=3, threads=1)
                self.assertPixels(rs_jxl.decode(encoded, out="torch", threads=1).numpy(), source)

    @unittest.skipUnless(os.environ.get("RS_JXL_TEST_CUDA") == "1", "CUDA transfer is opt-in")
    def test_gpu_transfer_requires_explicit_permission(self):
        self.assertTrue(self.torch.cuda.is_available(), "RS_JXL_TEST_CUDA=1 requires usable CUDA")
        source = _rgba()
        tensor = self.torch.from_numpy(source.copy()).permute(2, 0, 1).to("cuda")
        with self.assertRaises(ValueError):
            rs_jxl.encode(tensor, effort=3, threads=1)
        encoded = rs_jxl.encode(tensor, allow_device_copy=True, effort=3, threads=1)
        output = rs_jxl.decode(encoded, out="torch", device="cuda", threads=1)
        self.assertEqual(output.device.type, "cuda")
        self.assertPixels(output.cpu().numpy(), source)


@unittest.skipUnless("RS_JXL_DJXL" in os.environ, "official djxl is opt-in via RS_JXL_DJXL")
class OfficialDecoderTests(PixelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entries = os.environ["RS_JXL_DJXL"].split(os.pathsep)
        cls.decoders = []
        for entry in entries:
            if entry.startswith('"') and entry.endswith('"'):
                entry = entry[1:-1]
            if not entry or not Path(entry).is_file():
                raise AssertionError(f"RS_JXL_DJXL contains an invalid executable: {entry!r}")
            cls.decoders.append(Path(entry).resolve())

    def runDecoder(self, executable, encoded, destination, directory):
        source = Path(directory) / "input.jxl"
        source.write_bytes(encoded)
        self.assertFalse(destination.exists(), "oracle output must not be stale")
        command = [str(executable), str(source), str(destination), "--num_threads=1", "--quiet"]
        if destination.suffix == ".npy":
            command.append("--output_frames")
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=30, check=False)
        self.assertEqual(result.returncode, 0,
                         f"{executable}: stdout={result.stdout!r}; stderr={result.stderr!r}")
        self.assertTrue(destination.is_file(), "djxl succeeded without creating output")

    def test_official_hidden_rgba8_rgba16_and_zero_duration_frames(self):
        fixtures = []
        for dtype in (np.uint8, np.uint16):
            source = _rgba(dtype)
            fixtures.append((rs_jxl.encode(source, effort=3, threads=1), source[None]))
        frames = _frames()[:3]
        fixtures.append((rs_jxl.encode_multiframe(frames, durations=[0, 0, 1],
                                                  reference="previous", effort=3, threads=1),
                         frames[-1][None]))
        for decoder in self.decoders:
            for index, (encoded, expected) in enumerate(fixtures):
                with self.subTest(decoder=decoder, fixture=index), tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / "pixels.npy"
                    self.runDecoder(decoder, encoded, destination, directory)
                    actual = np.load(destination, allow_pickle=False)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, np.dtype("<f4"))
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertTrue(np.all((actual >= 0) & (actual <= 1)))
                    maximum = np.iinfo(expected.dtype).max
                    recovered = np.rint(actual.astype(np.float64) * maximum).astype(expected.dtype)
                    np.testing.assert_array_equal(recovered, expected)

    def test_official_byte_exact_jpeg_reconstruction(self):
        jpeg = _tiny_jpeg()
        encoded = rs_jxl.transcode_jpeg(jpeg, effort=3, threads=1)
        for decoder in self.decoders:
            with self.subTest(decoder=decoder), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "restored.jpg"
                self.runDecoder(decoder, encoded, destination, directory)
                self.assertEqual(destination.read_bytes(), jpeg)


if __name__ == "__main__":
    unittest.main()
