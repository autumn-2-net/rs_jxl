"""Layer request adapters and small effort-3 native fixtures; no Python compositor."""

from collections.abc import Mapping
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import Mock, patch

import numpy as np

import rs_jxl
from rs_jxl import _ffi, contexts, layers

try:
    import test_bindings as fixtures
except ModuleNotFoundError as error:
    if error.name != "test_bindings":
        raise
    spec = importlib.util.spec_from_file_location(
        "_layer_fixtures", Path(__file__).with_name("test_bindings.py"))
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)

try:
    import test_multiframe_analysis as png_fixtures
except ModuleNotFoundError as error:
    if error.name != "test_multiframe_analysis":
        raise
    spec = importlib.util.spec_from_file_location(
        "_layer_png_fixtures", Path(__file__).with_name("test_multiframe_analysis.py"))
    png_fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(png_fixtures)


class LayerAdapters(unittest.TestCase):
    def setUp(self):
        self.images = [fixtures._rgba(height=2, width=3),
                       fixtures._rgba(height=4, width=2)]

    def request(self, **options):
        with patch.object(layers, "execute", return_value=(b"encoded", {})) as execute:
            self.assertEqual(rs_jxl.encode_layers(self.images, **options), b"encoded")
        return execute.call_args.args[0]

    def test_defaults_preserve_raw_sizes_bytes_and_native_defaults(self):
        with patch.object(layers, "execute", return_value=(b"encoded", {})) as execute:
            rs_jxl.encode_layers(iter(self.images))
        request, payload = execute.call_args.args
        self.assertIn("encode_layers", rs_jxl.__all__)
        self.assertEqual(request["operation"], "encode_layers")
        self.assertEqual(request["layer_options"], {})
        self.assertEqual(request["options"], {"effort": 7, "threads": 1,
                                             "source_search": False, "strip_metadata": False})
        self.assertEqual([(spec["width"], spec["height"]) for spec in request["images"]],
                         [(3, 2), (2, 4)])
        self.assertEqual(bytes(payload), b"".join(image.tobytes() for image in self.images))
        self.assertNotIn("durations", request)
        self.assertNotIn("timebase", request)
        self.assertNotIn("auto", request)

    def test_mappings_normalize_numbers_without_mutating_inputs(self):
        blend = MappingProxyType({"mode": "blend", "source": np.uint8(2),
                                  "alpha_channel": np.int64(0), "clamp": True})
        frame = MappingProxyType({"origin": (np.int32(-1), np.int64(2)),
                                  "duration": np.uint32(7), "timecode": np.uint32(9),
                                  "name": "overlay", "blend": blend,
                                  "extra_blend": [blend], "save_as_reference": np.uint8(2)})
        request = self.request(canvas=(np.uint32(8), np.int64(6)),
                               layer_options=[frame, {}], tps=(np.uint32(24), np.int32(1)),
                               loops=np.uint32(3), encoder_options={"rct": 0},
                               modular_predictor=0)
        options = request["layer_options"]
        self.assertEqual(options["canvas"], [8, 6])
        self.assertEqual(options["frames"][0], {"origin": [-1, 2], "duration": 7,
                         "timecode": 9, "name": "overlay", "blend": dict(blend),
                         "extra_blend": [dict(blend)], "save_as_reference": 2})
        self.assertEqual(options["frames"][1], {})
        self.assertEqual(options["animation"], {"ticks_per_second_numerator": 24,
                         "ticks_per_second_denominator": 1, "loops": 3, "timecodes": True})
        self.assertIs(type(options["frames"][0]["blend"]["source"]), int)
        self.assertIsNot(options["frames"][0]["blend"], blend)
        self.assertIsInstance(frame["origin"], tuple)
        self.assertIsInstance(frame["blend"]["source"], np.uint8)
        self.assertEqual(request["options"]["rct"], 0)
        self.assertEqual(request["options"]["predictor"], 0)
        self.assertNotIn("frames", request["options"])
        json.dumps(request)

    def test_animation_is_explicit_only_for_timebase_loops_or_nonzero_timecodes(self):
        for frames in ([{}, {}], [{"duration": 0}, {}], [{"duration": 5}, {}],
                       [{"timecode": 0}, {}]):
            with self.subTest(frames=frames):
                self.assertNotIn("animation", self.request(layer_options=frames)["layer_options"])
        cases = [({"tps": (25, 2)}, 25, 2, 0, False),
                 ({"loops": 4}, 1000, 1, 4, False),
                 ({"layer_options": [{"timecode": 17}, {}]}, 1000, 1, 0, True)]
        for arguments, numerator, denominator, loops, timecodes in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.request(**arguments)["layer_options"]["animation"],
                                 {"ticks_per_second_numerator": numerator,
                                  "ticks_per_second_denominator": denominator,
                                  "loops": loops, "timecodes": timecodes})

    def test_structural_native_ranges_are_forwarded_without_reference_algorithms(self):
        request = self.request(canvas=(0xffffffff, 1), tps=(0xffffffff, 0xffffffff),
                               layer_options=[{"origin": (-(1 << 31), (1 << 31) - 1),
                                               "duration": 0xffffffff, "timecode": 0xffffffff,
                                               "save_as_reference": 0xff,
                                               "blend": {"source": 0xff, "alpha_channel": 0xff}}, {}])
        self.assertEqual(request["layer_options"]["frames"][0]["blend"]["source"], 0xff)
        self.assertEqual(request["layer_options"]["canvas"][0], 0xffffffff)
        self.assertTrue(self.request(source_search=True)["options"]["source_search"])
        for mode in ("replace", "add", "blend", "alpha_weighted_add", "multiply"):
            self.assertEqual(self.request(layer_options=[{"blend": {"mode": mode}}, {}])
                             ["layer_options"]["frames"][0]["blend"]["mode"], mode)

    def test_invalid_options_fail_before_consuming_images(self):
        class UnreadImages:
            def __iter__(self):
                raise AssertionError("invalid options consumed images")

        invalid = []
        for key in ("canvas", "tps"):
            for value in (1, True, "12", {0: 1, 1: 2}):
                invalid.append(({key: value}, TypeError))
            for value in ((), (1,), (1, 2, 3), (0, 1), (-1, 1), (1 << 32, 1)):
                invalid.append(({key: value}, ValueError))
            for value in (True, np.bool_(False), 1.5, "2", None):
                invalid.append(({key: (value, 1)}, TypeError))
        for value in ({}, "frames", 1, [None], [1], [[("duration", 1)]]):
            invalid.append(({"layer_options": value}, TypeError))
        for field in ("unknown", "reference_only", "save_before_color_transform", "animation"):
            invalid.append(({"layer_options": [{field: True}]}, TypeError))
        for field in ("duration", "timecode", "save_as_reference"):
            for value in (True, np.bool_(True), 1.5, "1", None):
                invalid.append(({"layer_options": [{field: value}]}, TypeError))
            invalid.append(({"layer_options": [{field: -1}]}, ValueError))
            maximum = 1 << (8 if field == "save_as_reference" else 32)
            invalid.append(({"layer_options": [{field: maximum}]}, ValueError))
        for value in (True, np.bool_(True), 1.5, "1", None):
            invalid.append(({"loops": value}, TypeError))
            invalid.append(({"layer_options": [{"origin": (0, value)}]}, TypeError))
        invalid.extend([({"loops": -1}, ValueError), ({"loops": 1 << 32}, ValueError)])
        for value in (None, False, b"name", 1):
            invalid.append(({"layer_options": [{"name": value}]}, TypeError))
        for value in (None, [], "replace", True):
            invalid.append(({"layer_options": [{"blend": value}]}, TypeError))
        for value in (None, {}, "blend", [None]):
            invalid.append(({"layer_options": [{"extra_blend": value}]}, TypeError))
        invalid.extend([({"layer_options": [{"blend": {"unknown": 0}}]}, TypeError),
                        ({"layer_options": [{"blend": {"mode": "over"}}]}, ValueError),
                        ({"layer_options": [{"blend": {"mode": 0}}]}, TypeError),
                        ({"layer_options": [{"origin": (1,)}]}, ValueError),
                        ({"layer_options": [{"origin": (1 << 31, 0)}]}, ValueError),
                        ({"layer_options": [{"origin": (-(1 << 31) - 1, 0)}]}, ValueError)])
        for value in (0, 1, "true", None, np.bool_(True)):
            invalid.append(({"layer_options": [{"blend": {"clamp": value}}]}, TypeError))
            invalid.append(({"source_search": value}, TypeError))
        for key in ("source", "alpha_channel"):
            for value in (True, np.bool_(False), 1.5, "1", None):
                invalid.append(({"layer_options": [{"blend": {key: value}}]}, TypeError))
            for value in (-1, 256):
                invalid.append(({"layer_options": [{"blend": {key: value}}]}, ValueError))
        invalid.extend([({"lossless": False}, ValueError), ({"distance": 1}, ValueError),
                        ({"effort": True}, TypeError), ({"threads": True}, TypeError),
                        ({"encoder_options": {"effort": 3}}, ValueError),
                        ({"encoder_options": {"rct": 0}, "modular_colorspace": 1}, ValueError)])
        with patch.object(layers, "execute") as execute:
            for options, error in invalid:
                with self.subTest(options=options), self.assertRaises(error):
                    rs_jxl.encode_layers(UnreadImages(), **options)
            execute.assert_not_called()

    def test_duplicate_frame_and_blend_fields_are_rejected(self):
        class Duplicate(Mapping):
            def __init__(self, key):
                self.key = key

            def __iter__(self):
                return iter((self.key, self.key))

            def __len__(self):
                return 2

            def __getitem__(self, key):
                return 0

        for frame in (Duplicate("duration"), {"blend": Duplicate("source")},
                      {"extra_blend": [Duplicate("source")]}):
            with self.subTest(frame=frame), self.assertRaisesRegex(ValueError, "duplicate"):
                self.request(layer_options=[frame, {}])

    def test_empty_images_and_frame_count_mismatch(self):
        with patch.object(layers, "execute") as execute:
            with self.assertRaises(ValueError):
                rs_jxl.encode_layers([])
            for options in ([], [{}], [{}, {}, {}]):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    rs_jxl.encode_layers(self.images, layer_options=options)
            execute.assert_not_called()

    def test_shared_transport_receives_layout_metadata_and_borrowing_controls(self):
        extras = [{"type": "depth", "data": [np.zeros(image.shape[:2], dtype=np.uint16)
                                                for image in self.images]}]
        with patch.object(layers, "_images", wraps=layers._images) as transport:
            request = self.request(extra_channels=extras, borrow_inputs=True,
                                   bits_per_sample=8, color_encoding="linear_srgb",
                                   allow_device_copy=True, strip_metadata=True)
        self.assertTrue(transport.call_args.kwargs["borrow_inputs"])
        self.assertTrue(transport.call_args.kwargs["allow_device_copy"])
        self.assertTrue(request["options"]["strip_metadata"])
        for spec in request["images"]:
            self.assertEqual(spec["bits_per_sample"], 8)
            self.assertEqual(spec["color_encoding"], "linear_srgb")
            self.assertEqual(spec["extra_channels"][0]["type"], "depth")

    def test_report_output_and_context_forwarding(self):
        report = {"operation": "encode_layers", "bytes": 7, "jpeg_input": False,
                  "canvas": {"width": 3, "height": 4}, "codec_seconds": 0.125,
                  "layers": [{"index": 0, "width": 3, "height": 2, "origin": [0, 0],
                              "duration": 0, "timecode": 0, "name": "", "extra_blend": [],
                              "blend": {"mode": "replace", "source": 1,
                                        "alpha_channel": 0, "clamp": False},
                              "save_as_reference": 0}]}
        context = object()
        with tempfile.TemporaryDirectory() as directory:
            path, stream = Path(directory) / "layers.jxl", io.BytesIO()
            for output in (None, path, stream):
                with patch.object(layers, "execute", return_value=(b"encoded", report)) as execute:
                    result, actual = rs_jxl.encode_layers(self.images, output, return_info=True,
                                                          _context=context)
                self.assertIs(actual, report)
                self.assertIs(execute.call_args.kwargs["context"], context)
                self.assertEqual(result, b"encoded" if output is None else path if output == path else None)
            self.assertEqual(path.read_bytes(), b"encoded")
            self.assertEqual(stream.getvalue(), b"encoded")

    def test_bound_context_inherits_workers_and_rejects_overrides_and_close(self):
        native = Mock(threads=2, closed=False)
        with patch.object(contexts, "NativeContext", return_value=native):
            encoder = rs_jxl.Encoder(threads=2)
        with patch.object(layers, "execute", return_value=(b"encoded", {})) as execute:
            self.assertEqual(encoder.encode_layers(self.images, effort=3), b"encoded")
            self.assertEqual(execute.call_args.args[0]["options"]["threads"], 2)
            self.assertIs(execute.call_args.kwargs["context"], native)
            with self.assertRaises(ValueError):
                encoder.encode_layers(self.images, threads=1)
            with self.assertRaises(TypeError):
                encoder.encode_layers(self.images, _context=object())
            native.closed = True
            with self.assertRaises(RuntimeError):
                encoder.encode_layers(self.images)
            self.assertEqual(execute.call_count, 1)


class NativeLayers(fixtures.PixelAssertions, unittest.TestCase):
    def test_default_heterogeneous_layers_preserve_hidden_rgb_and_have_blank_background(self):
        for dtype in (np.uint8, np.uint16):
            images = [fixtures._rgba(dtype, 3, 5), fixtures._rgba(dtype, 4, 2),
                      fixtures._rgba(dtype, 2, 3)]
            with self.subTest(dtype=dtype):
                data, report = rs_jxl.encode_layers(images, effort=3, return_info=True)
                self.assertEqual(report["operation"], "encode_layers")
                self.assertEqual(report["bytes"], len(data))
                self.assertFalse(report["jpeg_input"])
                self.assertEqual(report["canvas"], {"width": 5, "height": 4})
                self.assertTrue(math.isfinite(report["codec_seconds"]))
                self.assertGreaterEqual(report["codec_seconds"], 0)
                self.assertEqual(len(report["layers"]), len(images))
                metadata = rs_jxl.info(data, coalesced=False)
                self.assertIsNone(metadata["animation"])
                self.assertEqual(len(metadata["frames"]), len(images))
                for index, image in enumerate(images):
                    self.assertTrue(np.any(image[..., :3][image[..., 3] == 0]))
                    actual, details = rs_jxl.decode_layer(data, layer=index)
                    self.assertPixels(actual, image)
                    self.assertEqual(details["duration"], 0)
                    self.assertEqual(report["layers"][index], {
                        "index": index, "width": image.shape[1], "height": image.shape[0],
                        "origin": [0, 0], "duration": 0, "timecode": 0, "name": "",
                        "blend": {"mode": "replace", "source": 1,
                                  "alpha_channel": 0, "clamp": False},
                        "extra_blend": [], "save_as_reference": 0})
                displayed = rs_jxl.decode(data, frame=None)
                self.assertEqual(len(displayed), 1)
                expected = np.zeros((4, 5, 4), dtype=dtype)
                expected[:2, :3] = images[-1]
                self.assertPixels(displayed[0], expected)

    def test_positive_origins_infer_extent_and_negative_origins_require_canvas(self):
        images = [fixtures._rgba(height=2, width=3), fixtures._rgba(height=3, width=2)]
        options = [{"origin": (4, 1)}, {"origin": (1, 3), "name": "offset"}]
        data, report = rs_jxl.encode_layers(images, layer_options=options, effort=3, return_info=True)
        self.assertEqual(report["canvas"], {"width": 7, "height": 6})
        for index, image in enumerate(images):
            raw, details = rs_jxl.decode_layer(data, layer=index)
            self.assertPixels(raw, image)
            self.assertEqual(details["header"]["layer_info"]["crop_origin"], list(options[index]["origin"]))
        expected = np.zeros((6, 7, 4), dtype=np.uint8)
        expected[3:6, 1:3] = images[1]
        self.assertPixels(rs_jxl.decode(data), expected)
        with self.assertRaises(rs_jxl.CodecError):
            rs_jxl.encode_layers(images, layer_options=[{}, {"origin": (-1, 0)}], effort=3)

    def test_explicit_canvas_clips_only_composite_not_raw_samples(self):
        image = fixtures._rgba(np.uint16, 4, 5)
        for origin, visible in (((-1, -1), image[1:3, 1:4]), ((0, 0), image[:2, :3])):
            with self.subTest(origin=origin):
                data = rs_jxl.encode_layers([image], canvas=(3, 2),
                                            layer_options=[{"origin": origin}], effort=3)
                raw, details = rs_jxl.decode_layer(data)
                self.assertPixels(raw, image)
                self.assertEqual(details["header"]["layer_info"]["crop_origin"], list(origin))
                self.assertPixels(rs_jxl.decode(data), visible)

    def test_path_bytes_stream_and_array_inputs_share_pixel_transport(self):
        images = [fixtures._rgba(np.uint8, 2, 3), fixtures._rgba(np.uint8, 3, 2),
                  fixtures._rgba(np.uint8, 4, 2), fixtures._rgba(np.uint8, 2, 4)]
        # This fixture uses structured sRGB, matching the raw array metadata.
        pngs = [png_fixtures._png(image) for image in images[:3]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            path.write_bytes(pngs[0])
            data = rs_jxl.encode_layers([path, pngs[1], io.BytesIO(pngs[2]), images[3]], effort=3)
        for index, expected in enumerate(images):
            self.assertPixels(rs_jxl.decode_layer(data, layer=index)[0], expected)

    def test_uint16_encoded_and_array_sequences_use_matching_metadata(self):
        images = [fixtures._rgba(np.uint16, 2, 3), fixtures._rgba(np.uint16, 3, 2),
                  fixtures._rgba(np.uint16, 4, 2)]
        pngs = [rs_jxl.decode_to_png(rs_jxl.encode(image, effort=3)) for image in images]
        # Exported PNGs share a generated ICC; arrays use structured sRGB.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input16.png"
            path.write_bytes(pngs[0])
            encoded_data = rs_jxl.encode_layers([path, pngs[1], io.BytesIO(pngs[2])], effort=3)
        array_data = rs_jxl.encode_layers(images, effort=3)
        for route, data in (("encoded", encoded_data), ("array", array_data)):
            with self.subTest(route=route):
                for index, expected in enumerate(images):
                    self.assertPixels(rs_jxl.decode_layer(data, layer=index)[0], expected)

    def test_generated_icc_png_and_default_srgb_array_are_not_silently_converted(self):
        image = fixtures._rgba(np.uint16, 2, 3)
        png = rs_jxl.decode_to_png(rs_jxl.encode(image, effort=3))
        decoded, metadata = rs_jxl.decode(png, return_info=True)
        self.assertEqual(metadata["color_encoding"]["kind"], "icc")
        self.assertPixels(decoded, image)
        for encoded_first in (False, True):
            inputs = [png, image] if encoded_first else [image, png]
            for strip_metadata in (False, True):
                with self.subTest(encoded_first=encoded_first, strip_metadata=strip_metadata):
                    with self.assertRaises(rs_jxl.CodecError):
                        rs_jxl.encode_layers(inputs, effort=3, strip_metadata=strip_metadata)

    def test_jpeg_inputs_are_pixels_not_byte_transcoding(self):
        jpeg = fixtures._tiny_jpeg()
        expected = rs_jxl.decode(jpeg)
        data, report = rs_jxl.encode_layers([jpeg, expected], effort=3, return_info=True)
        self.assertTrue(report["jpeg_input"])
        for index in range(2):
            self.assertPixels(rs_jxl.decode_layer(data, layer=index)[0], expected)
        with self.assertRaises(rs_jxl.CodecError):
            rs_jxl.reconstruct_jpeg(data)

    def test_animation_inference_explicit_timebase_loops_and_timecodes(self):
        images = [fixtures._rgba(height=2, width=3), fixtures._rgba(height=3, width=2)]
        cases = [({"layer_options": [{"duration": 2}, {"duration": 3}]}, 1000, 1, 0, False),
                 ({"tps": (24, 1)}, 24, 1, 0, False),
                 ({"loops": 2}, 1000, 1, 2, False),
                 ({"layer_options": [{"timecode": 9}, {"timecode": 11}]}, 1000, 1, 0, True)]
        for options, numerator, denominator, loops, timecodes in cases:
            with self.subTest(options=options):
                data = rs_jxl.encode_layers(images, effort=3, **options)
                metadata = rs_jxl.info(data, coalesced=False)
                self.assertEqual(metadata["animation"], {"tps_numerator": numerator,
                                 "tps_denominator": denominator, "num_loops": loops,
                                 "have_timecodes": timecodes})
                for index, expected in enumerate(images):
                    self.assertPixels(rs_jxl.decode_layer(data, layer=index)[0], expected)
                if timecodes:
                    self.assertEqual([frame["timecode"] for frame in metadata["frames"]], [9, 11])

    def test_source_zero_reuses_previous_zero_duration_composite(self):
        images = [np.full((4, 5, 3), 11, dtype=np.uint8),
                  np.full((2, 2, 3), 23, dtype=np.uint8),
                  np.full((1, 1, 3), 31, dtype=np.uint8)]
        options = [{}, {"origin": (1, 1), "blend": {"source": 0}},
                   {"origin": (3, 2), "blend": {"source": 0}}]
        data = rs_jxl.encode_layers(images, layer_options=options, effort=3)
        expected = images[0].copy()
        expected[1:3, 1:3] = images[1]
        expected[2:3, 3:4] = images[2]
        self.assertPixels(rs_jxl.decode(data), expected)
        for index, image in enumerate(images):
            self.assertPixels(rs_jxl.decode_layer(data, layer=index)[0], image)

    def test_timed_frames_need_explicit_saved_reference_to_reuse_pixels(self):
        images = [np.full((4, 5, 3), 11, dtype=np.uint8),
                  np.full((2, 2, 3), 23, dtype=np.uint8)]
        for save, source in ((0, 0), (2, 2)):
            with self.subTest(save=save):
                options = [{"duration": 1, "save_as_reference": save},
                           {"duration": 1, "origin": (1, 1), "blend": {"source": source}}]
                data = rs_jxl.encode_layers(images, layer_options=options, effort=3)
                displayed = rs_jxl.decode(data, frame=None)
                self.assertEqual(len(displayed), 2)
                self.assertPixels(displayed[0], images[0])
                expected = images[0].copy() if save else np.zeros_like(images[0])
                expected[1:3, 1:3] = images[1]
                self.assertPixels(displayed[1], expected)
                self.assertPixels(rs_jxl.decode_layer(data, layer=1)[0], images[1])

    def test_add_blend_keeps_raw_layer_samples(self):
        base = np.full((3, 4, 3), 11, dtype=np.uint8)
        addition = np.full((2, 2, 3), 23, dtype=np.uint8)
        data = rs_jxl.encode_layers([base, addition], effort=3,
                                   layer_options=[{}, {"origin": (1, 1),
                                                       "blend": {"mode": "add", "source": 0}}])
        expected = base.copy()
        expected[1:3, 1:3] = 34
        self.assertPixels(rs_jxl.decode(data), expected)
        self.assertPixels(rs_jxl.decode_layer(data, layer=1)[0], addition)

    def test_per_layer_extra_planes_and_extra_blend(self):
        images = [fixtures._rgba(np.uint16, 2, 3), fixtures._rgba(np.uint16, 3, 2)]
        depths = [np.arange(image.shape[0] * image.shape[1], dtype=np.uint16).reshape(image.shape[:2])
                  for image in images]
        options = [{"extra_blend": [{"source": 1}, {"source": 1}]} for _ in images]
        data = rs_jxl.encode_layers(images, effort=3, layer_options=options,
                                   extra_channels=[{"data": depths, "type": "depth", "name": "depth"}])
        for index, expected in enumerate(images):
            raw, details = rs_jxl.decode_layer(data, layer=index, return_extra_channels=True)
            self.assertPixels(raw, expected)
            self.assertEqual(len(details["extra_channels"]), 1)
            self.assertPixels(details["extra_channels"][0]["data"], depths[index])

    def test_native_validation_rejects_incompatible_metadata_and_semantics(self):
        first = fixtures._rgba(height=2, width=3)
        for second in (np.zeros((3, 2, 3), dtype=np.uint8),
                       np.zeros((3, 2, 4), dtype=np.uint16)):
            with self.subTest(shape=second.shape, dtype=second.dtype), self.assertRaises(rs_jxl.CodecError):
                rs_jxl.encode_layers([first, second], effort=3)
        payload, specs = layers._images([first, first])
        specs[1]["color_encoding"] = "linear_srgb"
        with self.assertRaises(rs_jxl.CodecError):
            _ffi.execute({"operation": "encode_layers", "images": specs,
                          "options": {"effort": 3}, "layer_options": {}}, payload.data)
        invalid = [{"source_search": True}, {"unrecognized_option": True},
                   {"tps": (1 << 31, 1)}, {"tps": (1, 1025)},
                   {"layer_options": [{"save_as_reference": 1}]},
                   {"layer_options": [{"blend": {"source": 4}}]},
                   {"layer_options": [{"blend": {"alpha_channel": 1}}]},
                   {"layer_options": [{"blend": {"clamp": True}}]},
                   {"layer_options": [{"name": "x" * 1072}]},
                   {"layer_options": [{"extra_blend": [{}, {}]}]}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(rs_jxl.CodecError):
                rs_jxl.encode_layers([first], effort=3, **options)

    def test_native_infers_timecodes_when_animation_is_absent(self):
        image = fixtures._rgba(height=2, width=3)
        payload, specs = layers._images([image])
        for animation_fields in ({}, {"animation": None}):
            options = {"frames": [{"timecode": 1}], **animation_fields}
            request = {"operation": "encode_layers", "images": specs,
                       "options": {"effort": 3}, "layer_options": options}
            with self.subTest(animation_fields=animation_fields):
                data, _ = _ffi.execute(request, payload.data)
                metadata = rs_jxl.info(data, coalesced=False)
                self.assertEqual(metadata["animation"], {"tps_numerator": 1000,
                                 "tps_denominator": 1, "num_loops": 0, "have_timecodes": True})
                self.assertEqual(metadata["frames"][0]["timecode"], 1)
                self.assertEqual(metadata["frames"][0]["duration"], 0)
                self.assertPixels(rs_jxl.decode_layer(data)[0], image)

    def test_native_rejects_explicitly_disabled_timecodes(self):
        payload, specs = layers._images([fixtures._rgba(height=2, width=3)])
        for animation in ({"timecodes": False}, {"ticks_per_second_numerator": 1000,
                                                 "ticks_per_second_denominator": 1,
                                                 "loops": 0, "timecodes": False}):
            options = {"frames": [{"timecode": 1}], "animation": animation}
            request = {"operation": "encode_layers", "images": specs,
                       "options": {"effort": 3}, "layer_options": options}
            with self.subTest(animation=animation), self.assertRaises(rs_jxl.CodecError):
                _ffi.execute(request, payload.data)

    def test_native_rejects_layer_options_on_other_operations_and_automatic_fields(self):
        payload, specs = layers._images([fixtures._rgba(height=2, width=3)])
        for operation in ("encode", "encode_multiframe", "analyze_multiframe"):
            request = {"operation": operation, "images": specs, "layer_options": {}}
            with self.subTest(operation=operation), self.assertRaises(rs_jxl.CodecError):
                _ffi.execute(request, payload.data)
        for field, value in (("auto", False), ("reference", "full"), ("durations", [0]),
                             ("timebase", [1000, 1]), ("loops", 0), ("auto_options", {}),
                             ("analysis_options", {})):
            request = {"operation": "encode_layers", "images": specs,
                       "options": {"effort": 3}, "layer_options": {}, field: value}
            with self.subTest(field=field), self.assertRaises(rs_jxl.CodecError):
                _ffi.execute(request, payload.data)

    def test_context_reuse_after_error_does_not_retain_references(self):
        first = np.full((4, 5, 3), 11, dtype=np.uint8)
        second = np.full((2, 2, 3), 23, dtype=np.uint8)
        with rs_jxl.Encoder(threads=2) as encoder, rs_jxl.Decoder(threads=2) as decoder:
            data = encoder.encode_layers([first, second], effort=3,
                                         layer_options=[{"save_as_reference": 2},
                                                        {"blend": {"source": 2}}])
            self.assertPixels(decoder.decode_layer(data, layer=1)[0], second)
            with self.assertRaises(rs_jxl.CodecError):
                encoder.encode_layers([second], effort=3, source_search=True)
            data = encoder.encode_layers([second], canvas=(5, 4), effort=3,
                                         layer_options=[{"blend": {"source": 2}}])
            expected = np.zeros_like(first)
            expected[:2, :2] = second
            self.assertPixels(decoder.decode(data), expected)
        with self.assertRaises(RuntimeError):
            encoder.encode_layers([second], effort=3)

    def test_existing_multiframe_and_analysis_still_require_same_dimensions(self):
        images = [fixtures._rgba(height=2, width=3), fixtures._rgba(height=3, width=2)]
        for strategy in ("crop", "auto"):
            with self.subTest(strategy=strategy), self.assertRaises(rs_jxl.CodecError):
                rs_jxl.encode_multiframe(images, strategy=strategy, effort=3)
        with self.assertRaises(rs_jxl.CodecError):
            rs_jxl.analyze_multiframe(images)


@unittest.skipUnless("RS_JXL_DJXL" in os.environ, "official djxl is opt-in via RS_JXL_DJXL")
class OfficialLayerDecoderTests(fixtures.PixelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OfficialDecoderTests.setUpClass.__func__(cls)

    def assertOfficialPixels(self, encoded, expected):
        # Reuse executable validation, fresh outputs, one worker and the 30s
        # timeout without inheriting and rerunning unrelated oracle tests.
        for decoder in self.decoders:
            with self.subTest(decoder=decoder), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "layers.npy"
                fixtures.OfficialDecoderTests.runDecoder(self, decoder, encoded, destination, directory)
                actual = np.load(destination, allow_pickle=False)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, np.dtype("<f4"))
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue(np.all((actual >= 0) & (actual <= 1)))
                maximum = np.iinfo(expected.dtype).max
                recovered = np.rint(actual.astype(np.float64) * maximum).astype(expected.dtype)
                np.testing.assert_array_equal(recovered, expected)

    def test_default_zero_duration_layers_display_only_final_padded_layer(self):
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                images = [fixtures._rgba(dtype, 3, 5), fixtures._rgba(dtype, 4, 2),
                          fixtures._rgba(dtype, 2, 3)]
                data = rs_jxl.encode_layers(images, effort=3, threads=1)
                expected = np.zeros((1, 4, 5, 4), dtype=dtype)
                expected[0, :2, :3] = images[-1]
                self.assertTrue(np.any(expected[..., :3][expected[..., 3] == 0] != 0))
                self.assertOfficialPixels(data, expected)

    def test_unequal_timed_layers_start_partial_on_empty_source_one(self):
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                images = [fixtures._rgba(dtype, 2, 3), fixtures._rgba(dtype, 4, 2),
                          fixtures._rgba(dtype, 3, 5)]
                data = rs_jxl.encode_layers(images, effort=3, threads=1,
                                           layer_options=[{"duration": 1}, {"duration": 2},
                                                          {"duration": 3}])
                # Every source is the default blank slot 1, even after an
                # earlier timed layer. No input covers the inferred 5x4 canvas.
                expected = np.zeros((3, 4, 5, 4), dtype=dtype)
                expected[0, :2, :3] = images[0]
                expected[1, :4, :2] = images[1]
                expected[2, :3, :5] = images[2]
                self.assertTrue(np.any(expected[..., :3][expected[..., 3] == 0] != 0))
                self.assertOfficialPixels(data, expected)

    def test_timed_saved_composites_are_padded_and_clipped_without_losing_hidden_rgb(self):
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                images = [fixtures._rgba(dtype, 2, 3), fixtures._rgba(dtype, 3, 4),
                          fixtures._rgba(dtype, 3, 2)]
                options = [{"origin": (1, 1), "duration": 2, "save_as_reference": 2},
                           {"origin": (-1, 2), "duration": 3, "save_as_reference": 2,
                            "blend": {"source": 2}},
                           {"origin": (4, -1), "duration": 4, "blend": {"source": 2}}]
                data = rs_jxl.encode_layers(images, canvas=(5, 4), layer_options=options,
                                           tps=(24, 1), effort=3, threads=1)
                # Explicit expected composites, independent of the native
                # decoder: left/bottom clipping, then top/right clipping.
                expected = np.zeros((3, 4, 5, 4), dtype=dtype)
                expected[0, 1:3, 1:4] = images[0]
                expected[1] = expected[0]
                expected[1, 2:4, :3] = images[1][:2, 1:4]
                expected[2] = expected[1]
                expected[2, :2, 4:5] = images[2][1:3, :1]
                self.assertTrue(np.any(expected[..., :3][expected[..., 3] == 0] != 0))
                self.assertOfficialPixels(data, expected)


if __name__ == "__main__":
    unittest.main()
