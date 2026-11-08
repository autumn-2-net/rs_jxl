"""Small, exact combination checks through the real native Python FFI.

Native load/ABI failures deliberately fail these tests; no optional codec skips.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import rs_jxl


def _rgba(height=4, width=5):
    words = np.arange(height * width * 4, dtype=np.uint64)
    image = ((words * 1301 + 19) % 65536).astype(np.uint16)
    image = image.reshape(height, width, 4)
    image[..., 3] = 65535
    image[::2, :, 3] = 0
    image[1::2, ::2, 3] = 32769
    image[min(1, height - 1), min(1, width - 1), 3] = 1
    return image


def _planes(image):
    height, width = image.shape[:2]
    values = np.arange(height * width, dtype=np.uint32).reshape(height, width)
    return ((values * 2017 + 3).astype(np.uint16),
            ((values * 37 + 11) % 256).astype(np.uint8))


def _extras(depths, selections):
    return [{"type": "depth", "name": "distance", "data": depths},
            {"type": "selection_mask", "name": "selected", "data": selections}]


class LosslessCombinations(unittest.TestCase):
    def assertPixels(self, actual, expected):
        if expected.ndim == 2:
            expected = expected[..., None]
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def assertExtras(self, metadata, image, depth, selection):
        extras = metadata["extra_channels"]
        self.assertEqual([entry["type"] for entry in extras],
                         ["alpha", "depth", "selection_mask"])
        self.assertEqual([entry["index"] for entry in extras], [0, 1, 2])
        self.assertEqual([entry["name"] for entry in extras[1:]],
                         ["distance", "selected"])
        self.assertEqual([entry["bits_per_sample"] for entry in extras], [16, 16, 8])
        self.assertEqual([entry["dim_shift"] for entry in extras], [0, 0, 0])
        for entry, expected in zip(extras, (image[..., 3], depth, selection)):
            self.assertPixels(entry["data"], expected)

    def assertFrameMetadata(self, selected, full):
        # Selected decode merges frame fields onto image metadata. Compare the
        # frame projection only; packed-buffer offsets are call-local.
        for key, expected in full.items():
            if key not in ("offset", "extra_channels"):
                self.assertEqual(selected[key], expected, key)
        self.assertTrue(selected["complete"])
        self.assertEqual(len(selected["extra_channels"]), len(full["extra_channels"]))
        for actual, expected in zip(selected["extra_channels"], full["extra_channels"]):
            self.assertEqual({k: v for k, v in actual.items() if k not in ("offset", "data")},
                             {k: v for k, v in expected.items() if k not in ("offset", "data")})
            self.assertPixels(actual["data"], expected["data"])

    def assertSequence(self, encoded, images, depths, selections, durations):
        options = {"return_extra_channels": True, "include_alpha_extra": True,
                   "threads": 1}
        displayed, metadata = rs_jxl.decode(encoded, frame=None, **options)
        visible = [i for i, duration in enumerate(durations)
                   if duration != 0 or i == len(durations) - 1]
        self.assertEqual(len(displayed), len(visible))
        self.assertEqual(len(metadata["frames"]), len(visible))
        self.assertEqual([frame["duration"] for frame in metadata["frames"]],
                         [durations[i] for i in visible])
        for actual, details, index in zip(displayed, metadata["frames"], visible):
            self.assertPixels(actual, images[index])
            self.assertExtras(details, images[index], depths[index], selections[index])
            self.assertTrue(details["complete"])

        # Reverse/nonsequential access must agree with complete coalesced decode.
        for index in reversed(range(len(displayed))):
            selected, details = rs_jxl.decode(encoded, frame=index, **options)
            self.assertPixels(selected, displayed[index])
            self.assertFrameMetadata(details, metadata["frames"][index])

        raw, raw_meta = rs_jxl.decode(encoded, frame=None, coalesced=False, **options)
        self.assertEqual(len(raw), len(images))
        self.assertEqual(len(raw_meta["frames"]), len(images))
        self.assertEqual([frame["duration"] for frame in raw_meta["frames"]], durations)
        for index in reversed(range(len(raw))):
            details = raw_meta["frames"][index]
            layout = details["header"]["layer_info"]
            x, y = layout["crop_origin"]
            width, height = layout["width"], layout["height"]
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, images[index].shape[1])
            self.assertLessEqual(y + height, images[index].shape[0])
            crop = np.s_[y:y + height, x:x + width]
            self.assertPixels(raw[index], images[index][crop])
            self.assertExtras(details, images[index][crop], depths[index][crop],
                              selections[index][crop])
            selected, selected_meta = rs_jxl.decode_layer(encoded, layer=index, **options)
            self.assertPixels(selected, raw[index])
            self.assertFrameMetadata(selected_meta, details)

    def test_uint16_low_bits_fractional_alpha_and_full_resolution_extras(self):
        image = _rgba()
        depth, selection = _planes(image)
        self.assertTrue(np.any(image % 257 != 0))
        self.assertTrue(np.any(depth % 257 != 0))
        self.assertTrue(np.any(image[..., :3][image[..., 3] == 0] != 0))
        for effort in (3, 9):
            with self.subTest(effort=effort):
                encoded = rs_jxl.encode(image, extra_channels=_extras(depth, selection),
                                        effort=effort, threads=1)
                actual, details = rs_jxl.decode(encoded, return_extra_channels=True,
                                                include_alpha_extra=True, threads=1)
                self.assertPixels(actual, image)
                self.assertExtras(details, image, depth, selection)

    def test_crop_references_mixed_durations_hidden_rgb_and_extra_only_changes(self):
        base = _rgba()
        images = [base, base.copy(), base.copy(), base.copy()]
        images[1][0, 1, 0] ^= 1  # Only hidden RGB changes, by one low bit.
        images[2] = images[1].copy()  # Only extra planes change next.
        images[3][2, 3, 1] ^= 513  # Close to first, distant from previous.
        images.extend([images[3].copy(), images[3].copy()])
        depth, selection = _planes(base)
        depths = [depth.copy() for _ in images]
        selections = [selection.copy() for _ in images]
        depths[2][3, 4] ^= 1
        selections[2][1, 3] ^= 1
        durations = [0, 3, 0, 5, 7, 0]
        for effort in (3, 9):
            for reference in ("previous", "first", "auto"):
                with self.subTest(effort=effort, reference=reference):
                    encoded = rs_jxl.encode_multiframe(
                        images, durations=durations, reference=reference, strategy="crop",
                        extra_channels=_extras(depths, selections), effort=effort, threads=1)
                    self.assertSequence(encoded, images, depths, selections, durations)

    def test_all_zero_durations_keep_final_repeat_visible(self):
        images = [_rgba(), _rgba()]
        images[1][0, 1, 2] ^= 1
        images.append(images[1].copy())
        depth, selection = _planes(images[0])
        depths = [depth.copy() for _ in images]
        selections = [selection.copy() for _ in images]
        for effort in (3, 9):
            with self.subTest(effort=effort):
                encoded = rs_jxl.encode_multiframe(
                    images, durations=0, reference="previous", strategy="crop",
                    extra_channels=_extras(depths, selections), effort=effort, threads=1)
                self.assertSequence(encoded, images, depths, selections, [0, 0, 0])

    def test_heterogeneous_replace_crops_saved_reference_and_zero_duration_headers(self):
        images = [_rgba(), _rgba(2, 3), _rgba(1, 2)]
        images[1][..., :3] ^= 1025
        images[2][..., :3] ^= 4097
        planes = [_planes(image) for image in images]
        depths = [pair[0] for pair in planes]
        selections = [pair[1] for pair in planes]
        origins = [(0, 0), (1, 1), (3, 0)]
        sources = [1, 2, 0]
        names = ["base", "hidden-update", "final"]
        durations = [2, 0, 0]
        options = []
        for i in range(3):
            blend = {"mode": "replace", "source": sources[i]}
            options.append({"origin": origins[i], "duration": durations[i], "name": names[i],
                            "blend": blend, "extra_blend": [dict(blend) for _ in range(3)],
                            "save_as_reference": 2 if i == 0 else 0})
        expected = images[0].copy()
        expected[1:3, 1:4] = images[1]
        expected[:1, 3:5] = images[2]
        expected_depth = depths[0].copy()
        expected_depth[1:3, 1:4] = depths[1]
        expected_depth[:1, 3:5] = depths[2]
        expected_selection = selections[0].copy()
        expected_selection[1:3, 1:4] = selections[1]
        expected_selection[:1, 3:5] = selections[2]
        for effort in (3, 9):
            with self.subTest(effort=effort):
                encoded = rs_jxl.encode_layers(
                    images, canvas=(5, 4), layer_options=options, tps=(24, 1),
                    extra_channels=_extras(depths, selections), effort=effort, threads=1)
                decode_options = {"return_extra_channels": True, "include_alpha_extra": True,
                                  "threads": 1}
                displayed, meta = rs_jxl.decode(encoded, frame=None, **decode_options)
                self.assertEqual(len(displayed), 2)
                self.assertPixels(displayed[0], images[0])
                self.assertPixels(displayed[1], expected)
                self.assertExtras(meta["frames"][0], images[0], depths[0], selections[0])
                self.assertExtras(meta["frames"][1], expected, expected_depth, expected_selection)
                selected, details = rs_jxl.decode(encoded, frame=1, **decode_options)
                self.assertPixels(selected, expected)
                self.assertFrameMetadata(details, meta["frames"][1])
                raw, raw_meta = rs_jxl.decode(encoded, frame=None, coalesced=False,
                                              **decode_options)
                self.assertEqual(len(raw), 3)
                self.assertEqual(len(raw_meta["frames"]), 3)
                for i, image in enumerate(images):
                    self.assertPixels(raw[i], image)
                    details = raw_meta["frames"][i]
                    self.assertExtras(details, image, depths[i], selections[i])
                    header = details["header"]
                    self.assertEqual(header["name"], names[i])
                    self.assertEqual(header["duration"], durations[i])
                    self.assertEqual(header["is_last"], i == 2)
                    layout = header["layer_info"]
                    self.assertEqual(layout["have_crop"], i != 0)
                    self.assertEqual(layout["crop_origin"], list(origins[i]))
                    self.assertEqual((layout["width"], layout["height"]),
                                     (image.shape[1], image.shape[0]))
                    self.assertEqual(layout["save_as_reference"], [2, 0, None][i])
                    # Full-canvas REPLACE omits source on the wire (default 0).
                    blend = {"mode": "Replace", "source": 0 if i == 0 else sources[i],
                             "alpha_channel": 0, "clamp": False}
                    self.assertEqual(layout["blend_info"], blend)
                    self.assertEqual(layout["extra_channel_blend_info"], [blend] * 3)
                    selected, selected_meta = rs_jxl.decode_layer(encoded, layer=i,
                                                                  **decode_options)
                    self.assertPixels(selected, raw[i])
                    self.assertFrameMetadata(selected_meta, details)

    def test_reusable_decoder_path_bytes_selected_and_raw_adapter_parity(self):
        images = [_rgba(2, 3), _rgba(2, 3)]
        images[1][0, 1, 0] ^= 1
        depth, selection = _planes(images[0])
        encoded = rs_jxl.encode_multiframe(
            images, durations=[0, 1], reference="previous", strategy="crop",
            extra_channels=_extras([depth, depth], [selection, selection]), effort=3, threads=1)
        options = {"return_extra_channels": True, "include_alpha_extra": True}
        baseline, metadata = rs_jxl.decode(encoded, frame=None, threads=1, **options)
        raw, raw_meta = rs_jxl.decode(encoded, frame=None, coalesced=False, threads=1, **options)
        with tempfile.TemporaryDirectory() as directory, rs_jxl.Decoder(threads=1) as decoder:
            path = Path(directory) / "combination.jxl"
            path.write_bytes(encoded)
            for source in (path, encoded):
                with self.subTest(source=type(source).__name__):
                    actual, details = decoder.decode(source, frame=0, **options)
                    self.assertPixels(actual, baseline[0])
                    self.assertFrameMetadata(details, metadata["frames"][0])
                    actual, details = decoder.decode_layer(source, layer=1, **options)
                    self.assertPixels(actual, raw[1])
                    self.assertFrameMetadata(details, raw_meta["frames"][1])


if __name__ == "__main__":
    unittest.main()
