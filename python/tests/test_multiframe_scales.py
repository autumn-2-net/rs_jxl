"""Multiscale native partitions, history observations and decoder interoperability."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rs_jxl
import test_bindings as fixtures


def sequence(dtype=np.uint8):
    rng = np.random.default_rng(2871)
    first = rng.integers(0, 256, (40, 48, 4), dtype=dtype)
    first[..., 3] = 0
    if dtype == np.uint16:
        first[..., :3] *= 257
    result = [first]
    for value in (7, 19, 27):
        current = first.copy()
        current[1:3, 1:3, 0] ^= value
        current[25:27, 25:27, 1] ^= value
        result.append(current)
    return result


class NativeMultiscale(unittest.TestCase):
    def test_opaque_and_hidden_samples_extra_plane_and_context(self):
        for dtype in (np.uint8, np.uint16):
            frames = sequence(dtype)
            depths = [np.full(frame.shape[:2], 19, dtype=np.uint16) for frame in frames]
            depths[2][32, 32] = 29
            options = {"extra_channels": [{"data": depths, "type": "depth"}],
                       "strategy": "auto", "effort": 3, "return_info": True,
                       "auto_options": {"extra_tile_sizes": [8, 16, 64],
                                        "retained_reference": True, "reference_lookahead": True}}
            with rs_jxl.Encoder(threads=2) as encoder:
                data, report = encoder.encode_multiframe(frames, **options)
                decoded, metadata = rs_jxl.decode(data, frame=None, return_extra_channels=True)
                self.assertEqual(len(decoded), len(frames))
                for i, actual in enumerate(decoded):
                    np.testing.assert_array_equal(actual, frames[i])
                    np.testing.assert_array_equal(metadata['frames'][i]['extra_channels'][0]['data'][..., 0], depths[i])
                self.assertTrue(all('tile_size' in c for f in report['frames'] for c in f['candidates'] + f['pruned']))
                analysis = encoder.analyze_multiframe(frames, analysis_options={
                    'extra_tile_sizes': [8, 16, 64], 'history_window': 4})
                self.assertGreater(analysis['geometry']['multi_rect_frames'], 0)
                self.assertEqual(analysis['geometry']['candidate_count'], sum(
                    f['geometry']['candidate_count'] for f in analysis['frames']))
                self.assertTrue(all('source' not in h for f in analysis['frames'] for h in f['history']))

    def test_analysis_detects_recurrence_without_claiming_it_is_a_saved_reference(self):
        a = np.zeros((8, 8, 3), np.uint8)
        b = np.full_like(a, 72)
        c = np.full_like(a, 179)
        returned = b.copy()
        returned[3, 3, 0] += 1
        report = rs_jxl.analyze_multiframe([a, b, c, returned], analysis_options={'history_window': 3})
        self.assertEqual(report['history_window'], 3)
        history = report['frames'][3]['history']
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['reference_index'], 1)
        self.assertEqual(history[0]['stats']['changed_pixels'], 1)
        self.assertTrue(history[0]['improves_static_bbox'])
        self.assertNotIn('source', history[0])
        disabled = rs_jxl.analyze_multiframe([a, b, c, returned])
        self.assertTrue(all(not f['history'] for f in disabled['frames']))


@unittest.skipUnless('RS_JXL_DJXL' in os.environ, 'official djxl is opt-in')
class OfficialMultiscale(fixtures.PixelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OfficialDecoderTests.setUpClass.__func__(cls)

    def test_hidden_rgba_and_zero_duration_intermediates(self):
        for dtype in (np.uint8, np.uint16):
            frames = sequence(dtype)
            data = rs_jxl.encode_multiframe(
                frames, effort=3, strategy='auto', durations=[0, 2, 0, 3],
                auto_options={'extra_tile_sizes': [8, 16, 64], 'retained_reference': True,
                              'reference_lookahead': True})
            expected = np.stack([frames[1], frames[3]])
            for decoder in self.decoders:
                with self.subTest(dtype=dtype, decoder=decoder), tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / 'frames.npy'
                    fixtures.OfficialDecoderTests.runDecoder(self, decoder, data, destination, directory)
                    actual = np.load(destination, allow_pickle=False)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertTrue(np.isfinite(actual).all())
                    recovered = np.rint(actual.astype(np.float64) * np.iinfo(dtype).max).astype(dtype)
                    self.assertPixels(recovered, expected)


if __name__ == '__main__':
    unittest.main()
