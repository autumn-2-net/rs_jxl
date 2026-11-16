import unittest

from benchmark_jpeg_transcode import summary


class SummaryTests(unittest.TestCase):
    def test_distinguishes_equal_image_mean_and_pixel_weighted_bpp(self):
        cases = [
            dict(pixels=100, original_bytes=500, summaries=dict(candidate=dict(bytes=100, seconds=1, bpp=8))),
            dict(pixels=900, original_bytes=500, summaries=dict(candidate=dict(bytes=450, seconds=3, bpp=4))),
        ]
        result = summary(cases, 'candidate')
        self.assertEqual(result['bytes'], 550)
        self.assertEqual(result['seconds'], 4)
        self.assertEqual(result['bpp'], 4.4)
        self.assertEqual(result['mean_bpp'], 6)
        self.assertEqual(result['jpeg_ratio'], .55)
        self.assertEqual(result['mpix_per_second'], .00025)


if __name__ == '__main__':
    unittest.main()
