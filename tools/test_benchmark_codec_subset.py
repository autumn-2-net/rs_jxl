import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_codec_subset import NAMES, aggregates, decode_command, encode_command


class BenchmarkContractTests(unittest.TestCase):
    def test_float_null_output_matches_djxl_contract(self):
        for name in NAMES:
            cmd = decode_command(name, Path('codec.exe'), Path('input.jxl'),
                                 kind='png', threads=8, reps=10)
            self.assertIn('--disable_output', cmd)
            self.assertIn('--num_reps=10', cmd)
            self.assertEqual('--bits=32' in cmd, name == 'rust')
            self.assertNotIn('--reconstruct_jpeg', cmd)

    def test_archival_decode_never_falls_back_to_jpeg_encoding(self):
        for name in NAMES:
            cmd = decode_command(name, Path('codec.exe'), Path('input.jxl'),
                                 kind='jpeg', threads=8)
            self.assertIn('--reconstruct_jpeg', cmd)
            self.assertNotIn('--pixels_to_jpeg', cmd)

    def test_encode_routes_are_separate(self):
        for name in NAMES:
            cmd = encode_command(name, Path('codec.exe'), Path('input.jpg'),
                                 Path('out.jxl'), {'kind': 'jpeg'}, SimpleNamespace(threads=8))
            self.assertNotIn('--lossless_jpeg=0', cmd)
            if name == 'rust':
                self.assertIn('transcode-jpeg', cmd)
                self.assertIn('--source-search', cmd)
            else:
                self.assertIn('--lossless_jpeg=1', cmd)

    def test_aggregate_uses_total_pixels_not_mean_of_rates(self):
        cases = []
        for pixels, size, seconds in ((1_000_000, 100, 1), (3_000_000, 300, 9)):
            cases.append(dict(kind='png', pixels=pixels, input_bytes=1000,
                encode={n: dict(bytes=size, seconds=seconds) for n in NAMES},
                decode={'official': {n: dict(seconds=seconds) for n in NAMES}}))
        summary = aggregates(cases)['png']
        self.assertEqual(summary['encoders']['rust']['mpix_s'], .4)
        self.assertEqual(summary['encoders']['rust']['bpp'], .0008)
        self.assertEqual(summary['decoders']['rust']['mpix_s'], .4)


if __name__ == '__main__':
    unittest.main()
