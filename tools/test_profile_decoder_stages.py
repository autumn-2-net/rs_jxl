import unittest

from profile_decoder_stages import parse_profile


class ProfileParsingTests(unittest.TestCase):
    def test_coarse_nested_and_sampled_scopes_are_not_combined(self):
        parsed = parse_profile("""[rs-jxl profile] inclusive
stage | calls | total seconds | longest seconds
decode.jpeg.total | 5 | 1.2 | 0.3
[decoder stages] inclusive; sampled times are not totals
stage | calls | total seconds | mean ns
transform.rct.i16 | 5 | 0.010 | 2000000
sample.wp.predict | 100 | 0.000012 | 120
timer_pair_floor_ns | 20.0
""")
        self.assertEqual(parsed["scopes"]["decode.jpeg.total"]["longest_seconds"], 0.3)
        self.assertEqual(parsed["samples"]["sample.wp.predict"]["mean_ns"], 120)
        self.assertNotIn("sample.wp.predict", parsed["scopes"])
        self.assertEqual(parsed["timer_pair_floor_ns"], 20)

    def test_missing_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_profile("decode.jpeg.total | 5 | 1.2 | 0.3")


if __name__ == "__main__":
    unittest.main()
