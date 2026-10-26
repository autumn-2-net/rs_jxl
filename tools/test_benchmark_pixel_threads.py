import unittest

from benchmark_pixel_threads import decode_seconds


class DecoderTimerTests(unittest.TestCase):
    def test_exact_counter_and_timer(self):
        self.assertEqual(decode_seconds("Read 123 bytes.\n10 repetition(s) in 0.125000 s.\n", 10), 0.125)

    def test_missing_wrong_duplicate_or_zero_is_rejected(self):
        for text in ["", "9 repetition(s) in 0.125000 s.\n",
                     "10 repetition(s) in 0.125000 s.\n" * 2,
                     "10 repetition(s) in 0.000000 s.\n",
                     "10 repetition(s) in nan s.\n"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_seconds(text, 10)


if __name__ == "__main__":
    unittest.main()
