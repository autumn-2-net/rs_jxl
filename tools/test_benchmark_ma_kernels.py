import unittest

from benchmark_ma_kernels import parse_stages, summarize


class KernelBenchmarkTests(unittest.TestCase):
    def test_stage_table_does_not_parse_parent_edges_as_timing(self):
        report = ("stage | calls | total seconds | longest seconds\n"
                  "ma.learn | 3 | 1.2 | 0.7\n"
                  "[rs-jxl profile] parent edges (child <- parent | calls)\n"
                  "ma.learn <- encode | 3\n")
        self.assertEqual(parse_stages(report), {
            "ma.learn": {"calls": 3, "seconds": 1.2, "longest": 0.7}})
        with self.assertRaises(ValueError):
            parse_stages("no diagnostic table")

    def test_repeats_count_each_image_once_and_keep_size_changes(self):
        rows = [dict(source="a", variant="new", threads=4,
                     width=4, height=5, bytes=b, seconds=s)
                for b, s in [(10, 1), (12, 8), (14, 3)]]
        row, = summarize(rows)
        self.assertEqual(row["images"], 1)
        self.assertEqual(row["bytes"], 12)
        self.assertEqual(row["seconds"], 3)
        self.assertEqual(row["mean_bpp"], 4.8)


if __name__ == "__main__":
    unittest.main()
