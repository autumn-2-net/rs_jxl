import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_low_effort as bench


class LowEffortTests(unittest.TestCase):
    def test_official_command_forces_pixel_lossless_and_retains_hidden_rgb(self):
        command = bench.encoder_command("cjxl.exe", True, "input.png", "output.jxl", 3, 8)
        self.assertEqual(command[:7], ["cjxl.exe", "input.png", "output.jxl", "-d", "0", "-e", "3"])
        for flag in ["--num_threads=8", "--lossless_jpeg=0", "--keep_invisible=1", "--num_reps=1"]:
            self.assertIn(flag, command)
        self.assertIn("--direct-candidate", bench.encoder_command("lab.exe", False, "in.png", "out.jxl", 3, 8))

    def test_profile_parser_does_not_add_inclusive_parent_edges(self):
        rows = bench.profile_rows("stage | calls | total seconds | longest seconds\n"
                                  "modular.tokenize | 8 | 0.25 | 0.05\n"
                                  "modular.tokenize <- encode | 8\n")
        self.assertEqual(rows, {"modular.tokenize": dict(calls=8, seconds=.25, longest=.05)})

    def test_summary_uses_per_image_medians_and_separate_thread_series(self):
        rows = []
        for threads in (1, 8):
            for source, pixels, size, seconds in (("small", 100, 50, [1, 3, 2]),
                                                  ("large", 900, 900, [10, 30, 20])):
                for elapsed in seconds:
                    rows.append(dict(binary="new", effort=2, threads=threads, source=source,
                                     receipt=dict(bytes=size, seconds=elapsed / threads,
                                                  width=pixels, height=1)))
        summaries = bench.summarize(rows)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["seconds"], 22)
        self.assertEqual(summaries[1]["seconds"], 22 / 8)
        self.assertEqual(summaries[0]["mean_bpp"], 6)
        self.assertEqual(summaries[0]["weighted_bpp"], 7.6)
        self.assertEqual(summaries[0]["bytes"], 950)


if __name__ == "__main__":
    unittest.main()
