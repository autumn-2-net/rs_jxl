import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from benchmark_refinement import command, fixtures, validate_receipt
from compare_cpp_port import png_samples


class RefinementHarnessTests(unittest.TestCase):
    def test_generated_fixtures_preserve_full_depth_and_hidden_colors(self):
        with TemporaryDirectory() as directory:
            paths = fixtures(Path(directory))
            self.assertEqual(len(paths), 8)
            for path in paths:
                pixels = png_samples(path.read_bytes())
                self.assertEqual((pixels.width, pixels.height), (67, 51))
                self.assertEqual(pixels.depth, 16 if "16bit" in path.name else 8)

    def test_switches_are_independent_and_baseline_unchanged(self):
        args = SimpleNamespace(encoder=Path("new"), frozen=Path("old"),
                               effort=9, threads=8, buffering=None)
        for variant in ("baseline", "frozen", "entropy", "leaves", "combined"):
            cmd = command(args, "input.png", "out.jxl", variant)
            self.assertEqual("--refine-entropy" in cmd, variant in ("entropy", "combined"))
            self.assertEqual("--refine-leaves" in cmd, variant in ("leaves", "combined"))
            self.assertEqual(cmd[0], "old" if variant == "frozen" else "new")

    def test_receipts_distinguish_requested_and_winning_work(self):
        receipt = dict(entropy=True, leaves=True, domains=2, models=6,
                       refit_proposals=1, refit_wins=0, entropy_wins=1,
                       saved_bits=8, refit_seconds=0.1, model_seconds=0.1, write_seconds=0.1)
        validate_receipt({"refinement": receipt}, "combined")
        validate_receipt({"refinement": None}, "baseline")
        for field, invalid in (("models", 0), ("refit_wins", 2), ("refit_proposals", 3),
                               ("saved_bits", -1), ("write_seconds", float("nan"))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_receipt({"refinement": dict(receipt, **{field: invalid})}, "combined")
        with self.assertRaises(ValueError):
            validate_receipt({"refinement": receipt}, "baseline")
        with self.assertRaises(ValueError):
            validate_receipt({}, "leaves")


if __name__ == "__main__":
    unittest.main()
