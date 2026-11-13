import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_patch_representatives as bench


class RepresentativeHarnessTests(unittest.TestCase):
    def test_synthetic_originals_retain_16bit_and_invisible_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            cases = bench.fixtures(Path(folder))
            self.assertEqual(len(cases), 8)
            for case in cases:
                pixels = bench.patch.port.png_samples(case.read_bytes())
                self.assertEqual((pixels.width, pixels.height), (129, 83))
                self.assertEqual(pixels.depth, int(case.stem.split("-")[1]))

    def args(self):
        return SimpleNamespace(encoder=Path("patch_lab.exe"), effort=9, threads=8,
                               parallel_gather=False, candidates=8, occurrences=16, pixels=128)

    def receipt(self, mode="medoid"):
        return {"mode": mode, "candidates_budget": 8, "occurrences_budget": 16, "pixels_budget": 128,
                "retained_groups": 2, "occurrences": 9, "searched_groups": 2,
                "variable_sample_groups": 1, "changed_groups": 1, "candidates": 9,
                "sampled_occurrences": 9, "sampled_pixels": 256, "selection_seconds": 0.01,
                "baseline_proxy": 100.0, "selected_proxy": 90.0, "emitted_plans": 1,
                "atlas_pixels": 50, "dictionary_positions": 9, "float_fallback_groups": 0}

    def validate(self, receipt, variant="medoid"):
        with patch.object(bench.patch, "validate_measurement"):
            bench.validate_measurement(self.args(), {"representative_work": receipt}, variant, 100, 20, 20)

    def test_commands_do_not_change_e9_or_detection_parameters(self):
        args = self.args()
        for mode in ("first", "medoid", "entropy"):
            command = bench.command(args, Path("input.png"), Path("out.jxl"), mode)
            self.assertEqual(command[command.index("--effort") + 1], "9")
            self.assertEqual(command[command.index("--mode") + 1], "source")
            self.assertEqual(command[command.index("--source-representative") + 1], mode)
            self.assertNotIn("--block-size", command)
            self.assertNotIn("--previous-channels", command)
        self.assertNotIn("--source-representative", bench.command(args, Path("i"), Path("o"), "off"))

    def test_valid_and_no_detection_receipts(self):
        self.validate(self.receipt())
        work = self.receipt("first")
        for key in work:
            if key not in ("mode", "candidates_budget", "occurrences_budget", "pixels_budget"):
                work[key] = 0
        self.validate(work, "first")
        self.validate(None, "off")

    def test_rejects_missing_misleading_or_nonfinite_receipts(self):
        for key, value in (("changed_groups", 3), ("emitted_plans", 2), ("selected_proxy", 101),
                           ("selection_seconds", float("nan")), ("candidates_budget", 4),
                           ("atlas_pixels", 0), ("dictionary_positions", 0), ("mode", "first")):
            with self.subTest(key=key):
                work = self.receipt()
                work[key] = value
                with self.assertRaises(ValueError):
                    self.validate(work)
        with self.assertRaises(ValueError):
            self.validate(None)
        with self.assertRaises(ValueError):
            self.validate(self.receipt("first"), "first")


if __name__ == "__main__":
    unittest.main()
