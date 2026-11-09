import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import benchmark_lz77_extension as bench


class ExtensionBenchmarkTests(unittest.TestCase):
    def arguments(self, root, extra=()):
        return bench.arguments(["--encoder", "new.exe", "--decoder", "djxl.exe",
                                "--case", "input.png", "--out", str(root / "report.json"), *extra])

    def test_commands_do_not_modify_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory), ["--frozen", "old.exe"])
            for variant in bench.VARIANTS:
                command = bench.command(args, "input.png", "out.jxl", variant)
                self.assertEqual(command[0], "old.exe" if variant == "frozen" else "new.exe")
                self.assertEqual("--match-extension" in command, variant not in ("baseline", "frozen"))
                self.assertEqual("--batch-matches" in command, variant.endswith("-batch"))
                self.assertNotIn("--search", command)

    def test_invalid_matrix_and_report_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in (["--variants", "baseline", "frozen"], ["--variants", "avx2"],
                          ["--variants", "baseline", "baseline"], ["--timeout", "nan"],
                          ["--threads", "0"], ["--repeats", "0"], ["--warmups", "-1"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.arguments(root, extra)
            (root / "report.json").touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.arguments(root)

    def test_receipt_distinguishes_requested_and_active_batching(self):
        receipt = dict(mode="avx2", batch=True, active=True, batch_active=False,
                       extensions=9, batch_groups=0, fast_matchers=0, chain_matchers=2)
        bench.validate_receipt(dict(match_extension=receipt), "avx2-batch")
        for key, value in (("mode", "avx512"), ("extensions", -1), ("extensions", True),
                           ("batch_groups", 3), ("batch", False), ("batch_active", True),
                           ("active", False), ("chain_matchers", 1.5)):
            invalid = copy.copy(receipt)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_receipt(dict(match_extension=invalid), "avx2-batch")
        bench.validate_receipt(dict(match_extension=None), "baseline")
        bench.validate_receipt({}, "frozen")
        with self.assertRaises(ValueError):
            bench.validate_receipt({}, "baseline")
        with self.assertRaises(ValueError):
            bench.validate_receipt(dict(match_extension=receipt), "baseline")


if __name__ == "__main__":
    unittest.main()
