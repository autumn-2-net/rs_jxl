"""Contracts for the two-binary LZ77 benchmark, without native subprocesses."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_lz77_hotspots as bench
from test_benchmark_palette_runs import measurement
from test_compare_cpp_port import fixture


class HotspotBenchmark(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.png = fixture()
        self.source = self.root / "input.png"
        self.source.write_bytes(self.png)
        self.before, self.after, self.decoder = [self.root / name for name in ("before", "after", "djxl")]
        for path in (self.before, self.after, self.decoder):
            path.write_bytes(path.name.encode())
        self.argv = ["--before", str(self.before), "--after", str(self.after),
                     "--decoder", str(self.decoder), "--case", str(self.source),
                     "--out", str(self.root / "report.json"), "--repeats", "2"]
        self.args = bench.arguments(self.argv)

    def test_command_and_validation(self):
        command = bench.command(self.before, self.source, "out.jxl", self.args)
        self.assertEqual(command[4:], ["--mode", "default", "--effort", "9", "--threads", "8"])
        for extra in (["--repeats", "0"], ["--timeout", "nan"], ["--threads", "0"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv + extra)

    def fake_process(self, command, env, timeout):
        result = dict(returncode=0, stdout="", stderr="", seconds=0.2)
        if command[0] == str(self.decoder):
            Path(command[2]).write_bytes(self.png)
        else:
            Path(command[command.index("--output") + 1]).write_bytes(b"x" * 13)
            expected = bench.port.png_samples(self.png)
            result["stdout"] = json.dumps(measurement(self.args, width=expected.width, height=expected.height))
        return result

    def test_rotating_binaries_pixel_checks_and_cleanup(self):
        with patch.object(bench.port, "run_process", side_effect=self.fake_process) as process:
            with redirect_stdout(StringIO()):
                result = bench.benchmark(self.args)
        self.assertEqual(result["status"], "passed", result["errors"])
        self.assertTrue(result["temporary_outputs_cleaned"])
        encoded = [call.args[0][0] for call in process.call_args_list if "--output" in call.args[0]]
        self.assertEqual(encoded.count(str(self.before)), 2)
        self.assertEqual(encoded.count(str(self.after)), 2)
        self.assertEqual(encoded[:2], list(reversed(encoded[2:])))
        self.assertEqual(process.call_count, 8)
        self.assertEqual(result["aggregate"]["candidate"]["ratios_to_baseline"]["bytes"], 1)
        with self.assertRaises(FileExistsError):
            bench.benchmark(self.args)

    def test_changed_pixels_fail_even_when_size_matches(self):
        with patch.object(bench.port, "run_process", side_effect=self.fake_process):
            with patch.object(bench.port, "compare_pixels", side_effect=ValueError("pixel mismatch")):
                result = bench.benchmark(self.args)
        self.assertEqual(result["status"], "failed")
        self.assertIn("pixel mismatch", result["errors"][0])
        self.assertTrue(result["temporary_outputs_cleaned"])

    def test_different_projected_policy_fails(self):
        def changed_policy(command, env, timeout):
            result = self.fake_process(command, env, timeout)
            if command[0] == str(self.after):
                m = json.loads(result["stdout"])
                m["projected"]["groupwise"] = False
                m["configured"]["groupwise"] = False
                result["stdout"] = json.dumps(m)
            return result
        with patch.object(bench.port, "run_process", side_effect=changed_policy), redirect_stdout(StringIO()):
            result = bench.benchmark(self.args)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Native policy changed", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
