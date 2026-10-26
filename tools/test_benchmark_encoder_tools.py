"""Harness regression tests; codec subprocesses are mocked, no benchmarks run."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_encoder_tools as bench
from test_compare_cpp_port import fixture


class EnvironmentTests(unittest.TestCase):
    def test_explicit_override_reaches_only_encoder_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.png"
            source.write_bytes(fixture())
            encoder, decoder = root / "encoder.exe", root / "decoder.exe"
            encoder.write_bytes(b"encoder")
            decoder.write_bytes(b"decoder")
            calls = []

            def run(command, env, timeout):
                calls.append((command, dict(env)))
                if command[0] == str(encoder):
                    Path(command[command.index("--output") + 1]).write_bytes(b"JXL")
                    stdout = json.dumps(dict(bytes=3, bpp=12, seconds=0.1))
                else:
                    Path(command[2]).write_bytes(fixture())
                    stdout = ""
                return dict(command=command, returncode=0, stdout=stdout, stderr="", wall_seconds=0.2)

            argv = ["benchmark", "--encoder", str(encoder), "--djxl", str(decoder),
                    "--case", str(source), "--variant", "default=[]", "--out", str(root / "run"),
                    "--encoder-env", "RS_JXL_MA_RCP=1"]
            with patch.dict(os.environ, {"RS_JXL_PROFILE": "trace", "RS_JXL_MA_AVX512": "1"}), \
                    patch("sys.argv", argv), patch.object(bench.port, "run_process", side_effect=run), \
                    redirect_stdout(StringIO()):
                self.assertEqual(bench.main(), 0)
            self.assertEqual(len(calls), 2)
            for command, env in calls:
                self.assertNotIn("RS_JXL_PROFILE", env)
                self.assertNotIn("RS_JXL_MA_AVX512", env)
                self.assertEqual(env.get("RS_JXL_MA_RCP"), "1" if command[0] == str(encoder) else None)
            report = json.loads((root / "run" / "report.json").read_text())
            self.assertEqual(report["environment"]["explicit"], {"RS_JXL_MA_RCP": "1"})
            self.assertEqual(report["decoder_environment"]["explicit"], {})
            self.assertEqual({p.name for p in (root / "run").iterdir()}, {"report.json"})


class RepeatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input.png"
        self.source.write_bytes(fixture())
        self.encoder, self.decoder = self.root / "encoder.exe", self.root / "decoder.exe"
        self.encoder.write_bytes(b"encoder")
        self.decoder.write_bytes(b"decoder")
        self.out = self.root / "run"

    def argv(self, variants=("a", "b"), cases=None):
        result = ["benchmark", "--encoder", str(self.encoder), "--djxl", str(self.decoder),
                  "--out", str(self.out)]
        for source in cases if cases is not None else [self.source]:
            result += ["--case", str(source)]
        for name in variants:
            result += ["--variant", name + "=" + json.dumps(["--marker", name])]
        return result

    def run_harness(self, *, variants=("a", "b"), repeats=None, warmups=None, effort=None,
                    samples=None, cases=None, bad_decode=None, matrix=False):
        argv = self.argv(() if matrix else variants, cases)
        if matrix:
            path = self.root / "matrix.json"
            path.write_text(json.dumps({name: ["--marker", name] for name in variants}), encoding="utf-8")
            argv += ["--matrix", str(path)]
        if repeats is not None:
            argv += ["--repeats", str(repeats)]
        if warmups is not None:
            argv += ["--warmups", str(warmups)]
        if effort is not None:
            argv += ["--effort", str(effort)]
        calls, visits = [], {}
        decode_count = 0

        def run(command, env, timeout):
            nonlocal decode_count
            calls.append(command)
            if command[0] == str(self.encoder):
                name = command[command.index("--marker") + 1]
                attempt = visits.get(name, 0)
                visits[name] = attempt + 1
                size, seconds, wall = samples[name][attempt] if samples else (3, 0.1, 0.2)
                Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                stdout = json.dumps(dict(bytes=size, bpp=size * 4, seconds=seconds))
            else:
                decode_count += 1
                Path(command[2]).write_bytes(fixture(8 if decode_count == bad_decode else 16))
                stdout, wall = "", 0.5
            return dict(command=command, returncode=0, stdout=stdout, stderr="", wall_seconds=wall)

        output, errors = StringIO(), StringIO()
        with patch("sys.argv", argv), patch.object(bench.port, "run_process", side_effect=run), \
                redirect_stdout(output), redirect_stderr(errors):
            result = bench.main()
        report = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual({p.name for p in self.out.iterdir()}, {"report.json"})
        self.assertEqual(len(calls), sum(visits.values()) + decode_count)
        return result, report, output.getvalue(), calls

    def test_defaults_preserve_single_round_and_make_call_rate_primary(self):
        result, report, output, calls = self.run_harness()
        self.assertEqual(result, 0)
        self.assertEqual((report["repeats"], report["warmups"]), (1, 0))
        self.assertEqual(report["effort"], 9)
        for command in calls[::2]:
            self.assertEqual(command.count("--effort"), 1)
            self.assertEqual(command[command.index("--effort") + 1], "9")
        self.assertEqual(report["primary_metric"], "call_mpix_per_second")
        self.assertEqual(len(calls), 4)
        self.assertEqual([row["variant"] for row in report["rows"]], ["a", "b"])
        for row in report["rows"]:
            self.assertEqual((row["repeat"], row["warmup"], row["status"]), (0, False, "passed"))
            self.assertAlmostEqual(row["call_mpix_per_second"], 2e-6 / 0.1)
            self.assertAlmostEqual(row["whole_process_mpix_per_second"], 2e-6 / 0.2)
        verified = [line for line in output.splitlines() if line.endswith(" verified")]
        self.assertEqual(len(verified), 2)
        for line in verified:
            self.assertIn("repeat=1/1", line)
            self.assertLess(line.index("call="), line.index("wall="))
            self.assertEqual(line.count("MPix/s"), 2)
            self.assertIn("100.000 ms", line)
            self.assertIn("200.000 ms", line)
        self.assertEqual([item["trials"] for item in report["summaries"]], [1, 1])

    def test_explicit_effort_reaches_every_encoder_call_and_report(self):
        case = dict(index=0, label=self.source.name, kind="png", pixels=2)
        expected = SimpleNamespace(depth=16)
        for effort in (None, 1, 10):
            with self.subTest(effort=effort), \
                    patch.object(bench, "load_case", return_value=(case, fixture(), None, expected)), \
                    patch.object(bench.port, "png_samples", return_value=expected), \
                    patch.object(bench.port, "compare_pixels"):
                self.out = self.root / f"run-{effort}"
                result, report, _, calls = self.run_harness(
                    effort=effort, repeats=2, warmups=1, matrix=True)
                self.assertEqual(result, 0)
                selected = 9 if effort is None else effort
                self.assertEqual(report["effort"], selected)
                self.assertEqual(len(calls), 12)
                for row, command in zip(report["rows"], calls[::2]):
                    self.assertEqual(command.count("--effort"), 1)
                    self.assertEqual(command[command.index("--effort") + 1], str(selected))
                    self.assertEqual(row["encode"]["command"], command)
                for command in calls[1::2]:
                    self.assertNotIn("--effort", command)

    def test_invalid_effort_fails_before_io(self):
        for effort in ("0", "-1", "11", "12", "10.5", "bad"):
            with self.subTest(effort=effort), \
                    patch("sys.argv", self.argv() + ["--effort", effort]), \
                    patch.object(bench, "load_case") as load, \
                    patch.object(bench.port, "run_process") as run, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.out.exists())
                load.assert_not_called()
                run.assert_not_called()

    def test_variants_cannot_override_harness_effort(self):
        for tokens in (["--effort", "9"], ["--effort=10"]):
            for matrix in (False, True):
                argv = self.argv(variants=()) + ["--effort", "10"]
                if matrix:
                    path = self.root / "matrix.json"
                    path.write_text(json.dumps({"a": tokens}), encoding="utf-8")
                    argv += ["--matrix", str(path)]
                else:
                    argv += ["--variant", "a=" + json.dumps(tokens)]
                with self.subTest(tokens=tokens, matrix=matrix), patch("sys.argv", argv), \
                        patch.object(bench.port, "run_process") as run, redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        bench.main()
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(self.out.exists())
                    run.assert_not_called()

    def test_matrix_variants_use_the_same_repeat_order(self):
        result, report, _, calls = self.run_harness(matrix=True, repeats=2)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 8)
        self.assertEqual([row["variant"] for row in report["rows"]], ["a", "b", "b", "a"])
        self.assertEqual([item["trials"] for item in report["summaries"]], [2, 2])

    def test_every_round_reverses_order_and_every_warmup_is_verified(self):
        result, report, output, calls = self.run_harness(
            variants=("a", "b", "c"), repeats=3, warmups=2)
        self.assertEqual(result, 0)
        rows = report["rows"]
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(calls), 30)
        for round_index, repeat in enumerate(range(-2, 3)):
            selected = rows[round_index * 3:(round_index + 1) * 3]
            expected = ["a", "b", "c"] if repeat % 2 == 0 else ["c", "b", "a"]
            self.assertEqual([row["variant"] for row in selected], expected)
            self.assertEqual([row["order_index"] for row in selected], [0, 1, 2])
            self.assertTrue(all(row["repeat"] == repeat and row["warmup"] == (repeat < 0)
                                and row["status"] == "passed" for row in selected))
            commands = calls[round_index * 6:(round_index + 1) * 6]
            self.assertEqual([command[command.index("--marker") + 1] for command in commands[::2]], expected)
            self.assertTrue(all(command[0] == str(self.decoder) for command in commands[1::2]))
        verified = [line for line in output.splitlines() if line.endswith(" verified")]
        self.assertEqual(len(verified), 15)
        self.assertEqual(sum("warmup=" in line for line in verified), 6)
        self.assertEqual(sum("repeat=" in line for line in verified), 9)
        self.assertEqual([item["trials"] for item in report["summaries"]], [3, 3, 3])

    def test_medians_exclude_warmups_allow_varying_sizes_and_invert_median_time(self):
        samples = {
            "a": [(99, 9.0, 10.0), (5, 0.003, 0.009), (9, 0.001, 0.003)],
            "b": [(77, 7.0, 8.0), (6, 0.002, 0.005), (8, 0.004, 0.007)],
        }
        result, report, output, _ = self.run_harness(repeats=2, warmups=1, samples=samples)
        self.assertEqual(result, 0)
        self.assertEqual([row["variant"] for row in report["rows"]], ["b", "a", "a", "b", "b", "a"])
        a, b = report["summaries"]
        self.assertEqual((a["variant"], b["variant"]), ("a", "b"))
        self.assertEqual((a["trials"], a["median_bytes"], a["min_bytes"], a["max_bytes"]), (2, 7, 5, 9))
        self.assertEqual((a["median_bpp"], a["min_bpp"], a["max_bpp"]), (28, 20, 36))
        self.assertAlmostEqual(a["median_call_seconds"], 0.002)
        self.assertAlmostEqual(a["median_wall_seconds"], 0.006)
        self.assertAlmostEqual(a["call_mpix_per_second"], 2e-6 / 0.002)
        self.assertAlmostEqual(a["whole_process_mpix_per_second"], 2e-6 / 0.006)
        self.assertAlmostEqual(b["median_call_seconds"], 0.003)
        self.assertIn("median input.png a (n=2): call=0.001 MPix/s (2.000 ms)", output)
        self.assertIn("7 B range=[5, 9]", output)

    def test_same_basename_cases_remain_separate_summary_groups(self):
        second = self.root / "another" / self.source.name
        second.parent.mkdir()
        second.write_bytes(fixture())
        result, report, _, _ = self.run_harness(variants=("a",), cases=[self.source, second])
        self.assertEqual(result, 0)
        self.assertEqual([item["case_index"] for item in report["summaries"]], [0, 1])
        self.assertEqual([item["trials"] for item in report["summaries"]], [1, 1])
        self.assertEqual([row["case_index"] for row in report["rows"]], [0, 1])

    def test_failed_warmup_is_not_silently_ignored(self):
        result, report, output, calls = self.run_harness(repeats=2, warmups=1, bad_decode=1)
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summaries"], [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(report["rows"]), 1)
        self.assertTrue(report["rows"][0]["warmup"])
        self.assertEqual(report["rows"][0]["status"], "failed")
        self.assertFalse(any(line.endswith(" verified") for line in output.splitlines()))

    def test_incomplete_repeat_keeps_verified_rows_but_no_success_medians(self):
        result, report, output, _ = self.run_harness(repeats=2, bad_decode=3)
        self.assertEqual(result, 1)
        self.assertEqual(report["summaries"], [])
        self.assertEqual([row["status"] for row in report["rows"]], ["passed", "passed", "failed"])
        self.assertEqual(sum(line.endswith(" verified") for line in output.splitlines()), 2)
        self.assertFalse(any(line.startswith("median ") for line in output.splitlines()))

    def test_invalid_counts_and_overrides_fail_before_output_directory(self):
        for extra in (["--repeats", "0"], ["--repeats", "-1"], ["--repeats", "1.5"],
                      ["--warmups", "-1"], ["--warmups", "bad"], ["--threads", "0"],
                      ["--threads", "65"], ["--encoder-env", "PATH=x"],
                      ["--variant", "a=[]"], ["--case", str(self.root / "missing.png")]):
            with self.subTest(extra=extra), patch("sys.argv", self.argv() + extra), \
                    patch.object(bench.port, "run_process") as run, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.out.exists())
                run.assert_not_called()

    def test_invalid_matrices_fail_before_output_directory(self):
        matrix = self.root / "matrix.json"
        for content in ("not json", "[]", "{}", '{"a":"--foo"}', '{"":[]}',
                        '{"a":["--threads","4"]}', '{"a":["--output","other.jxl"]}'):
            matrix.write_text(content, encoding="utf-8")
            with self.subTest(content=content), \
                    patch("sys.argv", self.argv(variants=()) + ["--matrix", str(matrix)]), \
                    patch.object(bench.port, "run_process") as run, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.out.exists())
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
