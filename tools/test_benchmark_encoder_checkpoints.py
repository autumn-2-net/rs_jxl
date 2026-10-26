"""Focused harness tests; all encoder, decoder and version processes are mocked.

PNG checks reuse the existing numpy/OpenCV oracle and PNG test fixtures. No
builds, real codec processes, corpus benchmarks or dependency installation.
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import benchmark_encoder_checkpoints as bench
from test_compare_cpp_port import fixture, make_png


class CommandTests(unittest.TestCase):
    def test_explicit_threads_apply_to_both_pixel_and_jpeg_commands(self):
        for threads in (1, 16, 256):
            for kind in ("png", "jpeg"):
                command = bench.encoder_command("rust", "in", "out", kind, 123, threads=threads)
                self.assertEqual(command[command.index("--threads") + 1], str(threads))

    def test_fixed_png_and_original_jpeg_routes(self):
        for kind in ("png", "jpeg"):
            with self.subTest(kind=kind):
                command = bench.encoder_command("rust", "original", "out.jxl", kind, 123,
                                                ["--experimental-scout=fast"])
                self.assertEqual(command[1], "encode" if kind == "png" else "transcode-jpeg")
                self.assertEqual(command[2:4], ["original", "out.jxl"])
                self.assertEqual(command[command.index("--threads") + 1], "8")
                self.assertIn("--source-search", command)
                effort = "--source-effort" if kind == "png" else "--effort"
                self.assertEqual(command[command.index(effort) + 1], "9")
                self.assertEqual(command[-1], "--experimental-scout=fast")
                self.assertEqual("--strip-metadata" in command, kind == "png")
                self.assertEqual("--source-sample-limit" in command, kind == "png")

    def test_extra_arguments_cannot_replace_contract_or_paths(self):
        bench.validate_extra_args(["--source-scout", "--pilot-size", "17", "--route=fast"])
        for name in bench.RESERVED_ARGS:
            for token in (name, name + "=1"):
                with self.subTest(token=token), self.assertRaises(ValueError):
                    bench.validate_extra_args([token])
        for tokens in ([""], ["--"], ["--route=\0"], ["output.jxl"], ["-h"],
                       ["--route=x", "output.jxl"], ["--route", "x", "output.jxl"]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                bench.validate_extra_args(tokens)

    def test_jpeg_search_diagnostic_absence_is_explicit(self):
        process = dict(stdout="", stderr="")
        diagnostics = bench.search_diagnostics("jpeg", process)
        self.assertTrue(diagnostics["requested"])
        self.assertFalse(diagnostics["diagnostics_observed"])
        self.assertIn("does not guarantee", diagnostics["evidence"])


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("before", "after", "djxl"):
            (self.root / (name + ".exe")).write_bytes(name.encode())
        self.png = self.root / "source.png"
        self.png.write_bytes(fixture())
        self.jpeg = self.root / "original.jpg"
        self.jpeg_bytes = b"\xff\xd8\xfforiginal-JPEG-metadata-and-coefficients\xff\xd9trailing"
        self.jpeg.write_bytes(self.jpeg_bytes)
        self.calls = []
        self.output_number = 0
        self.failure = None
        self.mismatch = None
        self.no_search_evidence = False
        self.times = {"before": [3], "after": [1]}
        self.counts = {"before": 0, "after": 0}

    def arguments(self, *extra, cases=None):
        self.out = self.root / f"run-{self.output_number}"
        self.output_number += 1
        argv = ["--before", str(self.root / "before.exe"),
                "--after", str(self.root / "after.exe"),
                "--official-djxl", str(self.root / "djxl.exe"),
                "--out", str(self.out), "--repeats", "2", "--warmups", "1"]
        for path in ([self.png] if cases is None else cases):
            argv += ["--case", str(path)]
        return argv + list(extra)

    def test_worker_limit_validation(self):
        for threads in (0, -1, 257):
            args = bench.parser().parse_args(self.arguments("--threads", str(threads)))
            with self.assertRaisesRegex(ValueError, "Threads"):
                bench.validate_args(args)
        args = bench.parser().parse_args(self.arguments("--threads", "16"))
        bench.validate_args(args)

    def test_explicit_encoder_environment_is_applied_and_decoder_is_clean(self):
        with patch.dict(os.environ, {"RS_JXL_MA_AVX512": "1", "RS_JXL_PROFILE": "trace"}):
            report = self.run_benchmark("--after-env", "RS_JXL_MA_RCP=1")
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["environments"]["after"]["explicit"], {"RS_JXL_MA_RCP": "1"})
        self.assertEqual(report["environments"]["before"]["explicit"], {})
        self.assertEqual(report["environments"]["decoder"]["explicit"], {})
        for command, env, _ in self.calls:
            self.assertNotIn("RS_JXL_MA_AVX512", env)
            self.assertNotIn("RS_JXL_PROFILE", env)
            self.assertEqual(env.get("RS_JXL_MA_RCP"), "1" if Path(command[0]).stem == "after" else None)

    def test_environment_override_rejects_unrelated_or_duplicate_variables(self):
        for values in [["PATH=anything"], ["JXL_OTHER=1"], ["RS_JXL_X=1", "RS_JXL_X=2"], ["RS_JXL_X=\0"]]:
            extra = [token for value in values for token in ("--after-env", value)]
            args = bench.parser().parse_args(self.arguments(*extra))
            with self.assertRaises(ValueError):
                bench.validate_args(args)
            self.assertFalse(self.out.exists())

    def fake_process(self, command, environment, timeout):
        self.calls.append((list(command), dict(environment), timeout))
        name = Path(command[0]).stem
        record = dict(command=list(command), returncode=0, stdout="", stderr="", wall_seconds=0.5)
        if "--version" in command:
            record["stdout"] = name + " fake version"
            return record
        if name == self.failure:
            record.update(returncode=None, error="Process timeout after guard", stderr="partial scout log")
            return record
        if name == "djxl":
            data = Path(command[1]).read_bytes()[3:]
            if self.mismatch == "png":
                data = make_png(16, 6, [[258, 513, 65534, 0, 1, 257, 32769, 65535]])
            elif self.mismatch == "jpeg":
                data = data[:-1]
            Path(command[2]).write_bytes(data)
        else:
            durations = self.times[name]
            record["wall_seconds"] = durations[self.counts[name] % len(durations)]
            self.counts[name] += 1
            Path(command[3]).write_bytes(b"JXL" + Path(command[2]).read_bytes())
            if command[1] == "encode" and not self.no_search_evidence:
                record["stderr"] = ("source mode: effort=9 search=true sample_limit=33554432\n"
                                    "source search: full_encodes=3 probe_encodes=4 winner=1\n"
                                    "scout: candidates=2\n")
        return record

    def run_benchmark(self, *extra, cases=None, process=None):
        args = bench.parser().parse_args(self.arguments(*extra, cases=cases))
        with patch.object(bench.port, "run_process", side_effect=process or self.fake_process), \
                patch.object(bench, "jpeg_dimensions", return_value=(2, 1)), \
                redirect_stdout(StringIO()):
            report = bench.benchmark(args)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual({path.name for path in self.out.iterdir()}, {"report.json", "REPORT.md"})
        self.assertEqual(json.loads((self.out / "report.json").read_text()), report)
        return report

    def test_serial_rotation_warmups_medians_diagnostics_and_identities(self):
        self.times = {"before": [99, 2, 4], "after": [77, 1, 3]}
        with patch.dict(os.environ, {"JXL_INHERITED": "1", "RS_JXL_PROFILE": "1"}):
            report = self.run_benchmark()
        self.assertEqual(report["status"], "passed", report["errors"])
        rows = report["rows"]
        self.assertEqual([row["encoder"] for row in rows],
                         ["before", "after", "after", "before", "before", "after"])
        self.assertEqual([row["warmup"] for row in rows], [True, True, False, False, False, False])
        summary = report["summaries"][0]
        self.assertEqual(summary["before"]["median_wall_seconds"], 3)
        self.assertEqual(summary["after"]["median_wall_seconds"], 2)
        self.assertEqual(summary["before_over_after_wall_speedup"], 1.5)
        self.assertEqual(summary["after_over_before_bytes"], 1)
        self.assertEqual(summary["before"]["mpix_per_second"], 2 / 1e6 / 3)
        self.assertEqual(summary["before"]["median_bpp"], rows[2]["bytes"] * 8 / 2)
        processes = [command for command, _, _ in self.calls if "--version" not in command]
        self.assertEqual(len(processes), 12)
        for encode, decode in zip(processes[::2], processes[1::2]):
            self.assertEqual(encode[1], "encode")
            self.assertEqual(Path(decode[0]).stem, "djxl")
            self.assertEqual(encode[3], decode[1])
            self.assertIn("--bits_per_sample=16", decode)
            self.assertIn("--num_threads=1", decode)
            self.assertTrue(Path(encode[3]).is_relative_to(self.out))
        for row in rows:
            self.assertTrue(row["pixels_match_including_hidden_rgb"])
            self.assertEqual(row["source_search"]["evidence"], "source-search summary observed")
            self.assertIn("scout: candidates=2", row["source_search"]["lines"])
        for info in report["binaries"].values():
            self.assertTrue(info["unchanged_during_run"])
            self.assertEqual(len(info["sha256"]), 64)
            self.assertTrue(info["version_probe"]["command"])
        for _, env, _ in self.calls:
            self.assertNotIn("JXL_INHERITED", env)
            self.assertNotIn("RS_JXL_PROFILE", env)
        self.assertEqual(self.png.read_bytes(), fixture())

    def test_png_snapshot_strips_metadata_not_original_samples(self):
        self.png.write_bytes(make_png(16, 6, [[259, 513, 65534, 0, 1, 257, 32769, 65535]],
                                     extra=[(b"tEXt", b"key\0value")]))
        original = self.png.read_bytes()
        report = self.run_benchmark()
        self.assertEqual(report["status"], "passed", report["errors"])
        case = report["cases"][0]
        self.assertEqual(case["removed_metadata_chunks"], ["tEXt"])
        self.assertEqual(case["payload_sha256"], bench.port.sha256(fixture()))
        self.assertEqual(case["original_sha256"], bench.port.sha256(original))
        self.assertEqual(self.png.read_bytes(), original)

    def test_mixed_cases_jpeg_original_bytes_and_per_binary_flags(self):
        report = self.run_benchmark("--before-png-arg=--baseline-scout=off", "--after-arg=--source-scout",
                                    "--after-jpeg-arg=--order=source",
                                    cases=[self.png, self.jpeg])
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(len(report["summaries"]), 2)
        self.assertEqual([row["encoder"] for row in report["rows"][6:8]], ["after", "before"])
        jpeg = report["cases"][1]
        self.assertEqual(jpeg["payload_sha256"], bench.port.sha256(self.jpeg_bytes))
        self.assertEqual(jpeg["payload_sha256"], jpeg["original_sha256"])
        for row in report["rows"]:
            command = row["process"]["command"]
            self.assertEqual("--baseline-scout=off" in command,
                             row["encoder"] == "before" and row["case_index"] == 0)
            self.assertEqual("--source-scout" in command,
                             row["encoder"] == "after" and row["case_index"] == 0)
            self.assertEqual("--order=source" in command,
                             row["encoder"] == "after" and row["case_index"] == 1)
            if row["case_index"] == 1:
                self.assertEqual(command[1], "transcode-jpeg")
                self.assertIn("--reconstruct_jpeg", row["decode"]["command"])
                self.assertTrue(row["original_jpeg_bytes_match"])
                self.assertFalse(row["source_search"]["diagnostics_observed"])
                self.assertNotIn("pixels_match_including_hidden_rgb", row)
        self.assertEqual(report["config"]["encode_args"]["after"],
                         dict(png=["--source-scout"], jpeg=["--order=source"]))
        self.assertEqual(self.jpeg.read_bytes(), self.jpeg_bytes)

    def test_hidden_rgb_or_jpeg_byte_mismatch_never_passes(self):
        for kind, path, message in (("png", self.png, "hidden RGB"),
                                    ("jpeg", self.jpeg, "JPEG bytes differ")):
            with self.subTest(kind=kind):
                self.mismatch = kind
                report = self.run_benchmark(cases=[path])
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(report["rows"]), 1)
                self.assertIn(message, report["rows"][0]["error"])
                self.assertNotIn("before", report["summaries"][0])

    def test_encode_or_decode_timeout_cleans_and_fails(self):
        for name in ("before", "after", "djxl"):
            with self.subTest(name=name):
                self.failure = name
                report = self.run_benchmark()
                self.assertEqual(report["status"], "failed")
                self.assertIn("timeout", report["rows"][-1]["error"])
                self.assertNotIn("before", report["summaries"][0])

    def test_version_probe_timeout_fails_before_encoding(self):
        def timeout(command, env, limit):
            return dict(command=list(command), returncode=None, error="version timeout",
                        stdout="", stderr="", wall_seconds=limit)

        report = self.run_benchmark(process=timeout)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["rows"], [])
        self.assertIn("timeout", report["errors"][0])

    def test_no_search_evidence_is_not_a_valid_png_trial(self):
        self.no_search_evidence = True
        report = self.run_benchmark()
        self.assertEqual(report["status"], "failed")
        self.assertIn("diagnostics", report["rows"][0]["error"])
        self.assertNotIn("decode", report["rows"][0])

    def test_missing_or_empty_jxl_nonzero_exit_and_bad_wall_time_fail(self):
        for failure in ("missing", "empty", "exit", "zero", "negative"):
            def fail(command, env, limit):
                record = self.fake_process(command, env, limit)
                if command[1] == "encode":
                    if failure == "missing":
                        Path(command[3]).unlink()
                    elif failure == "empty":
                        Path(command[3]).write_bytes(b"")
                    elif failure == "exit":
                        record["returncode"] = 2
                    else:
                        record["wall_seconds"] = 0 if failure == "zero" else -1
                return record

            with self.subTest(failure=failure):
                report = self.run_benchmark(process=fail)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["rows"][0]["status"], "failed")

    def test_interrupt_after_one_trial_preserves_failure_and_cleans(self):
        def interrupted(command, env, limit):
            if Path(command[0]).stem == "after" and "--version" not in command:
                raise KeyboardInterrupt
            return self.fake_process(command, env, limit)

        report = self.run_benchmark("--warmups", "0", process=interrupted)
        self.assertEqual(report["status"], "failed")
        self.assertEqual([row["status"] for row in report["rows"]], ["passed", "failed"])
        self.assertIn("KeyboardInterrupt", report["rows"][1]["error"])

    def test_changed_binary_or_original_invalidates_all_medians(self):
        for path in (self.root / "after.exe", self.png):
            before = path.read_bytes()

            def changed(command, env, limit):
                result = self.fake_process(command, env, limit)
                if command[1] == "encode":
                    path.write_bytes(before + b"changed")
                return result

            with self.subTest(path=path):
                report = self.run_benchmark(process=changed)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(all(row["status"] == "passed" for row in report["rows"]))
                self.assertTrue(any("measurements are invalid" in error for error in report["errors"]))
                self.assertNotIn("before", report["summaries"][0])
            path.write_bytes(before)

    def test_later_input_error_cannot_mark_completed_case_or_run_passed(self):
        invalid = self.root / "invalid.png"
        invalid.write_bytes(b"not a PNG or JPEG")
        report = self.run_benchmark(cases=[self.png, invalid])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["rows"]), 6)
        self.assertTrue(all(row["status"] == "passed" for row in report["rows"]))
        self.assertEqual(report["summaries"][0]["status"], "failed")
        self.assertNotIn("before", report["summaries"][0])

    def test_missing_duplicate_or_failed_warmup_rows_fail_summary(self):
        report = self.run_benchmark()
        rows = report["rows"]
        variants = [rows[:-1], rows + [rows[0]],
                    [dict(row, status="failed") if index == 0 else row for index, row in enumerate(rows)]]
        for variant in variants:
            with self.subTest(length=len(variant)):
                summary = bench.summarize(report["cases"], variant, 2, 1, [])
                self.assertEqual(summary[0]["status"], "failed")
                self.assertNotIn("before", summary[0])

    def test_invalid_settings_existing_output_and_required_cases(self):
        for extra in (("--timeout", "nan"), ("--timeout", "inf"), ("--timeout", "0"),
                      ("--repeats", "0"), ("--warmups", "-1"), ("--source-sample-limit", "0")):
            with self.subTest(extra=extra):
                args = bench.parser().parse_args(self.arguments(*extra))
                with self.assertRaises(ValueError):
                    bench.validate_args(args)
                self.assertFalse(self.out.exists())
        args = bench.parser().parse_args(self.arguments())
        self.out.mkdir()
        with self.assertRaises(FileExistsError), patch.object(bench.port, "run_process") as process:
            bench.benchmark(args)
        process.assert_not_called()
        argv = self.arguments()
        del argv[argv.index("--case"):]
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.parser().parse_args(argv)

    def test_main_failure_exit_and_exact_invocation_report(self):
        self.mismatch = "png"
        argv = self.arguments()
        with patch.object(bench.port, "run_process", side_effect=self.fake_process), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(bench.main(argv), 1)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["invocation"][2:], argv)
        self.assertEqual(report["status"], "failed")

    def test_real_process_helper_waits_and_reports_timeout_as_failure(self):
        with patch.object(bench.port.subprocess, "run", side_effect=subprocess.TimeoutExpired(
                ["fake"], 0.1, stderr=b"partial")):
            record = bench.port.run_process(["fake"], {}, 0.1)
        self.assertEqual(record["stderr"], "partial")
        with self.assertRaises(RuntimeError):
            bench.port.check_process(record)


if __name__ == "__main__":
    unittest.main()
