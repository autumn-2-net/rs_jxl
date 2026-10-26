"""Lightweight tests of the validation harness, without invoking cargo."""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("verify_platform", Path(__file__).with_name("verify_platform.py"))
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PlatformRunnerTests(unittest.TestCase):
    def test_windows_probe_never_invokes_or_looks_up_wsl(self):
        with tempfile.TemporaryDirectory(prefix="rs-jxl-platform-test-") as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            (root / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
            commands, lookups = [], []

            def available(name):
                lookups.append(name)
                return name  # Even if WSL is installed, it must not be queried.

            def fake_run(argv, cwd, out, label, timeout):
                commands.append(argv)
                return {"label": label, "argv": argv, "status": "passed", "returncode": 0,
                        "stdout_tail": "host: x86_64-pc-windows-msvc\n", "stderr_tail": ""}

            with (patch.object(sys, "argv", ["verify_platform.py", "--mode", "probe",
                                             "--repo", str(root), "--out", str(root / "report")]),
                  patch.object(runner, "os", SimpleNamespace(name="nt", environ={}, cpu_count=lambda: 4)),
                  patch.object(runner.platform, "system", return_value="Windows"),
                  patch.object(runner.platform, "machine", return_value="AMD64"),
                  patch.object(runner.shutil, "which", side_effect=available),
                  patch.object(runner, "run_command", side_effect=fake_run),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 0)
            self.assertTrue(commands)
            self.assertFalse(any("wsl" in str(argument).lower() for command in commands for argument in command))
            self.assertNotIn("wsl", lookups)
            report = json.loads((root / "report/report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["validation_status"], "not_run")
            self.assertNotIn("wsl", report["executables"])

    def test_probe_has_no_build_or_test_commands(self):
        self.assertEqual(runner.validation_commands(["cargo"], "probe", "lossless", None), [])

    def test_check_only_accepts_cross_target(self):
        commands = runner.validation_commands(["cargo"], "check", "lossless", "x86_64-unknown-linux-gnu")
        self.assertEqual(len(commands), 1)
        self.assertIn("check", commands[0])
        self.assertNotIn("test", commands[0])
        self.assertEqual(commands[0][-2:], ["--target", "x86_64-unknown-linux-gnu"])

    def test_lossless_commands_include_hostile_tests_are_offline_serial_and_locked(self):
        cargo = ["rustup", "run", "1.89.0", "cargo"]
        commands = runner.validation_commands(cargo, "test", "lossless", None)
        self.assertEqual(len(commands), 2)
        self.assertIn("decode_hostile", commands[0])
        self.assertIn("decode_wide_integer", commands[0])
        self.assertIn("decode_streaming", commands[0])
        self.assertIn("jpeg_dc_refinement", commands[0])
        for command in commands:
            self.assertEqual(command[:len(cargo)], cargo)
            self.assertIn("--offline", command)
            self.assertIn("--locked", command)
            self.assertEqual(command[-2:], ["--", "--test-threads=1"])
            self.assertNotIn("--ignored", command)
            self.assertNotIn("--include-ignored", command)

    def test_binaries_mode_does_not_compile(self):
        self.assertEqual(runner.validation_commands(["cargo"], "binaries", "lossless", None), [])

    def test_non_executable_is_not_runtime_evidence(self):
        with tempfile.TemporaryDirectory(prefix="rs-jxl-platform-test-") as directory:
            path = Path(directory) / "not-executable"
            path.write_bytes(b"a report is not a native test executable")
            with self.assertRaises(ValueError):
                runner.native_binary_kind(path)

    def test_nonzero_exit_is_not_success(self):
        with tempfile.TemporaryDirectory(prefix="rs-jxl-platform-test-") as directory:
            path = Path(directory)
            command = [sys.executable, "-B", "-c", "import sys; print('failure evidence'); sys.exit(7)"]
            row = runner.run_command(command, path, path, "fail", 10)
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["returncode"], 7)
            self.assertEqual(row["argv"], command)
            self.assertIn("failure evidence", row["stdout_tail"])

    def test_timeout_is_failure_and_child_is_reaped(self):
        with tempfile.TemporaryDirectory(prefix="rs-jxl-platform-test-") as directory:
            path = Path(directory)
            row = runner.run_command(
                [sys.executable, "-B", "-c", "import time; time.sleep(30)"], path, path, "timeout", 0.1,
            )
            self.assertEqual(row["status"], "timed_out")
            self.assertIsNotNone(row["cleanup"]["reaped_returncode"])
            self.assertIsNotNone(row["returncode"])

    def test_missing_executable_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="rs-jxl-platform-test-") as directory:
            path = Path(directory)
            row = runner.run_command([str(path / "not-installed")], path, path, "missing", 10)
            self.assertEqual(row["status"], "start_failed")
            self.assertIsNone(row["returncode"])
            self.assertIn("error", row)

    def test_utf16_wsl_and_utf8_output(self):
        self.assertEqual(runner.decode_output("Ubuntu\r\n".encode("utf-16-le")), "Ubuntu\r\n")
        self.assertEqual(runner.decode_output(b"rustc 1.89.0\n"), "rustc 1.89.0\n")


if __name__ == "__main__":
    unittest.main()
