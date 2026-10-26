"""Focused harness tests; fake subprocesses only, no builds or real JXL codecs."""

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import compare_cpp_port as port
import compare_decoder_checkpoints as bench
from test_compare_cpp_port import fixture, make_png


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.argv = []
        for option, name in (("before", "before"), ("after", "after"), ("official-djxl", "official")):
            path = self.root / (name + ".exe")
            path.write_bytes(name.encode())
            self.argv += ["--" + option, str(path)]
        # Deliberately misleading extensions: classification must inspect content.
        self.png = self.root / "integer.jpg"
        self.png.write_bytes(fixture())
        self.jpeg = self.root / "archival.png"
        image = BytesIO()
        Image.new("RGB", (2, 1), (19, 81, 253)).save(image, format="JPEG")
        self.jpeg.write_bytes(image.getvalue())
        self.originals = {}
        for name, original in (("pixels", self.png), ("jpeg", self.jpeg)):
            path = self.root / (name + ".jxl")
            path.write_bytes(b"pre-encoded fixture " + name.encode())
            self.originals[str(path)] = original
            self.argv += ["--case", str(path), str(original)]
        self.calls = []
        self.counts = {}
        self.failure = self.mismatch = self.change = None
        self.run_count = 0

    def fake_process(self, command, environment, timeout):
        self.calls.append((list(command), dict(environment), timeout))
        name = Path(command[0]).stem
        phase = "version" if "--version" in command else "decode" if "--disable_output" in command else "verify"
        record = dict(command=list(command), returncode=0, stdout="", stderr="", wall_seconds=1)
        if self.failure and (phase, name) == self.failure[:2]:
            record.update(self.failure[2])
            return record
        if phase == "version":
            record["stdout"] = name + " fake version"
            return record
        source_index = 1 if name == "official" else 2
        source = command[source_index]
        if phase == "verify":
            original = self.originals[source]
            restored = original.read_bytes()
            if self.mismatch == "png" and original == self.png:
                restored = make_png(16, 6, [[258, 513, 65534, 0, 1, 257, 32769, 65535]])
            elif self.mismatch == "jpeg" and original == self.jpeg:
                restored += b"changed"
            Path(command[source_index + 1]).write_bytes(restored)
        else:
            threads = next(arg for arg in command if arg.startswith("--num_threads="))
            key = (source, name, threads)
            count = self.counts.get(key, 0)
            self.counts[key] = count + 1
            record["wall_seconds"] = (10000, 10, 30, 999)[count] * {"before": 2, "after": 1, "official": .5}[name]
            if self.change:
                self.change.write_bytes(b"changed during timing")
        return record

    def run_harness(self, extra=()):
        out = self.root / f"report-{self.run_count}"
        self.run_count += 1
        self.calls.clear()
        self.counts.clear()
        with patch.object(bench, "run_process", side_effect=self.fake_process), \
                patch.object(port, "run_process", side_effect=self.fake_process), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = bench.main([*self.argv, "--out", str(out), *extra])
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0 if report["status"] == "passed" else 1)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual({p.name for p in out.iterdir()}, {"report.json", "REPORT.md"})
        self.assertIn(report["status"], (out / "REPORT.md").read_text(encoding="utf-8"))
        return report

    def test_exact_verification_commands_rotation_and_robust_metrics(self):
        with patch.dict(os.environ, {"JXL_TEST": "1", "RS_JXL_PROFILE": "trace"}):
            report = self.run_harness()
        self.assertEqual(report["status"], "passed")
        self.assertEqual([case["kind"] for case in report["cases"]], ["png", "jpeg"])
        self.assertEqual(report["cases"][0]["depth"], 16)
        self.assertEqual(len(report["trials"]), 2 * (6 + 2 * 4 * 3))
        for case in report["cases"]:
            rows = [r for r in report["trials"] if r["case_index"] == case["index"]]
            self.assertTrue(all(r["phase"] == "verify" and r["verified"] for r in rows[:6]))
            self.assertEqual({(r["decoder"], r["threads"]) for r in rows[:6]},
                             {(name, threads) for name in bench.NAMES for threads in (1, 8)})
            for thread_index, threads in enumerate((1, 8)):
                for repeat in (-1, 0, 1, 2):
                    timed = [r for r in rows[6:] if r["threads"] == threads and r["repeat"] == repeat]
                    start = (case["index"] + thread_index + repeat) % 3
                    self.assertEqual([r["decoder"] for r in timed], list(bench.NAMES[start:] + bench.NAMES[:start]))
            for row in rows:
                cmd = row["command"]
                timed, rust = row["phase"] == "decode", row["decoder"] != "official"
                self.assertEqual("decode" in cmd, rust)
                self.assertIn(f"--num_threads={row['threads']}", cmd)
                self.assertEqual("--reconstruct_jpeg" in cmd, case["kind"] == "jpeg")
                self.assertNotIn("--pixels_to_jpeg", cmd)
                self.assertEqual("--bits=32" in cmd, timed and rust and case["kind"] == "png")
                self.assertEqual("--bits_per_sample=16" in cmd, not timed and case["kind"] == "png")
                if timed:
                    self.assertIn("--num_reps=10", cmd)
                    self.assertEqual(len([s for s in cmd[3 if rust else 2:] if not s.startswith("--")]), 0)
        for summary in report["summaries"]:
            self.assertEqual(summary["decoders"]["before"]["median_seconds"], 6)
            self.assertEqual(summary["decoders"]["after"]["median_seconds"], 3)
            self.assertEqual(summary["decoders"]["after"]["mpix_s"], 2 / 1e6 / 3)
            self.assertEqual(summary["before_over_after_speedup"], 2)
            case = report["cases"][summary["case_index"]]
            self.assertEqual(summary["bpp"], case["jxl_bytes"] * 8 / 2)
        for name, info in report["binaries"].items():
            self.assertEqual(info["sha256"], port.file_sha256(self.root / (name + ".exe")))
        markdown = (self.root / "report-0" / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("--disable_output", markdown)
        self.assertIn(report["binaries"]["before"]["sha256"], markdown)
        for _, env, timeout in self.calls:
            self.assertNotIn("JXL_TEST", env)
            self.assertNotIn("RS_JXL_PROFILE", env)
            self.assertTrue(0 < timeout <= 300)
        self.assertEqual(self.png.read_bytes(), fixture())

    def test_integer_hidden_rgb_and_jpeg_byte_mismatches_stop_case_before_timing(self):
        for kind, index, message in (("png", 0, "hidden RGB"), ("jpeg", 1, "JPEG bytes")):
            with self.subTest(kind=kind):
                self.mismatch = kind
                report = self.run_harness()
                self.assertEqual(report["status"], "failed")
                self.assertIn(message, report["errors"][0])
                self.assertEqual(report["trials"][-1]["status"], "failed")
                self.assertFalse(any(r["phase"] == "decode" and r["case_index"] == index
                                     for r in report["trials"]))

    def test_timeouts_nonzero_exits_and_version_failures_are_reported(self):
        for phase, name, failure in (
            ("verify", "after", dict(returncode=None, error="Process timeout after 0.2s")),
            ("decode", "official", dict(returncode=None, error="Process timeout after 0.2s")),
            ("decode", "before", dict(returncode=7, stderr="decoder failed")),
            ("version", "official", dict(returncode=None, error="Process timeout after 0.2s")),
        ):
            with self.subTest(phase=phase, name=name):
                self.failure = (phase, name, failure)
                report = self.run_harness(["--timeout", "0.2"])
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                self.assertEqual(report["summaries"], [])
                self.assertTrue(all(timeout == .2 for _, _, timeout in self.calls))
                if phase == "version":
                    self.assertEqual(report["trials"], [])
                    self.assertIn("error", report["binaries"][name]["version_probe"])
                else:
                    self.assertEqual(report["trials"][-1]["returncode"], failure["returncode"])

    def test_invalid_settings_and_existing_output_directory_are_rejected(self):
        out = self.root / "existing"
        out.mkdir()
        sentinel = out / "keep"
        sentinel.write_bytes(b"untouched")
        argv = [*self.argv, "--out", str(out)]
        for extra in (["--threads", "0"], ["--threads", "257"], ["--threads", "1", "1"],
                      ["--repeats", "0"], ["--warmups", "-1"], ["--decode-reps", "0"],
                      ["--timeout", "nan"], ["--timeout", "inf"], ["--timeout", "0"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                bench.validate_args(bench.parser().parse_args([*argv, *extra]))
        with patch.object(bench, "binary_info") as probe, self.assertRaises(FileExistsError):
            bench.benchmark(bench.parser().parse_args(argv))
        probe.assert_not_called()
        self.assertEqual(list(out.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_changed_checkpoint_or_jxl_invalidates_results(self):
        for path in (self.root / "after.exe", self.root / "pixels.jxl"):
            with self.subTest(path=path):
                self.change = path
                report = self.run_harness()
                self.assertEqual(report["status"], "failed")
                self.assertIn("measurements invalid", report["errors"][0])
                self.assertIn(str(path), report["errors"][0])

    def test_unsupported_or_missing_original_is_a_reported_failure_not_a_skip(self):
        for payload in (b"not an image", make_png(16, 6, [[1] * 8], extra=[(b"acTL", b"\0" * 8)]), None):
            with self.subTest(payload=payload):
                if payload is None:
                    self.png.unlink()
                else:
                    self.png.write_bytes(payload)
                report = self.run_harness()
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                self.assertEqual(report["trials"], [])
                self.assertTrue(all("--version" in command for command, _, _ in self.calls))


if __name__ == "__main__":
    unittest.main()
