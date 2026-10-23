"""Lightweight tests only: no Rust/C++ build, real encoder or djxl execution."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

import compare_cpp_port as port


def chunk(kind, body):
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body)))


def make_png(depth, color, rows, *, width=2, extra=(), interlace=False):
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    header = struct.pack(">IIBBBBB", width, len(rows), depth, color, 0, 0, int(interlace))
    raw = bytearray()
    passes = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
              (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)) if interlace else ((0, 0, 1, 1),)
    for x0, y0, dx, dy in passes:
        if x0 >= width:
            continue
        for y in range(y0, len(rows), dy):
            samples = [value for x in range(x0, width, dx)
                       for value in rows[y][x * channels:(x + 1) * channels]]
            raw.append(0)
            raw.extend(struct.pack(">" + ("B" if depth == 8 else "H") * len(samples), *samples))
    return (port.PNG_SIGNATURE + chunk(b"IHDR", header)
            + b"".join(chunk(kind, body) for kind, body in extra)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def fixture(depth=16):
    samples = [259, 513, 65534, 0, 1, 257, 32769, 65535] if depth == 16 else [19, 81, 253, 0, 7, 11, 13, 255]
    return make_png(depth, 6, [samples])


class PngTests(unittest.TestCase):
    def test_gray_ga_rgb_rgba_at_both_depths(self):
        for depth in (8, 16):
            top = (1 << depth) - 1
            a, b, c = (3, 19, 253) if depth == 8 else (259, 513, 65534)
            for color, samples, expected in (
                (0, [a, c], [[a, a, a, top], [c, c, c, top]]),
                (2, [a, b, c, c, b, a], [[a, b, c, top], [c, b, a, top]]),
                (4, [a, 0, c, top], [[a, a, a, 0], [c, c, c, top]]),
                (6, [a, b, c, 0, c, b, a, top], [[a, b, c, 0], [c, b, a, top]]),
            ):
                with self.subTest(depth=depth, color=color):
                    decoded = port.png_samples(make_png(depth, color, [samples]))
                    self.assertEqual(decoded.depth, depth)
                    self.assertEqual(decoded.rgba.tolist(), [expected])

    def test_hidden_rgb_and_sixteen_bit_low_bits_are_compared(self):
        for depth in (8, 16):
            expected = port.png_samples(fixture(depth))
            actual = port.png_samples(fixture(depth))
            actual.rgba[0, 0, 0] ^= 1
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, "hidden RGB"):
                port.compare_pixels(expected, actual)

    def test_transparency_keys_keep_hidden_samples(self):
        for depth in (8, 16):
            a, b, c = (3, 19, 253) if depth == 8 else (259, 513, 65534)
            for color, samples, key in ((0, [a, c], [a]), (2, [a, b, c, c, b, a], [a, b, c])):
                png = make_png(depth, color, [samples], extra=[(b"tRNS", struct.pack(">" + "H" * len(key), *key))])
                actual = port.png_samples(png)
                self.assertEqual(actual.rgba[0, 0, 3], 0)
                self.assertEqual(actual.rgba[0, 0, 0], a)
                self.assertEqual(actual.rgba[0, 1, 3], (1 << depth) - 1)

    def test_palette_hidden_rgb_is_expanded(self):
        png = make_png(8, 3, [[0, 1]], extra=[(b"PLTE", bytes([19, 81, 253, 7, 11, 13])), (b"tRNS", b"\0\xff")])
        self.assertEqual(port.png_samples(png).rgba.tolist(), [[[19, 81, 253, 0], [7, 11, 13, 255]]])

    def test_adam7_preserves_sixteen_bit_values(self):
        rows = [[259 + x + y * 50 for x in range(20)] for y in range(5)]
        for depth in (8, 16):
            samples = [[value % (1 << depth) for value in row] for row in rows]
            expected = port.png_samples(make_png(depth, 6, samples, width=5))
            actual = port.png_samples(make_png(depth, 6, samples, width=5, interlace=True))
            port.compare_pixels(expected, actual)

    def test_metadata_stripping_never_recompresses_or_rescales_samples(self):
        png = make_png(16, 6, [[259, 513, 65534, 0, 1, 257, 32769, 65535]],
                       extra=[(b"gAMA", struct.pack(">I", 45455)),
                              (b"sBIT", bytes([12, 12, 12, 12])), (b"tEXt", b"key\0value")])
        payload, info = port.png_payload(png)
        self.assertEqual(payload, fixture())
        self.assertEqual(info["removed_metadata_chunks"], ["gAMA", "sBIT", "tEXt"])
        port.compare_pixels(port.png_samples(png), port.png_samples(payload))

    def test_reject_non_png_apng_low_bits_bad_crc_and_truncation(self):
        png = fixture()
        header1 = struct.pack(">IIBBBBB", 2, 1, 1, 0, 0, 0, 0)
        cases = [b"", b"\xff\xd8\xffJPEG", png[:-2], png + b"trailing",
                 png[:33] + chunk(b"acTL", struct.pack(">II", 1, 0)) + png[33:],
                 port.PNG_SIGNATURE + chunk(b"IHDR", header1) + png[33:],
                 png[:33] + chunk(b"ABCD", b"unknown") + png[33:]]
        corrupt = bytearray(png)
        corrupt[29] ^= 1
        cases.append(bytes(corrupt))
        for data in cases:
            with self.subTest(data=data[:16]), self.assertRaises(ValueError):
                port.png_samples(data)

    def test_depth_change_is_not_silently_normalized(self):
        with self.assertRaisesRegex(ValueError, "bit depth"):
            port.compare_pixels(port.png_samples(fixture()), port.png_samples(fixture(8)))

    def test_oracle_narrowing_is_rejected(self):
        cv2, np = port.pixel_libraries()
        with patch.object(cv2, "imdecode", return_value=np.zeros((1, 2, 4), dtype=np.uint8)):
            with self.assertRaisesRegex(ValueError, "narrowed"):
                port.png_samples(fixture())


class ProtocolTests(unittest.TestCase):
    def test_environments_are_clean_explicit_and_case_insensitive(self):
        parent = {"PATH": "bin", "JXL_MA_SEARCH": "1", "jxl_hidden": "1", "RS_JXL_PROFILE": "trace"}
        env, record = port.clean_environment(parent, ["JXL_LOSSLESS_CALIBRATE=1"], cpp=True)
        self.assertEqual(env, {"PATH": "bin", "JXL_LOSSLESS_CALIBRATE": "1", "JXL_LOSSLESS_SEARCH_TRACE": "1"})
        self.assertEqual(record["explicit"]["JXL_LOSSLESS_SEARCH_TRACE"], "1")
        self.assertEqual(set(record["removed_inherited_names"]), set(parent) - {"PATH"})
        self.assertEqual(parent["JXL_MA_SEARCH"], "1")
        decoder, _ = port.clean_environment(parent)
        self.assertEqual(decoder, {"PATH": "bin"})

    def test_invalid_or_conflicting_overrides(self):
        for overrides in (["PATH=x"], ["JXL_X"], ["JXL_X=1", "JXL_X=2"],
                          ["JXL_X=\0"], ["JXL_LOSSLESS_SEARCH_TRACE=0"], ["jxl_x=1"]):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                port.clean_environment({}, overrides, cpp=True)

    def test_commands_have_same_threads_and_distinct_search_policies(self):
        for threads in (1, 8):
            for effort in (1, 3, 7, 9, 10):
                for mode in port.MODES:
                    rust = port.encoder_command("rust", "rust", "same.png", "r.jxl", threads, effort, mode, 12345)
                    cpp = port.encoder_command("cpp", "cpp", "same.png", "c.jxl", threads, effort, mode, 12345)
                    self.assertEqual(rust[2], cpp[1])
                    self.assertEqual(rust[rust.index("--threads") + 1], str(threads))
                    self.assertIn(f"--num_threads={threads}", cpp)
                    self.assertIn("--distance=0", cpp)
                    self.assertIn("--lossless_jpeg=0", cpp)
                    self.assertIn("--keep_invisible=1", cpp)
                    self.assertEqual("--source-search" in rust, mode == "full-scheduler")
                    self.assertEqual("--iterations=50" in cpp, mode == "single-candidate")
                    self.assertEqual(rust[rust.index("--source-effort") + 1], str(effort))
                    self.assertIn(f"--effort={effort}", cpp)

    def test_decoder_depth_is_explicit_and_no_color_conversion(self):
        for depth in (8, 16):
            command = port.decoder_command("official", "a.jxl", "a.png", depth)
            self.assertIn(f"--bits_per_sample={depth}", command)
            self.assertIn("--num_threads=1", command)
            self.assertFalse(any("color_space" in arg or "alpha_blend" in arg for arg in command))

    def test_search_must_be_observed_for_full_scheduler(self):
        trace = "lossless-search event=start schema=1\nlossless-search event=end schema=1\n"
        port.scheduler_evidence("cpp", "full-scheduler", trace)
        port.scheduler_evidence("cpp", "single-candidate", "normal CLI")
        for mode, stderr in (("full-scheduler", ""), ("single-candidate", trace)):
            with self.assertRaises(ValueError):
                port.scheduler_evidence("cpp", mode, stderr)
        port.scheduler_evidence("rust", "single-candidate", "source mode: search=false sample_limit=1")
        port.scheduler_evidence("rust", "full-scheduler", "source mode: search=true sample_limit=1\nsource search: full_encodes=2")
        with self.assertRaises(ValueError):
            port.scheduler_evidence("rust", "full-scheduler", "source mode: search=true sample_limit=1")

    def test_timeout_and_exit_failure_are_structured(self):
        for side_effect, result in (
            (subprocess.TimeoutExpired(["fake"], 0.1, stderr=b"partial"), None),
            (None, subprocess.CompletedProcess(["fake"], 3, b"", b"failed")),
            (FileNotFoundError("missing"), None),
        ):
            with patch.object(port.subprocess, "run", side_effect=side_effect, return_value=result):
                record = port.run_process(["fake"], {}, 0.1)
                with self.assertRaises(RuntimeError):
                    port.check_process(record)
                self.assertGreaterEqual(record["wall_seconds"], 0)

    def test_provenance_does_not_trust_matching_version_cache_or_current_head(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            binary = root / "libjxl" / "tools" / "cjxl.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fake")
            (binary.parent / "tool_version_git.h").write_text(f'#define JPEGXL_VERSION "{port.CPP_COMMIT}"')
            (root / "CMakeCache.txt").write_text("JXL_SOURCE_DIR:PATH=D:/frozen/libjxl\n")
            (root / ".ninja_log").write_text("1\t2\t3\tlibjxl/tools/cjxl.exe\tcommand-hash\n")
            with patch.object(port, "run_process", return_value={"returncode": 0, "stdout": port.CPP_COMMIT}):
                report = port.cpp_provenance(binary, root)
            self.assertEqual(report["status"], "unverified")
            self.assertEqual(report["expected_commit"], port.CPP_COMMIT)
            self.assertEqual(len(report["build_evidence"]), 3)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "source.png"
        self.image.write_bytes(fixture())
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        for name in ("rust.exe", "cpp.exe", "djxl.exe", "djxl2.exe"):
            (self.root / name).write_bytes(name.encode())
        self.argv = ["--rust", str(self.root / "rust.exe"), "--cpp", str(self.root / "cpp.exe"),
                     "--official-djxl", str(self.root / "djxl.exe"), "--official-djxl", str(self.root / "djxl2.exe"),
                     "--mode", "both", "--temp-root", str(self.scratch), str(self.image)]
        self.calls = []
        self.failure = None
        self.mismatch = False

    def fake_process(self, command, environment, timeout):
        self.calls.append((command, dict(environment)))
        binary = Path(command[0]).name
        record = dict(command=command, returncode=0, stdout="", stderr="", wall_seconds=2 if binary == "rust.exe" else 1)
        if "--version" in command:
            record["stdout"] = binary + " fake version"
            return record
        if binary == self.failure:
            record.update(returncode=None, error="simulated timeout")
            return record
        if binary.startswith("djxl"):
            png = Path(command[1]).read_bytes()[3:]
            if self.mismatch:
                png = make_png(16, 6, [[258, 513, 65534, 0, 1, 257, 32769, 65535]])
            Path(command[2]).write_bytes(png)
        elif binary == "rust.exe":
            Path(command[3]).write_bytes(b"JXL" + Path(command[2]).read_bytes())
            search = "--source-search" in command
            record["stderr"] = f"source mode: search={str(search).lower()} sample_limit=1048576\n"
            if search:
                record["stderr"] += "source search: full_encodes=2 probe_encodes=1 winner=0\n"
        else:
            Path(command[2]).write_bytes(b"JXL" + Path(command[1]).read_bytes())
            if "--iterations=50" not in command:
                record["stderr"] = "lossless-search event=start schema=1\nlossless-search event=end schema=1\n"
        return record

    def run_benchmark(self, extra=()):
        args = port.parser().parse_args([*extra, *self.argv])
        with patch.object(port, "run_process", side_effect=self.fake_process), \
                patch.object(port, "cpp_provenance", return_value={"status": "unverified"}), \
                redirect_stdout(StringIO()):
            report = port.benchmark(args)
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual(self.image.read_bytes(), fixture())
        return report

    def test_paired_repeats_warmups_metrics_order_and_all_decoders(self):
        with patch.dict(os.environ, {"JXL_BAD": "1", "RS_JXL_PROFILE": "trace"}):
            report = self.run_benchmark(["--repeats", "2", "--warmups", "1", "--threads", "1", "2"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["rows"]), 24)
        self.assertEqual(len(report["summaries"]), 4)
        self.assertEqual([row["encoder"] for row in report["rows"][:4]], ["rust", "cpp", "cpp", "rust"])
        for row in report["rows"]:
            self.assertEqual(row["bpp"], row["bytes"] * 8 / 2)
            self.assertEqual(row["mpix"], 2 / 1e6)
            self.assertEqual(len(row["decodes"]), 2)
            self.assertTrue(all(result["pixels_match_including_hidden_rgb"] for result in row["decodes"]))
        for item in report["summaries"]:
            self.assertEqual(item["cpp_over_rust_wall_speedup"], 0.5)
            self.assertEqual(item["rust"]["median_wall_seconds"], 2)
            self.assertEqual(item["rust_over_cpp_bytes"], 1)
        for _, environment in self.calls:
            self.assertNotIn("JXL_BAD", environment)
            self.assertNotIn("RS_JXL_PROFILE", environment)

    def test_both_encoders_matching_wrong_pixels_still_fail_against_source(self):
        self.mismatch = True
        report = self.run_benchmark()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(all("hidden RGB" in row["error"] for row in report["rows"]))
        self.assertTrue(all("cpp_over_rust_wall_speedup" not in row for row in report["summaries"]))

    def test_cleanup_on_encode_and_decode_timeout(self):
        for binary in ("rust.exe", "cpp.exe", "djxl.exe"):
            with self.subTest(binary=binary):
                self.failure = binary
                report = self.run_benchmark()
                self.assertEqual(report["status"], "failed")

    def test_reject_jpeg_disguised_as_png_before_any_encode(self):
        self.image.write_bytes(b"\xff\xd8\xffJPEG")
        args = port.parser().parse_args(self.argv)
        with patch.object(port, "run_process", side_effect=self.fake_process), \
                patch.object(port, "cpp_provenance", return_value={"status": "unverified"}), \
                redirect_stdout(StringIO()):
            report = port.benchmark(args)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["rows"], [])
        self.assertTrue(all("--version" in command for command, _ in self.calls))
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_new_json_and_md_reports_on_failure(self):
        self.mismatch = True
        json_path, md_path = self.root / "report.json", self.root / "report.md"
        with patch.object(port, "run_process", side_effect=self.fake_process), \
                patch.object(port, "cpp_provenance", return_value={"status": "unverified"}), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = port.main([*self.argv, "--json", str(json_path), "--md", str(md_path)])
        self.assertEqual(code, 1)
        report = json.loads(json_path.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("unverified", md_path.read_text())
        self.assertTrue(report["temporary_outputs_cleaned"])
        args = port.parser().parse_args([*self.argv, "--json", str(json_path)])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            port.validate_args(args)

    def test_validation_rejects_bad_limits(self):
        for extra in (["--threads", "0"], ["--threads", "257"], ["--efforts", "11"],
                      ["--repeats", "0"], ["--warmups", "-1"], ["--timeout", "nan"],
                      ["--timeout", "inf"], ["--timeout", "0"], ["--threads", "1", "1"],
                      ["--source-sample-limit", "0"]):
            args = port.parser().parse_args([*extra, *self.argv])
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                port.validate_args(args)

    def test_explicit_cpp_and_mode_are_required(self):
        for option in ("--cpp", "--mode", "--official-djxl"):
            argv = list(self.argv)
            while option in argv:
                index = argv.index(option)
                del argv[index:index + 2]
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                port.parser().parse_args(argv)

    def test_incomplete_pair_does_not_crash_summary_after_interrupt(self):
        rows = [dict(input_index=0, threads=1, effort=9, mode="single-candidate", encoder="rust",
                     warmup=False, repeat=0, status="passed", wall_seconds=1, bytes=12, bpp=48, mpix=0.000002)]
        summary = port.summarize(rows)
        self.assertEqual(summary[0]["status"], "failed")

    def test_interrupt_after_one_encoder_cleans_up_and_retains_partial_results(self):
        def interrupted(command, environment, timeout):
            if Path(command[0]).name == "cpp.exe" and "--version" not in command:
                raise KeyboardInterrupt
            return self.fake_process(command, environment, timeout)

        args = port.parser().parse_args(self.argv)
        with patch.object(port, "run_process", side_effect=interrupted), \
                patch.object(port, "cpp_provenance", return_value={"status": "unverified"}), \
                redirect_stdout(StringIO()):
            report = port.benchmark(args)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["errors"], ["Interrupted"])
        self.assertEqual(len(report["rows"]), 1)
        self.assertEqual(report["summaries"][0]["status"], "failed")
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_changed_binary_invalidates_the_run(self):
        def changed(command, environment, timeout):
            result = self.fake_process(command, environment, timeout)
            if Path(command[0]).name == "cpp.exe" and "--version" not in command:
                (self.root / "cpp.exe").write_bytes(b"changed")
            return result

        args = port.parser().parse_args(self.argv)
        with patch.object(port, "run_process", side_effect=changed), \
                patch.object(port, "cpp_provenance", return_value={"status": "unverified"}), \
                redirect_stdout(StringIO()):
            report = port.benchmark(args)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["binaries"]["cpp"]["unchanged_during_run"])
        self.assertIn("measurements are invalid", report["errors"][0])

    def test_no_reports_by_default_but_environment_and_unverified_are_visible(self):
        before = set(self.root.iterdir())
        output = StringIO()
        with patch.object(port, "run_process", side_effect=self.fake_process), \
                patch.object(port, "cpp_provenance", return_value={"status": "unverified"}), \
                redirect_stdout(output):
            self.assertEqual(port.main(self.argv), 0)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertIn('"JXL_LOSSLESS_SEARCH_TRACE": "1"', output.getvalue())
        self.assertIn("provenance=unverified", output.getvalue())


if __name__ == "__main__":
    unittest.main()
