import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_chain_checkpoints as bench


class ChainCheckpointTests(unittest.TestCase):
    def argv(self, root):
        return ["--before", str(root / "before"), "--after", str(root / "after"),
                "--official-djxl", str(root / "djxl"), "--case", str(root / "input.png"),
                "--profile", "--out", str(root / "report.json")]

    def profile(self):
        return (
            "[rs-jxl profile] lz77.source.route: tokens=65536 method=Optc256 "
            "mode=SymbolId indexed=false bucket=false chain_batch=true\n"
            "[rs-jxl profile] lz77.source.chain_batch: tokens=65536 workers=4 lanes=1 adaptive=true\n"
            "[rs-jxl profile] lz77.source.chain_batch_work: tokens=65536 computed=16384 "
            "consumed=4096 overflow_requeries=3 demand_queries=0 unused=12288 "
            "demand_only=false adaptive=true lanes=1 interleaved_positions=0\n"
            "[rs-jxl profile] lz77.source.chain_schedule: cooperative=true roots=2 "
            "streams=3 eligible=2 streams_enabled=true\n")

    def test_commands_differ_only_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family in ("automatic", "literal", "cross6"):
                args = bench.arguments(self.argv(root) + ["--family", family])
                before, after = [bench.command(args, "source", "output", v) for v in bench.VARIANTS]
                self.assertEqual(before[0], str(args.before))
                self.assertEqual(after[0], str(args.after))
                self.assertEqual(before[1:], after[1:])
                self.assertEqual(before, [str(args.before), "source", "--effort", "9",
                                         "--threads", "8", "--family", family,
                                         "--output", "output", "--chain-batch-parallel"])
                self.assertEqual(args.expect, "active")
            with self.assertRaises(ValueError):
                bench.command(args, "source", "output", "ordinary")

    def test_bounds_required_profile_and_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in (["--threads", "0"], ["--threads", "257"], ["--repeats", "0"],
                          ["--repeats", "11"], ["--warmups", "4"], ["--warmups", "-1"],
                          ["--timeout", "nan"], ["--timeout", "0"], ["--expect", "flag"],
                          ["--threads", "1"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    bench.arguments(self.argv(root) + extra)
            for missing, count in (("--profile", 1), ("--case", 2), ("--before", 2), ("--after", 2)):
                argv = self.argv(root)
                start = argv.index(missing)
                del argv[start:start + count]
                with self.subTest(missing=missing), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    bench.arguments(argv)
            sources = [root / "input.png", root / "other.png"]
            for source in sources:
                source.touch()
            args = bench.arguments(self.argv(root) + ["--case", str(sources[1])])
            self.assertEqual(bench.shared.load_cases(args), [p.resolve() for p in sources])
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(cases=[dict(original=p.name) for p in sources])),
                                encoding="utf-8")
            argv = self.argv(root)
            start = argv.index("--case")
            argv[start:start + 2] = ["--manifest", str(manifest)]
            self.assertEqual(bench.shared.load_cases(bench.arguments(argv)), [p.resolve() for p in sources])
            args.out.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                bench.benchmark(args)
            self.assertEqual(args.out.read_text(encoding="utf-8"), "keep")

    def run_mock(self, root, *, failure=None, expect="active", threads=8, active=True):
        for name in ("before", "after", "djxl", "input.png"):
            (root / name).write_bytes(name.encode("ascii"))
        args = bench.arguments(self.argv(root) + ["--warmups", "1", "--repeats", "2",
                                                 "--expect", expect, "--threads", str(threads)])
        expected = SimpleNamespace(width=256, height=256, depth=16)
        decoded = SimpleNamespace(width=256, height=256, depth=16)
        calls, artifacts, counts = [], [], dict.fromkeys(bench.VARIANTS, 0)
        digests = {name: bench.shared.port.file_sha256(root / name) for name in bench.VARIANTS}
        original_open, original_dump = Path.open, json.dump
        original_hash = bench.shared.port.file_sha256
        report_modes, publications = [], []

        def open_file(path, mode="r", *positional, **keywords):
            if path == args.out and any(flag in mode for flag in "wax+"):
                report_modes.append(mode)
            return original_open(path, mode, *positional, **keywords)

        def publish(report, output, **keywords):
            self.assertIn("Prebuilt before/after", report["scope"])
            for name in bench.VARIANTS:
                self.assertIn(name, report)
                self.assertIn("verified_unchanged", report[name])
            publications.append(report["status"])
            return original_dump(report, output, **keywords)

        def hash_file(path):
            if (failure == "final-hash" and path == root / "after"
                    and any(call.args[0] == path for call in hashes.call_args_list[:-1])):
                raise RuntimeError("injected final hash failure")
            return original_hash(path)

        def process(command, env, timeout):
            self.assertEqual(timeout, 180)
            variant = next((v for v in bench.VARIANTS if command[0] == str(root / v)), None)
            if variant:
                if not calls:
                    for name in bench.VARIANTS:
                        self.assertTrue(any(call.args[0] == root / name for call in hashes.call_args_list))
                self.assertNotIn("JXL_INHERITED", env)
                self.assertNotIn("RS_JXL_OTHER", env)
                self.assertEqual(Path(command[1]).read_bytes(), b"input.png")
                self.assertEqual(command.count("--chain-batch-parallel"), 1)
                profiled = env.get("RS_JXL_PROFILE") == "1"
                calls.append((variant, profiled))
                if failure == "encoder" and variant == "after":
                    return dict(returncode=None, error="timeout", stdout="", stderr="partial")
                output = Path(command[command.index("--output") + 1])
                artifacts.append(output)
                size = 100 if variant == "before" else 125
                output.write_bytes(bytes([len(calls)]) * size)
                if not profiled:
                    counts[variant] += 1
                seconds = 999 if profiled else (77 if counts[variant] == 1 else
                                                (1 if variant == "before" else 2))
                measurement = dict(seconds=seconds, bytes=size, bpp=size * 8 / 65536,
                                   mpix_per_second=65536 / 1e6 / seconds,
                                   width=256, height=256, search=False)
                if failure == "metric" and variant == "after":
                    measurement["bpp"] = 100
                stderr = self.profile() if active else ""
                if failure == "route" and variant == "after":
                    stderr = self.profile().splitlines()[0]
                if variant == "after" and failure in ("mutate-before", "mutate-after"):
                    (root / failure.removeprefix("mutate-")).write_bytes(b"changed")
                return dict(returncode=0, stdout=json.dumps(measurement),
                            stderr=stderr if profiled else "", command=command, wall_seconds=5000)
            self.assertEqual(command[0], str(root / "djxl"))
            self.assertIn("--bits_per_sample=16", command)
            self.assertIn("--num_threads=1", command)
            if failure == "decoder":
                return dict(returncode=1, stdout="", stderr="oracle failed")
            Path(command[2]).write_bytes(b"decoded")
            return dict(returncode=0, stdout="", stderr="", command=command)

        with patch.object(bench.shared.port, "run_process", side_effect=process), \
                patch.object(bench.shared.port, "png_samples", side_effect=lambda data:
                             decoded if data == b"decoded" else expected), \
                patch.object(bench.shared.port, "compare_pixels",
                             side_effect=ValueError("invisible RGB mismatch") if failure == "pixels" else None) as compare, \
                patch.object(bench.shared.port, "file_sha256", side_effect=hash_file) as hashes, \
                patch.object(Path, "open", autospec=True, side_effect=open_file), \
                patch.object(bench.shared.json, "dump", side_effect=publish), \
                patch.dict(os.environ, {"RS_JXL_PROFILE": "trace", "RS_JXL_OTHER": "1", "JXL_INHERITED": "1"}), \
                contextlib.redirect_stdout(io.StringIO()):
            report = bench.benchmark(args)
        for call in compare.call_args_list:
            self.assertIs(call.args[0], expected)
            self.assertIs(call.args[1], decoded)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual(report_modes, ["x"])
        self.assertEqual(publications, [report["status"]])
        self.assertTrue(all(not p.exists() for p in artifacts))
        self.assertEqual(json.loads(args.out.read_text(encoding="utf-8")),
                         json.loads(json.dumps(report, default=str)))
        for name in bench.VARIANTS:
            self.assertEqual(report[name]["path"], str((root / name).resolve()))
            self.assertEqual(report[name]["sha256"], digests[name])
            self.assertGreaterEqual(sum(call.args[0] == root / name for call in hashes.call_args_list), 2)
        return report, calls, compare.call_count

    def test_alternating_native_trials_regressions_and_both_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checks = self.run_mock(Path(directory))
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(calls, [("before", True), ("after", True),
                                 ("after", False), ("before", False),
                                 ("before", False), ("after", False),
                                 ("after", False), ("before", False)])
        self.assertEqual(checks, 8)
        self.assertNotEqual(report["before"]["sha256"], report["after"]["sha256"])
        case = report["cases"][0]
        for name in bench.VARIANTS:
            self.assertTrue(report[name]["verified_unchanged"])
            self.assertEqual(report[name]["sha256_after"], report[name]["sha256"])
            self.assertEqual(len(case["variants"][name]), 3)
            self.assertEqual(case["profiles"][name]["measurement"]["seconds"], 999)
            self.assertFalse(case["profiles"][name]["included_in_timings"])
            self.assertTrue(case["profiles"][name]["route"]["active"])
        summary = case["summaries"]["after"]
        self.assertEqual(case["summaries"]["before"]["seconds"], 1)
        self.assertEqual(summary["seconds"], 2)
        self.assertEqual(summary["ratios_to_before"]["seconds"], 2)
        self.assertEqual(summary["ratios_to_before"]["bytes"], 1.25)
        self.assertEqual(summary["bpp"], 125 * 8 / 65536)
        self.assertEqual(summary["mpix_per_second"], 65536 / 1e6 / 2)

    def test_mutation_of_either_checkpoint_fails_persisted_report(self):
        for name in bench.VARIANTS:
            with self.subTest(checkpoint=name), tempfile.TemporaryDirectory() as directory:
                report, _, _ = self.run_mock(Path(directory), failure=f"mutate-{name}")
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report[name]["verified_unchanged"])
                self.assertNotEqual(report[name]["sha256"], report[name]["sha256_after"])
                self.assertTrue(any("changed during benchmark" in error for error in report["errors"]))
                self.assertTrue(report["cases"][0]["summaries"])

    def test_failures_keep_receipts_and_cleanup(self):
        for failure in ("encoder", "decoder", "pixels", "metric", "route"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, _ = self.run_mock(Path(directory), failure=failure)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                case = report["cases"][0]
                self.assertEqual(case["summaries"], {})
                self.assertTrue(all(profiled for _, profiled in calls))
                self.assertTrue(case["profiles"])
                if failure == "route":
                    self.assertIn("notes", case["profiles"]["after"])
                    self.assertIn("stderr", case["profiles"]["after"]["encoder"])

    def test_final_hash_exception_is_published_as_failed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _ = self.run_mock(Path(directory), failure="final-hash")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["after"]["verified_unchanged"])
        self.assertTrue(any("injected final hash failure" in error for error in report["errors"]))
        self.assertTrue(report["cases"][0]["summaries"])

    def test_only_explicit_controls_accept_inactive_profiles(self):
        for expect, threads, status in (("active", 8, "failed"), ("inactive", 1, "passed"),
                                        ("either", 8, "passed")):
            with self.subTest(expect=expect), tempfile.TemporaryDirectory() as directory:
                report, _, _ = self.run_mock(Path(directory), expect=expect, threads=threads, active=False)
                self.assertEqual(report["status"], status, report["errors"])
        with tempfile.TemporaryDirectory() as directory:
            report, _, _ = self.run_mock(Path(directory), expect="inactive")
            self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
