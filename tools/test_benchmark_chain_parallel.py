import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_chain_parallel as bench


class ChainParallelTests(unittest.TestCase):
    def argv(self, root):
        return ["--encoder", str(root / "encoder"), "--official-djxl", str(root / "djxl"),
                "--case", str(root / "hot-screenshot.png"), "--profile",
                "--out", str(root / "report.json")]

    def profile(self, active=False, workers=4, demand_only=False):
        result = ("[rs-jxl profile] lz77.source.route: tokens=65536 method=Optc256 "
                  "mode=SymbolId indexed=false bucket=false "
                  f"chain_batch={str(active).lower()}\n")
        if not active:
            return result
        computed, demand, unused, overflow = ((0, 4096, 0, 0) if demand_only
                                             else (16384, 0, 12288, 3))
        return result + (
            "[rs-jxl profile] lz77.source.chain_batch: tokens=65536 "
            f"workers={workers} lanes=1 adaptive=true\n"
            "[rs-jxl profile] lz77.source.chain_batch_work: tokens=65536 "
            f"computed={computed} consumed=4096 overflow_requeries={overflow} "
            f"demand_queries={demand} unused={unused} demand_only={str(demand_only).lower()} "
            "adaptive=true lanes=1 interleaved_positions=0\n"
            "[rs-jxl profile] lz77.source.chain_schedule: cooperative=true roots=2 "
            "streams=3 eligible=2 streams_enabled=true\n")

    def test_arguments_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family in ("automatic", "literal", "cross6"):
                for threads, expect in ((8, "active"), (1, "inactive"), (8, "either")):
                    args = bench.arguments(self.argv(root) + ["--family", family, "--threads",
                                                             str(threads), "--expect", expect])
                    self.assertEqual(args.variants, ["ordinary", "parallel"])
                    self.assertEqual(args.expect, expect)
                    for variant in bench.VARIANTS:
                        command = bench.command(args, "source", "output", variant)
                        self.assertEqual(command[:2], [str(args.encoder), "source"])
                        for flag, value in (("--effort", "9"), ("--family", family),
                                            ("--threads", str(threads)), ("--output", "output")):
                            self.assertEqual(command[command.index(flag) + 1], value)
                        self.assertEqual("--chain-batch-parallel" in command, variant == "parallel")
                        if variant == "parallel":
                            self.assertEqual(command, bench.command(args, "source", "output", "ordinary")
                                             + ["--chain-batch-parallel"])
                        for flag in ("--chain-batch-lanes", "--chain-batch", "--chain-batch-legacy-schedule",
                                     "--chain-batch-fixed-window", "--search"):
                            self.assertNotIn(flag, command)
            self.assertEqual(bench.arguments(self.argv(root)).expect, "either")
            with self.assertRaises(ValueError):
                bench.command(args, "source", "output", "combined")

    def test_argument_bounds_and_mandatory_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in (["--threads", "0"], ["--threads", "257"], ["--effort", "10"],
                          ["--timeout", "nan"], ["--timeout", "inf"], ["--timeout", "0"],
                          ["--repeats", "0"], ["--repeats", "11"], ["--warmups", "-1"],
                          ["--warmups", "4"], ["--expect", "flag"],
                          ["--threads", "1", "--expect", "active"], ["--variants", "parallel"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    bench.arguments(self.argv(root) + extra)
            for missing, count in (("--profile", 1), ("--case", 2)):
                argv = self.argv(root)
                index = argv.index(missing)
                del argv[index:index + count]
                with self.subTest(missing=missing), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    bench.arguments(argv)
            args = bench.arguments(self.argv(root))
            args.out.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                bench.benchmark(args)
            self.assertEqual(args.out.read_text(encoding="utf-8"), "keep")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv(root))

    def test_active_requires_real_receipts_and_combined_validator(self):
        for demand_only in (False, True):
            notes = bench.profile_notes(self.profile(True, demand_only=demand_only))
            with patch.object(bench.batch, "validate_route", wraps=bench.batch.validate_route) as check:
                result = bench.validate_route(notes, "parallel", expect="active")
            check.assert_called_once_with(notes, "combined")
            self.assertTrue(result["active"])
            self.assertEqual(result["disposition"], "active")
            self.assertEqual(result["workers"], [4])
            self.assertEqual(result["lanes"], 1)
            self.assertTrue(result["adaptive"])
            self.assertTrue(result["streams"])
            self.assertEqual(result["schedule_count"], 1)
            self.assertEqual(result["schedule_eligible"], 2)
            self.assertEqual(result["consumed"], 4096)
            self.assertEqual(result["computed"] == 0, demand_only)
            self.assertTrue(bench.validate_route(notes, "parallel", expect="either")["active"])
            for variant, threads, expect in (("ordinary", 8, "either"), ("parallel", 1, "either"),
                                             ("parallel", 8, "inactive")):
                with self.subTest(variant=variant, threads=threads, expect=expect), \
                        self.assertRaises(ValueError):
                    bench.validate_route(notes, variant, threads=threads, expect=expect)

    def test_bypasses_without_optc_or_any_source_receipts(self):
        for text in (self.profile(), "",
                     self.profile().replace("Optc256", "Rle"),
                     self.profile().replace("indexed=false", "indexed=true"),
                     self.profile().replace("mode=SymbolId", "mode=Frontier"),
                     self.profile().replace("tokens=65536", "tokens=0")):
            notes = bench.profile_notes(text)
            for variant in bench.VARIANTS:
                for threads in (1, 8):
                    for expect in ("inactive", "either"):
                        result = bench.validate_route(notes, variant, threads=threads, expect=expect)
                        self.assertTrue(result["verified"])
                        self.assertFalse(result["active"])
                        self.assertEqual(result["consumed"], 0)
                        self.assertEqual(result["schedule_count"], 0)
                        self.assertEqual(result["schedule_eligible"], 0)
                        self.assertEqual(result["disposition"],
                                         "ordinary" if variant == "ordinary" else "bypass")
            with self.assertRaises(ValueError):
                bench.validate_route(notes, "parallel", expect="active")
        notes = bench.profile_notes(self.profile())
        with patch.object(bench.batch, "validate_route", wraps=bench.batch.validate_route) as check:
            bench.validate_route(notes, "parallel", expect="inactive")
        self.assertEqual(check.call_args.args[1], "ordinary")
        with patch.object(bench.batch, "validate_route") as check:
            bench.validate_route(bench.profile_notes(""), "parallel", expect="inactive")
        check.assert_not_called()

    def test_schedule_intent_is_not_activity(self):
        notes = bench.profile_notes(self.profile())
        notes[bench.NOTE_NAMES[3]] = bench.profile_notes(self.profile(True))[bench.NOTE_NAMES[3]]
        for expect in ("inactive", "either"):
            result = bench.validate_route(notes, "parallel", threads=8, expect=expect)
            self.assertFalse(result["active"])
            self.assertEqual(result["schedule_count"], 1)
            self.assertEqual(result["schedule_eligible"], 2)
        with self.assertRaises(ValueError):
            bench.validate_route(notes, "parallel", threads=8, expect="active")
        with self.assertRaises(ValueError):
            bench.validate_route(notes, "ordinary")
        notes[bench.NOTE_NAMES[3]][0].update(eligible="0")
        result = bench.validate_route(notes, "parallel")
        self.assertFalse(result["active"])
        self.assertEqual(result["schedule_eligible"], 0)
        for key, value in (("streams_enabled", "false"), ("cooperative", "unknown"),
                           ("cooperative", "false"), ("roots", "0"),
                           ("streams", "1"), ("eligible", "4")):
            bad = copy.deepcopy(notes)
            bad[bench.NOTE_NAMES[3]][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate_route(bad, "parallel")

    def test_t1_rejects_all_schedules_without_batch_work(self):
        for text in ("", self.profile()):
            for cooperative in ("true", "false"):
                notes = bench.profile_notes(text)
                notes[bench.NOTE_NAMES[3]] = [dict(cooperative=cooperative, roots="1",
                                                  streams="3", eligible="2", streams_enabled="true")]
                for expect in ("inactive", "either"):
                    with self.subTest(routes=bool(text), cooperative=cooperative, expect=expect), \
                            self.assertRaisesRegex(ValueError, "scheduling at threads=1"):
                        bench.validate_route(notes, "parallel", threads=1, expect=expect)
                result = bench.validate_route(notes, "parallel", threads=8, expect="inactive")
                self.assertFalse(result["active"])
                self.assertEqual(result["schedule_count"], 1)
                self.assertEqual(result["schedule_eligible"], 2)
                notes[bench.NOTE_NAMES[3]].append(dict(notes[bench.NOTE_NAMES[3]][0], eligible="1"))
                result = bench.validate_route(notes, "parallel", threads=8, expect="either")
                self.assertFalse(result["active"])
                self.assertEqual(result["schedule_count"], 2)
                self.assertEqual(result["schedule_eligible"], 3)

    def test_incomplete_receipts_never_become_a_bypass(self):
        original = bench.profile_notes(self.profile(True))
        for name in bench.NOTE_NAMES:
            notes = copy.deepcopy(original)
            notes[name] = []
            for expect in bench.EXPECTATIONS:
                with self.subTest(missing=name, expect=expect), self.assertRaises(ValueError):
                    bench.validate_route(notes, "parallel", expect=expect)
        for name in bench.NOTE_NAMES[:3]:
            notes = bench.profile_notes("")
            notes[name] = original[name]
            for variant in bench.VARIANTS:
                with self.subTest(only=name, variant=variant), self.assertRaises(ValueError):
                    bench.validate_route(notes, variant)
        notes = bench.profile_notes(self.profile(True) + self.profile(True))
        notes[bench.NOTE_NAMES[2]].pop()
        with self.assertRaises(ValueError):
            bench.validate_route(notes, "parallel")

    def test_workers_configuration_and_accounting_rejections(self):
        original = bench.profile_notes(self.profile(True))
        for name, key, value in (
                (bench.NOTE_NAMES[0], "chain_batch", None),
                (bench.NOTE_NAMES[0], "chain_batch", "false"),
                (bench.NOTE_NAMES[0], "chain_batch", "True"),
                (bench.NOTE_NAMES[0], "method", "Rle"),
                (bench.NOTE_NAMES[1], "workers", None),
                (bench.NOTE_NAMES[1], "workers", "0"),
                (bench.NOTE_NAMES[1], "workers", "1"),
                (bench.NOTE_NAMES[1], "workers", "9"),
                (bench.NOTE_NAMES[1], "workers", "true"),
                (bench.NOTE_NAMES[1], "lanes", "2"),
                (bench.NOTE_NAMES[1], "adaptive", "false"),
                (bench.NOTE_NAMES[2], "adaptive", None),
                (bench.NOTE_NAMES[2], "lanes", "2"),
                (bench.NOTE_NAMES[2], "tokens", "65537"),
                (bench.NOTE_NAMES[2], "computed", "65537"),
                (bench.NOTE_NAMES[2], "computed", "0"),
                (bench.NOTE_NAMES[2], "consumed", "0"),
                (bench.NOTE_NAMES[2], "demand_queries", None),
                (bench.NOTE_NAMES[2], "unused", "12287"),
                (bench.NOTE_NAMES[2], "overflow_requeries", "4097"),
                (bench.NOTE_NAMES[2], "interleaved_positions", "1"),
                (bench.NOTE_NAMES[3], "streams_enabled", "false"),
                (bench.NOTE_NAMES[3], "roots", "0")):
            notes = copy.deepcopy(original)
            if value is None:
                del notes[name][0][key]
            else:
                notes[name][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate_route(notes, "parallel")
        notes = bench.profile_notes(self.profile(True) + self.profile(True, workers=1))
        with self.assertRaises(ValueError):
            bench.validate_route(notes, "parallel")
        notes = bench.profile_notes(self.profile(True) + self.profile())
        self.assertTrue(bench.validate_route(notes, "parallel")["active"])
        for suffix in ("workers=4 workers=2", "tokens=", "not_a_field"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                bench.profile_notes(f"[rs-jxl profile] lz77.source.chain_batch: {suffix}")
        with self.assertRaises(ValueError):
            bench.validate_route(bench.profile_notes(self.profile().replace("chain_batch=false", "")),
                                 "ordinary")

    def run_mock(self, root, *, threads=8, expect="active", mixed=False, failure=None):
        for name in ("encoder", "djxl", "hot-screenshot.png", "other.png", "small.png"):
            (root / name).write_bytes(b"small" if name == "small.png" else b"original")
        argv = self.argv(root) + ["--threads", str(threads), "--expect", expect,
                                 "--repeats", "2", "--warmups", "1"]
        if mixed:
            argv += ["--case", str(root / "other.png"), "--case", str(root / "small.png")]
        args = bench.arguments(argv)
        calls, artifacts, counts = [], [], {}

        def process(command, env, timeout):
            self.assertEqual(timeout, 180)
            if command[0] == str(args.encoder):
                self.assertNotIn("RS_JXL_OTHER", env)
                self.assertNotIn("JXL_INHERITED", env)
                variant = "parallel" if "--chain-batch-parallel" in command else "ordinary"
                profiled = env.get("RS_JXL_PROFILE") == "1"
                pixels = Path(command[1]).read_bytes()
                small = pixels == b"small"
                # Cases 1 and 2 deliberately bypass, including no source notes.
                case_index = len(calls) // 8
                calls.append((case_index, variant, profiled))
                if failure == "encoder":
                    return dict(returncode=None, error="timeout", stdout="", stderr="partial")
                output = Path(command[command.index("--output") + 1])
                artifacts.append(output)
                key = (case_index, variant)
                if not profiled:
                    counts[key] = counts.get(key, 0) + 1
                seconds = 999 if profiled else (77 if counts[key] == 1 else
                                               (1 if variant == "ordinary" else 2))
                size = 100 if variant == "ordinary" else 125
                output.write_bytes(bytes([len(calls)]) * size)
                measurement = dict(seconds=seconds, bytes=size, bpp=size * 8 / 65536,
                                   mpix_per_second=65536 / 1e6 / seconds,
                                   width=256, height=256, search=False)
                if failure == "metric":
                    measurement["bpp"] = 100
                active = variant == "parallel" and threads > 1 and case_index == 0
                stderr = "" if small else self.profile(active)
                if failure == "route" and variant == "parallel":
                    stderr = self.profile()
                if failure == "identity":
                    args.encoder.write_bytes(b"changed")
                return dict(returncode=0, stdout=json.dumps(measurement),
                            stderr=stderr if profiled else "", command=command, wall_seconds=5000)
            self.assertIn("--bits_per_sample=16", command)
            self.assertIn("--num_threads=1", command)
            if failure == "decoder":
                return dict(returncode=1, stdout="", stderr="oracle failed")
            Path(command[2]).write_bytes(b"decoded")
            return dict(returncode=0, stdout="", stderr="", command=command)

        with patch.object(bench.shared.port, "run_process", side_effect=process), \
                patch.object(bench.shared.port, "png_samples", return_value=SimpleNamespace(
                    width=256, height=256, depth=16)), \
                patch.object(bench.shared.port, "compare_pixels",
                             side_effect=ValueError("invisible RGB mismatch") if failure == "pixels" else None) as compare, \
                patch.dict(os.environ, {"RS_JXL_PROFILE": "trace", "RS_JXL_OTHER": "1", "JXL_INHERITED": "1"}), \
                contextlib.redirect_stdout(io.StringIO()):
            report = bench.benchmark(args)
        self.assertEqual(json.loads(args.out.read_text(encoding="utf-8"))["status"], report["status"])
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertTrue(all(not path.exists() for path in artifacts))
        return report, calls, compare.call_count

    def test_hot_screenshot_require_active_t1_inactive_and_mixed_t8_either(self):
        for threads, expect, mixed in ((8, "active", False), (1, "inactive", False), (8, "either", True)):
            with self.subTest(threads=threads, expect=expect), tempfile.TemporaryDirectory() as directory:
                report, calls, checks = self.run_mock(Path(directory), threads=threads, expect=expect, mixed=mixed)
            self.assertEqual(report["status"], "passed", report["errors"])
            case_count = 3 if mixed else 1
            self.assertEqual(checks, 8 * case_count)
            expected_calls = []
            for index in range(case_count):
                expected_calls += [(index, v, True)
                                   for v in bench.shared.trial_order(list(bench.VARIANTS), 0, index)]
                expected_calls += [(index, v, False) for repeat in (-1, 0, 1)
                                   for v in bench.shared.trial_order(list(bench.VARIANTS), repeat, index)]
            self.assertEqual(calls, expected_calls)
            self.assertEqual(report["ratio_references"], dict(baseline="ordinary", comparison="parallel"))
            for index, case in enumerate(report["cases"]):
                self.assertFalse(case["profiles"]["ordinary"]["route"]["active"])
                diagnostic = case["profiles"]["parallel"]
                self.assertEqual(diagnostic["route"]["active"], threads > 1 and index == 0)
                self.assertFalse(diagnostic["included_in_timings"])
                self.assertEqual(diagnostic["measurement"]["seconds"], 999)
                ordinary, parallel = (case["summaries"][v] for v in bench.VARIANTS)
                self.assertEqual(ordinary["seconds"], 1)
                self.assertEqual(parallel["seconds"], 2)
                self.assertEqual(parallel["ratios_to_ordinary"]["bytes"], 1.25)
                self.assertEqual(parallel["ratios_to_ordinary"]["seconds"], 2)
                for metric in ("bytes", "bpp", "mpix_per_second"):
                    self.assertGreater(parallel[metric], 0)
                self.assertNotIn("active", parallel)
            for name in ("encoder", "decoder"):
                self.assertEqual(len(report[name]["sha256"]), 64)
            self.assertEqual(report["environment"]["explicit"], {})
            self.assertEqual(report["profile_environment"]["explicit"], {"RS_JXL_PROFILE": "1"})

    def test_failures_preserve_receipts_and_clean_outputs(self):
        for failure in ("encoder", "decoder", "pixels", "metric", "route", "identity"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, _ = self.run_mock(Path(directory), failure=failure)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["errors"])
            if failure == "route":
                self.assertTrue(all(profiled for _, _, profiled in calls))
                case = report["cases"][0]
                self.assertEqual(case["summaries"], {})
                self.assertIn("notes", case["profiles"]["parallel"])
                self.assertTrue(all(not rows for rows in case["variants"].values()))


if __name__ == "__main__":
    unittest.main()
