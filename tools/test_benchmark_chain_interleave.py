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

import benchmark_chain_interleave as bench


class ChainInterleaveTests(unittest.TestCase):
    def argv(self, root):
        return ["--encoder", str(root / "encoder"), "--official-djxl", str(root / "djxl"),
                "--case", str(root / "input.png"), "--profile", "--out", str(root / "report.json")]

    def measurement(self, seconds=0.5, size=100):
        return dict(seconds=seconds, bytes=size, bpp=8 * size / 65536,
                    mpix_per_second=65536 / 1e6 / seconds, width=256, height=256, search=False)

    def profile(self, variant):
        batch = variant != "baseline"
        lanes = bench.lanes_for(variant)
        result = ("[rs-jxl profile] lz77.source.route: tokens=65536 method=Optc256 "
                  "mode=SymbolId indexed=false bucket=false chain_limit=None seed_tokens=50000 "
                  f"chain_batch={str(batch).lower()}\n")
        if batch:
            result += ("[rs-jxl profile] lz77.source.chain_batch: tokens=65536 workers=2 "
                       f"refined=true scratch_bound=12345 batch=16384 lanes={lanes}\n"
                       "[rs-jxl profile] lz77.source.chain_batch_work: tokens=65536 computed=16384 "
                       f"consumed=4096 overflow_requeries=3 demand_only=true lanes={lanes} "
                       f"interleaved_positions={16384 if lanes > 1 else 0}\n")
        return result + "lz77.source.chain_batch_fill | 1 | 0.012345 | 0.012345\n"

    def test_arguments_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)))
            self.assertEqual(args.variants, list(bench.VARIANTS))
            self.assertEqual(args.family, "automatic")
            for variant in bench.VARIANTS:
                command = bench.command(args, "original.png", "output.jxl", variant)
                self.assertEqual(command[:2], [str(args.encoder), "original.png"])
                self.assertEqual(command[command.index("--family") + 1], "automatic")
                for flag in ("--search", "--lz", "--matcher", "--match-extension", "--model-costs"):
                    self.assertNotIn(flag, command)
                self.assertEqual("--chain-batch" in command, variant != "baseline")
                self.assertEqual("--chain-batch-lanes" in command, variant != "baseline")
                self.assertEqual("--chain-batch-legacy-schedule" in command, variant != "baseline")
                self.assertEqual("--chain-batch-fixed-window" in command, variant != "baseline")
                if variant != "baseline":
                    self.assertEqual(command[command.index("--chain-batch-lanes") + 1], variant[-1])
            with self.assertRaises(ValueError):
                bench.command(args, "a", "b", "batch3")
            for family in ("literal", "cross6"):
                args = bench.arguments(self.argv(Path(directory)) + ["--family", family])
                for variant in bench.VARIANTS:
                    command = bench.command(args, "a", "b", variant)
                    self.assertEqual(command[command.index("--family") + 1], family)

    def test_argument_rejections_and_exclusive_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in (["--threads", "0"], ["--timeout", "nan"], ["--timeout", "inf"],
                          ["--timeout", "0"], ["--repeats", "0"], ["--repeats", "11"],
                          ["--warmups", "-1"], ["--effort", "8"], ["--variants", "batch4"],
                          ["--family", "unknown"], ["--variants", "baseline", "batch4"],
                          ["--variants", "baseline", "batch1"],
                          ["--variants", "baseline"], ["--variants", "baseline", "baseline"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    bench.arguments(self.argv(root) + extra)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments([item for item in self.argv(root) if item != "--profile"])
            args = bench.arguments(self.argv(root))
            args.out.write_text("keep this report", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                bench.benchmark(args)
            self.assertEqual(args.out.read_text(encoding="utf-8"), "keep this report")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv(root))

    def test_actual_routes_for_each_mode_and_not_just_timer_rows(self):
        for variant in bench.VARIANTS:
            notes = bench.profile_notes(self.profile(variant))
            route = bench.validate_route(notes, variant)
            self.assertTrue(route["verified"])
            self.assertEqual(route["lanes"], bench.lanes_for(variant))
            self.assertEqual(route["interleaved_positions"] > 0, variant in ("batch2", "batch4", "batch8"))
        notes = bench.profile_notes("lz77.source.chain_batch_fill | 1 | 0.5 | 0.5\n")
        with self.assertRaisesRegex(ValueError, "No ordinary"):
            bench.validate_route(notes, "batch4")
        with self.assertRaisesRegex(ValueError, "did no observed work"):
            bench.validate_route(bench.profile_notes(self.profile("baseline")), "batch4")
        with self.assertRaises(ValueError):
            bench.validate_route(bench.profile_notes(self.profile("batch1")), "baseline")

    def test_route_rejects_wrong_lanes_missing_counters_and_zero_work(self):
        valid = bench.profile_notes(self.profile("batch4"))
        for name, key, value in (
                (bench.NOTE_NAMES[0], "bucket", "true"),
                (bench.NOTE_NAMES[0], "mode", "Frontier"),
                (bench.NOTE_NAMES[0], "chain_batch", "false"),
                (bench.NOTE_NAMES[1], "lanes", "2"),
                (bench.NOTE_NAMES[2], "lanes", "8"),
                (bench.NOTE_NAMES[2], "tokens", "65537"),
                (bench.NOTE_NAMES[2], "computed", "0"),
                (bench.NOTE_NAMES[2], "computed", "65537"),
                (bench.NOTE_NAMES[2], "consumed", "0"),
                (bench.NOTE_NAMES[2], "consumed", "65537"),
                (bench.NOTE_NAMES[2], "overflow_requeries", "4097"),
                (bench.NOTE_NAMES[2], "overflow_requeries", "-1"),
                (bench.NOTE_NAMES[2], "overflow_requeries", None),
                (bench.NOTE_NAMES[2], "demand_only", "True"),
                (bench.NOTE_NAMES[2], "demand_only", "1"),
                (bench.NOTE_NAMES[2], "demand_only", None),
                (bench.NOTE_NAMES[2], "interleaved_positions", "0"),
                (bench.NOTE_NAMES[2], "interleaved_positions", "16383"),
                (bench.NOTE_NAMES[2], "interleaved_positions", "16385"),
                (bench.NOTE_NAMES[2], "interleaved_positions", "-1"),
                (bench.NOTE_NAMES[2], "interleaved_positions", "true"),
                (bench.NOTE_NAMES[2], "interleaved_positions", None)):
            notes = copy.deepcopy(valid)
            if value is None:
                del notes[name][0][key]
            else:
                notes[name][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate_route(notes, "batch4")
        notes = bench.profile_notes(self.profile("batch1"))
        notes[bench.NOTE_NAMES[2]][0]["interleaved_positions"] = "1"
        with self.assertRaises(ValueError):
            bench.validate_route(notes, "batch1")

    def test_profile_multistream_order_and_malformed_notes(self):
        notes = bench.profile_notes(self.profile("batch8") * 2)
        notes[bench.NOTE_NAMES[2]][1]["demand_only"] = "false"
        route = bench.validate_route(notes, "batch8")
        self.assertEqual(route["batch_streams"], 2)
        self.assertEqual(route["computed"], 32768)
        self.assertEqual(route["consumed"], 8192)
        self.assertEqual(route["overflow_requeries"], 6)
        self.assertEqual(route["demand_only_streams"], 1)
        notes[bench.NOTE_NAMES[2]].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            bench.validate_route(notes, "batch8")
        for suffix in ("lanes=4 lanes=8", "tokens=", "not_a_field"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                bench.profile_notes(f"[rs-jxl profile] lz77.source.chain_batch: {suffix}")

    def test_work_counter_bounds_are_checked_per_stream(self):
        for variant in bench.VARIANTS[1:]:
            notes = bench.profile_notes(self.profile(variant))
            row = notes[bench.NOTE_NAMES[2]][0]
            row.update(computed="65536", consumed="65536", overflow_requeries="65536",
                       interleaved_positions="0" if variant == "batch1" else "65536",
                       demand_only="false")
            route = bench.validate_route(notes, variant)
            self.assertEqual(route["computed"], 65536)
            self.assertEqual(route["consumed"], 65536)
            self.assertEqual(route["overflow_requeries"], 65536)
            self.assertEqual(route["demand_only_streams"], 0)
            notes = bench.profile_notes(self.profile(variant) * 2)
            rows = notes[bench.NOTE_NAMES[2]]
            rows[0].update(computed="65537", interleaved_positions=(
                "0" if variant == "batch1" else "65537"))
            rows[1].update(computed="1", interleaved_positions="0" if variant == "batch1" else "1")
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                bench.validate_route(notes, variant)

    def test_native_measurements_do_not_require_unrelated_experiment_receipts(self):
        expected = SimpleNamespace(width=256, height=256)
        bench.validate_measurement(self.measurement(), 100, expected)
        for key, value in (("seconds", True), ("seconds", 0), ("seconds", float("nan")),
                           ("bytes", 100.0), ("bytes", 99), ("search", 0), ("search", True),
                           ("bpp", 7), ("mpix_per_second", 1), ("width", 1)):
            data = self.measurement()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate_measurement(data, 100, expected)

    def run_mock(self, root, failure=None):
        for name in ("encoder", "djxl", "input.png"):
            (root / name).write_bytes(b"original")
        args = bench.arguments(self.argv(root) + ["--repeats", "2", "--warmups", "1"])
        calls, artifacts = [], []
        count = {variant: 0 for variant in bench.VARIANTS}

        def process(command, env, timeout):
            self.assertEqual(timeout, 180)
            if command[0] == str(args.encoder):
                self.assertEqual(Path(command[1]).read_bytes(), b"original")
                variant = ("batch" + command[command.index("--chain-batch-lanes") + 1]
                           if "--chain-batch" in command else "baseline")
                profiled = env.get("RS_JXL_PROFILE") == "1"
                calls.append((variant, profiled))
                self.assertNotIn("JXL_INHERITED", env)
                self.assertNotIn("RS_JXL_OTHER", env)
                if failure == "encoder":
                    return dict(returncode=None, error="Process timeout after 180s", stdout="", stderr="partial")
                path = Path(command[command.index("--output") + 1])
                artifacts.append(path)
                size = 100 if variant == "baseline" else 125
                path.write_bytes(b"x" * size)
                if not profiled:
                    count[variant] += 1
                seconds = 999 if profiled else (77 if count[variant] == 1 else
                                                (1 if variant == "baseline" else 2))
                measurement = self.measurement(seconds, size)
                if failure == "metric":
                    measurement["seconds"] = 0
                stderr = self.profile(variant) if profiled else ""
                if failure == "route" and profiled and variant == "batch4":
                    stderr = self.profile("baseline")
                if failure == "identity" and profiled:
                    args.encoder.write_bytes(b"changed binary")
                return dict(returncode=0, stdout=json.dumps(measurement), stderr=stderr,
                            command=command, wall_seconds=2000)
            self.assertIn("--bits_per_sample=16", command)
            self.assertIn("--num_threads=1", command)
            if failure == "decoder":
                return dict(returncode=1, stdout="", stderr="djxl failed")
            Path(command[2]).write_bytes(b"decoded")
            return dict(returncode=0, stdout="", stderr="", command=command)

        with patch.object(bench.port, "run_process", side_effect=process), \
                patch.object(bench.port, "png_samples", return_value=SimpleNamespace(
                    width=256, height=256, depth=16)), \
                patch.object(bench.port, "compare_pixels",
                             side_effect=ValueError("hidden RGB mismatch") if failure == "pixels" else None) as compare, \
                patch.dict(os.environ, {"RS_JXL_PROFILE": "trace", "JXL_INHERITED": "1", "RS_JXL_OTHER": "x"}), \
                contextlib.redirect_stdout(io.StringIO()):
            report = bench.benchmark(args)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertTrue(all(not artifact.exists() for artifact in artifacts))
        self.assertEqual(json.loads(args.out.read_text(encoding="utf-8"))["status"], report["status"])
        return report, calls, compare.call_count

    def test_rotation_separate_profiles_native_medians_and_regressions_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checks = self.run_mock(Path(directory))
        self.assertEqual(report["status"], "passed", report["errors"])
        expected_order = [(v, True) for v in bench.VARIANTS]
        expected_order += [(v, False) for repeat in (-1, 0, 1)
                           for v in bench.trial_order(list(bench.VARIANTS), repeat, 0)]
        self.assertEqual(calls, expected_order)
        self.assertEqual(checks, 20)
        case = report["cases"][0]
        self.assertEqual(case["summaries"]["baseline"]["seconds"], 1)
        for variant in bench.VARIANTS[1:]:
            summary = case["summaries"][variant]
            self.assertEqual(summary["seconds"], 2)
            self.assertEqual(summary["ratios_to_baseline"]["seconds"], 2)
            self.assertEqual(summary["ratios_to_baseline"]["bytes"], 1.25)
            self.assertEqual(summary["ratios_to_batch1"]["seconds"], 1)
            diagnostic = case["profiles"][variant]
            self.assertFalse(diagnostic["included_in_timings"])
            self.assertEqual(diagnostic["measurement"]["seconds"], 999)
            self.assertTrue(diagnostic["route"]["verified"])
            self.assertEqual(diagnostic["route"]["computed"], 16384)
            self.assertEqual(diagnostic["route"]["consumed"], 4096)
            self.assertEqual(diagnostic["route"]["overflow_requeries"], 3)
            self.assertEqual(diagnostic["route"]["demand_only_streams"], 1)
            self.assertNotIn("computed", summary)
        self.assertEqual(report["environment"]["explicit"], {})
        self.assertEqual(case["summaries"]["baseline"]["ratios_to_batch1"]["seconds"], 0.5)
        self.assertEqual(report["profile_environment"]["explicit"], {"RS_JXL_PROFILE": "1"})

    def test_failures_have_receipts_and_cleanup(self):
        for failure in ("pixels", "metric", "encoder", "decoder", "route", "identity"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, checks = self.run_mock(Path(directory), failure)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                self.assertTrue(report["cases"][0]["variants"])
                if failure == "route":
                    self.assertEqual(report["cases"][0]["summaries"], {})
                    self.assertTrue(all(profiled for _, profiled in calls))
                    self.assertTrue(all(not samples for samples in report["cases"][0]["variants"].values()))
                    diagnostic = report["cases"][0]["profiles"]["batch4"]
                    self.assertIn("notes", diagnostic)
                    self.assertIn("stderr", diagnostic["encoder"])

    def test_explicit_large_manifest_has_no_case_cap_or_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("encoder", "djxl", "unselected.png"):
                (root / name).touch()
            sources = [root / f"{index}.png" for index in range(100)]
            for source in sources:
                source.touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(cases=[dict(original=p.name) for p in sources])),
                                encoding="utf-8")
            argv = self.argv(root)
            index = argv.index("--case")
            argv[index:index + 2] = ["--manifest", str(manifest)]
            args = bench.arguments(argv)
            self.assertEqual(bench.load_cases(args), [p.resolve() for p in sources])
            with patch.object(bench.port, "png_samples", side_effect=ValueError("reached sample validation")) as samples, \
                    patch.object(bench.port, "run_process") as process:
                report = bench.benchmark(args)
            self.assertEqual(report["status"], "failed")
            self.assertIn("reached sample validation", report["errors"][0])
            samples.assert_called_once_with(b"")
            process.assert_not_called()

    def test_extensible_profile_parser_and_invalid_ratio_reference(self):
        name = "lz77.source.chain_schedule"
        notes = bench.profile_notes(f"[rs-jxl profile] {name}: roots=2", (*bench.NOTE_NAMES, name))
        self.assertEqual(notes[name], [dict(roots="2")])
        self.assertNotIn(name, bench.profile_notes(f"[rs-jxl profile] {name}: roots=2"))
        with tempfile.TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)))
            with patch.object(bench.port, "run_process") as process:
                report = bench.benchmark(args, baseline="missing")
            self.assertEqual(report["status"], "failed")
            self.assertIn("Ratio references", report["errors"][0])
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
