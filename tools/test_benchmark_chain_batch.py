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

import benchmark_chain_batch as bench


class ChainBatchTests(unittest.TestCase):
    def argv(self, root):
        return ["--encoder", str(root / "encoder"), "--official-djxl", str(root / "djxl"),
                "--case", str(root / "input.png"), "--profile", "--out", str(root / "report.json")]

    def profile(self, variant, demand_only=False, cooperative=None):
        cfg = bench.configuration(variant)
        result = ("[rs-jxl profile] lz77.source.route: tokens=65536 method=Optc256 "
                  "mode=SymbolId indexed=false bucket=false "
                  f"chain_batch={str(cfg['batch']).lower()}\n")
        if not cfg["batch"]:
            return result
        adaptive = str(cfg["adaptive"]).lower()
        computed, demand, unused, overflow = ((0, 4096, 0, 0) if demand_only
                                             else (16384, 0, 12288, 3))
        cooperative = cfg["streams"] if cooperative is None else cooperative
        return result + (
            "[rs-jxl profile] lz77.source.chain_batch: tokens=65536 workers=1 "
            f"lanes=1 adaptive={adaptive}\n"
            "[rs-jxl profile] lz77.source.chain_batch_work: tokens=65536 "
            f"computed={computed} consumed=4096 overflow_requeries={overflow} "
            f"demand_queries={demand} unused={unused} demand_only={str(demand_only).lower()} "
            f"adaptive={adaptive} lanes=1 interleaved_positions=0\n"
            "[rs-jxl profile] lz77.source.chain_schedule: "
            f"cooperative={str(cooperative).lower()} roots=1 streams=3 eligible=2 "
            f"streams_enabled={str(cfg['streams']).lower()}\n")

    def test_fixed_five_commands_and_no_inherited_legacy_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family in ("automatic", "literal", "cross6"):
                args = bench.arguments(self.argv(root) + ["--family", family])
                self.assertEqual(args.variants, list(bench.VARIANTS))
                commands = []
                for variant in bench.VARIANTS:
                    cfg = bench.configuration(variant)
                    command = bench.command(args, "input", "output", variant)
                    commands.append(tuple(command))
                    self.assertEqual(command[command.index("--family") + 1], family)
                    self.assertEqual(command[command.index("--effort") + 1], "9")
                    self.assertEqual(command[command.index("--threads") + 1], "8")
                    self.assertEqual("--chain-batch" in command, cfg["batch"])
                    self.assertEqual("--chain-batch-legacy-schedule" in command,
                                     cfg["batch"] and not cfg["streams"])
                    self.assertEqual("--chain-batch-fixed-window" in command,
                                     cfg["batch"] and not cfg["adaptive"])
                    if cfg["batch"]:
                        self.assertEqual(command[command.index("--chain-batch-lanes") + 1], "1")
                    self.assertNotIn("--search", command)
                self.assertEqual(len(set(commands)), 5)
            with self.assertRaises(ValueError):
                bench.command(args, "input", "output", "batch8")

    def test_arguments_reject_unsafe_or_partial_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in (["--threads", "0"], ["--threads", "257"], ["--timeout", "nan"],
                          ["--timeout", "inf"], ["--timeout", "0"], ["--effort", "10"],
                          ["--repeats", "0"], ["--warmups", "-1"], ["--variants", "ordinary"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    bench.arguments(self.argv(root) + extra)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments([a for a in self.argv(root) if a != "--profile"])
            args = bench.arguments(self.argv(root))
            args.out.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                bench.benchmark(args)
            self.assertEqual(args.out.read_text(encoding="utf-8"), "preserve")

    def test_actual_configuration_all_modes_and_demand_only_activation(self):
        for variant in bench.VARIANTS:
            route = bench.validate_route(bench.profile_notes(self.profile(variant)), variant)
            self.assertTrue(route["verified"])
            self.assertEqual(route["adaptive"], bench.configuration(variant)["adaptive"])
        for variant in ("adaptive-only", "combined"):
            notes = bench.profile_notes(self.profile(variant, demand_only=True, cooperative=False))
            route = bench.validate_route(notes, variant)
            self.assertEqual(route["computed"], 0)
            self.assertEqual(route["demand_queries"], route["consumed"])
            self.assertEqual(route["cooperative_schedules"], 0)
        # Dense cases are equally valid: do not demand low utilization/adaptive fallback.
        notes = bench.profile_notes(self.profile("combined"))
        notes[bench.NOTE_NAMES[2]][0].update(computed="4096", unused="0")
        self.assertTrue(bench.validate_route(notes, "combined")["verified"])

    def test_wrong_activation_and_counter_receipts_are_rejected(self):
        original = bench.profile_notes(self.profile("combined"))
        for name, key, value in (
                (bench.NOTE_NAMES[0], "chain_batch", "false"),
                (bench.NOTE_NAMES[1], "lanes", "2"),
                (bench.NOTE_NAMES[1], "adaptive", "false"),
                (bench.NOTE_NAMES[2], "adaptive", None),
                (bench.NOTE_NAMES[2], "tokens", "65537"),
                (bench.NOTE_NAMES[2], "computed", "65537"),
                (bench.NOTE_NAMES[2], "computed", "0"),
                (bench.NOTE_NAMES[2], "consumed", "0"),
                (bench.NOTE_NAMES[2], "consumed", "65537"),
                (bench.NOTE_NAMES[2], "overflow_requeries", "4097"),
                (bench.NOTE_NAMES[2], "demand_queries", None),
                (bench.NOTE_NAMES[2], "demand_queries", "4097"),
                (bench.NOTE_NAMES[2], "unused", None),
                (bench.NOTE_NAMES[2], "unused", "12287"),
                (bench.NOTE_NAMES[2], "interleaved_positions", "1"),
                (bench.NOTE_NAMES[3], "streams_enabled", "false"),
                (bench.NOTE_NAMES[3], "streams_enabled", None),
                (bench.NOTE_NAMES[3], "cooperative", "1"),
                (bench.NOTE_NAMES[3], "roots", "0"),
                (bench.NOTE_NAMES[3], "eligible", "4"),
                (bench.NOTE_NAMES[3], "streams", "true")):
            notes = copy.deepcopy(original)
            if value is None:
                del notes[name][0][key]
            else:
                notes[name][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate_route(notes, "combined")
        for name in bench.NOTE_NAMES:
            notes = copy.deepcopy(original)
            notes[name] = []
            with self.subTest(missing=name), self.assertRaises(ValueError):
                bench.validate_route(notes, "combined")
        with self.assertRaises(ValueError):
            bench.validate_route(original, "ordinary")
        with self.assertRaises(ValueError):
            bench.profile_notes("[rs-jxl profile] lz77.source.chain_schedule: roots=1 roots=2")

    def test_multistream_accounting_and_incomplete_receipts(self):
        notes = bench.profile_notes(self.profile("combined") + self.profile("combined", demand_only=True))
        route = bench.validate_route(notes, "combined")
        self.assertEqual(route["batch_streams"], 2)
        self.assertEqual(route["tokens"], 131072)
        self.assertEqual(route["consumed"], 8192)
        self.assertEqual(route["demand_queries"], 4096)
        self.assertEqual(route["unused"], 12288)
        notes[bench.NOTE_NAMES[2]].pop()
        with self.assertRaises(ValueError):
            bench.validate_route(notes, "combined")

    def run_mock(self, root, failure=None):
        for name in ("encoder", "djxl", "input.png"):
            (root / name).write_bytes(b"original")
        args = bench.arguments(self.argv(root) + ["--repeats", "2", "--warmups", "1"])
        calls, counts = [], {v: 0 for v in bench.VARIANTS}

        def process(command, env, timeout):
            if command[0] == str(args.encoder):
                self.assertNotIn("RS_JXL_OTHER", env)
                profiled = env.get("RS_JXL_PROFILE") == "1"
                variant = next(v for v in bench.VARIANTS
                               if bench.command(args, command[1], command[command.index("--output") + 1], v)
                               == command)
                calls.append((variant, profiled))
                if failure == "encoder":
                    return dict(returncode=None, error="timeout", stdout="", stderr="partial")
                if not profiled:
                    counts[variant] += 1
                seconds = 999 if profiled else (77 if counts[variant] == 1 else
                                               (1 if variant == "ordinary" else 2))
                size = 100 if variant == "ordinary" else 125
                # Repeated codestream bytes deliberately differ: only source pixels count.
                Path(command[command.index("--output") + 1]).write_bytes(
                    bytes([counts[variant] + 1]) * size)
                receipt = dict(seconds=seconds, bytes=size, bpp=8 * size / 65536,
                               mpix_per_second=65536 / 1e6 / seconds,
                               width=256, height=256, search=False)
                if failure == "metric":
                    receipt["bpp"] = 1
                stderr = self.profile(variant, demand_only=variant in ("adaptive-only", "combined"))
                if failure == "route" and variant == "combined":
                    stderr = self.profile("ordinary")
                if failure == "identity":
                    args.encoder.write_bytes(b"changed")
                return dict(returncode=0, stdout=json.dumps(receipt), stderr=stderr if profiled else "",
                            command=command, wall_seconds=5000)
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
                patch.dict(os.environ, {"RS_JXL_PROFILE": "trace", "RS_JXL_OTHER": "1"}), \
                contextlib.redirect_stdout(io.StringIO()):
            report = bench.benchmark(args)
        self.assertEqual(json.loads(args.out.read_text(encoding="utf-8"))["status"], report["status"])
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual(set(report["cases"][0]["variants"]), set(bench.VARIANTS))
        return report, calls, compare.call_count

    def test_shared_callbacks_real_names_native_medians_and_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checks = self.run_mock(Path(directory))
        self.assertEqual(report["status"], "passed", report["errors"])
        expected = [(v, True) for v in bench.VARIANTS]
        expected += [(v, False) for repeat in (-1, 0, 1)
                     for v in bench.shared.trial_order(list(bench.VARIANTS), repeat, 0)]
        self.assertEqual(calls, expected)
        self.assertEqual(checks, 20)
        self.assertEqual(report["ratio_references"], dict(baseline="ordinary", comparison="legacy"))
        summaries = report["cases"][0]["summaries"]
        self.assertEqual(summaries["ordinary"]["seconds"], 1)
        for variant in bench.VARIANTS[1:]:
            summary = summaries[variant]
            self.assertEqual(summary["seconds"], 2)
            self.assertEqual(summary["ratios_to_ordinary"]["seconds"], 2)
            self.assertEqual(summary["ratios_to_ordinary"]["bytes"], 1.25)
            self.assertEqual(summary["ratios_to_legacy"]["seconds"], 1)
            self.assertNotIn("ratios_to_batch1", summary)

    def test_failures_keep_real_mode_receipts_and_cleanup(self):
        for failure in ("encoder", "decoder", "pixels", "metric", "route", "identity"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, _ = self.run_mock(Path(directory), failure)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])
                if failure == "route":
                    self.assertTrue(all(profiled for _, profiled in calls))
                    self.assertEqual(report["cases"][0]["summaries"], {})
                    self.assertIn("notes", report["cases"][0]["profiles"]["combined"])

    def test_manifest_has_no_arbitrary_case_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root / f"{i}.png" for i in range(100)]
            for source in sources:
                source.touch()
            manifest = root / "selected.json"
            manifest.write_text(json.dumps(dict(cases=[dict(original=p.name) for p in sources])),
                                encoding="utf-8")
            argv = self.argv(root)
            index = argv.index("--case")
            argv[index:index + 2] = ["--manifest", str(manifest)]
            self.assertEqual(bench.shared.load_cases(bench.arguments(argv)), [p.resolve() for p in sources])


if __name__ == "__main__":
    unittest.main()
