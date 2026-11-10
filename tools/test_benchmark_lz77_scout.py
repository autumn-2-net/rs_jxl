import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import benchmark_lz77_scout as bench


def rgba_png(depth=16, hidden=1234):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    values = (hidden, 25, 77, 0, 100, 200, 250, (1 << depth) - 1)
    row = struct.pack(">8H", *values) if depth == 16 else bytes(values)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, depth, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0" + row)) + chunk(b"IEND", b""))


class ScoutBenchmarkTests(unittest.TestCase):
    def arguments(self, root, extra=()):
        return bench.arguments(["--encoder", str(root / "encoder.exe"),
                                "--decoder", str(root / "djxl.exe"),
                                "--case", str(root / "input.png"),
                                "--out", str(root / "report.json"), *extra])

    def measurement(self, size=100, seconds=0.25, mode="baseline"):
        scout, quick = None, None
        if mode in ("observe", "route"):
            scout = dict(mode=mode, calls=1, total_seconds=0.01, full_calls=1,
                         literal_calls=0, fast_calls=0, latest={"decision": "full"})
        elif mode != "baseline":
            quick = dict(mode=mode, calls=1, queries=1, query_misses=1,
                         useful_matches=0, skipped_positions=0,
                         cost_ranked_calls=int(mode == "double-fast-cost"), preserved_careful_calls=0)
        return dict(bytes=size, seconds=seconds, bpp=4 * size,
                    mpix_per_second=2 / 1e6 / seconds, width=2, height=1, search=False, lz=0,
                    lz_scout=scout, lz_quick=quick)

    def test_activation_receipts_reject_noop_and_wrong_modes(self):
        for mode in bench.MODES:
            with self.subTest(mode=mode):
                valid = self.measurement(mode=mode)
                bench.validate_activation(valid, mode)
                for key in ("lz_scout", "lz_quick"):
                    invalid = dict(valid)
                    del invalid[key]
                    with self.assertRaisesRegex(ValueError, "Missing"):
                        bench.validate_activation(invalid, mode)
                if mode == "baseline":
                    for key in ("lz_scout", "lz_quick"):
                        with self.assertRaises(ValueError):
                            bench.validate_activation(dict(valid, **{key: {}}), mode)
                elif mode in ("observe", "route"):
                    for receipt in (None, {}, dict(mode="other", calls=1),
                                    dict(mode=mode, calls=0), dict(mode=mode, calls=True),
                                    dict(mode=mode, calls=-1), dict(mode=mode, calls=1.5)):
                        with self.assertRaises(ValueError):
                            bench.validate_activation(dict(valid, lz_scout=receipt), mode)
                    with self.assertRaises(ValueError):
                        bench.validate_activation(dict(valid, lz_quick={"mode": "sparse", "calls": 1}), mode)
                else:
                    for key, value in (("mode", "other"), ("calls", 0), ("calls", True),
                                       ("queries", -1), ("query_misses", 0.5),
                                       ("useful_matches", None), ("skipped_positions", False)):
                        receipt = dict(valid["lz_quick"], **{key: value})
                        with self.assertRaises(ValueError):
                            bench.validate_activation(dict(valid, lz_quick=receipt), mode)
                    with self.assertRaises(ValueError):
                        bench.validate_activation(dict(valid, lz_scout={"mode": "observe", "calls": 1}), mode)

    def test_default_matrix_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            original_modes = ("baseline", "observe", "route", "sparse", "double-fast", "double-fast-sparse")
            self.assertEqual(bench.DEFAULT_MODES, original_modes)
            self.assertEqual(args.mode, list(original_modes))
            self.assertNotIn("double-fast-cost", args.mode)
            self.assertEqual((args.effort, args.threads, args.repeats), (9, 8, 3))
            for mode in bench.MODES:
                command = bench.command(args, "in.png", "out.jxl", mode)
                self.assertEqual(command[0], str(args.encoder))
                self.assertNotIn("--search", command)
                self.assertNotIn("--model-costs", command)
                self.assertEqual(command[command.index("--lz") + 1], "0")
                if mode == "baseline":
                    self.assertNotIn("--lz-scout", command)
                    self.assertNotIn("--lz-quick", command)
                else:
                    self.assertEqual(command[-2:], ["--lz-scout" if mode in ("observe", "route")
                                                    else "--lz-quick", mode])
            with self.assertRaises(ValueError):
                bench.command(args, "in.png", "out.jxl", "unknown")
            args.lz = 1
            for mode in bench.MODES:
                command = bench.command(args, "in.png", "out.jxl", mode)
                self.assertEqual(command[command.index("--lz") + 1], "1")

    def test_double_fast_cost_is_explicit_and_requires_its_own_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory), ["--mode", "baseline", "--mode", "double-fast-cost"])
            self.assertEqual(args.mode, ["baseline", "double-fast-cost"])
            self.assertEqual(bench.command(args, "in.png", "out.jxl", "double-fast-cost")[-2:],
                             ["--lz-quick", "double-fast-cost"])
        for other in ("sparse", "double-fast", "double-fast-sparse"):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "Quick receipt"):
                bench.validate_activation(self.measurement(mode=other), "double-fast-cost")

    def test_double_fast_cost_counters_and_intended_careful_bypass(self):
        mode = "double-fast-cost"
        valid = self.measurement(mode=mode)
        self.assertEqual(bench.validate_activation(valid, mode), "quick-active")
        bypass = dict(valid, lz_quick=dict(valid["lz_quick"], calls=0, queries=0, query_misses=0,
                                          cost_ranked_calls=0, preserved_careful_calls=2))
        self.assertEqual(bench.validate_activation(bypass, mode), "preserved-careful-bypass")
        mixed = dict(valid, lz_quick=dict(valid["lz_quick"], preserved_careful_calls=2))
        self.assertEqual(bench.validate_activation(mixed, mode), "quick-active")
        for key in ("cost_ranked_calls", "preserved_careful_calls"):
            receipt = dict(valid["lz_quick"])
            del receipt[key]
            with self.subTest(key=key, value="missing"), self.assertRaisesRegex(ValueError, key):
                bench.validate_activation(dict(valid, lz_quick=receipt), mode)
            for value in (None, True, False, -1, 0.5, "1"):
                receipt = dict(valid["lz_quick"], **{key: value})
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    bench.validate_activation(dict(valid, lz_quick=receipt), mode)
        for calls, ranked, careful in ((1, 0, 0), (1, 2, 1), (0, 1, 2), (0, 0, 0)):
            receipt = dict(valid["lz_quick"], calls=calls, cost_ranked_calls=ranked,
                           preserved_careful_calls=careful)
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                bench.validate_activation(dict(valid, lz_quick=receipt), mode)
        for other in ("sparse", "double-fast", "double-fast-sparse"):
            measurement = self.measurement(mode=other)
            measurement["lz_quick"].update(calls=0, preserved_careful_calls=2)
            with self.subTest(mode=other), self.assertRaisesRegex(ValueError, "did no work"):
                bench.validate_activation(measurement, other)

    def test_argument_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extra in (["--mode", "route"], ["--mode", "baseline", "--mode", "baseline"],
                          ["--threads", "0"], ["--threads", "257"], ["--repeats", "0"],
                          ["--warmups", "-1"], ["--timeout", "nan"], ["--timeout", "inf"],
                          ["--timeout", "0"], ["--effort", "11"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.arguments(root, extra)
            (root / "report.json").touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.arguments(root)

    def test_native_measurement_no_wall_time_or_diagnostics_contract(self):
        expected = SimpleNamespace(width=2, height=1)
        valid = self.measurement()
        bench.validate_measurement(valid, 100, expected)
        bench.validate_measurement(dict(valid, lz_scout={"future": [1, 2]}), 100, expected)
        for key, value in (("seconds", None), ("seconds", True), ("seconds", 0),
                           ("seconds", float("nan")), ("bytes", 99), ("bytes", True),
                           ("bpp", 8), ("mpix_per_second", 5), ("search", True),
                           ("lz", 2), ("lz", False),
                           ("width", 1), ("height", True)):
            invalid = dict(valid, wall_seconds=0.01)
            invalid[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate_measurement(invalid, 100, expected)
        with self.assertRaises(ValueError):
            bench.validate_measurement([], 100, expected)

    def test_manifest_relative_paths_and_no_count_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"{i}.png" for i in range(20)]
            for path in paths:
                path.touch()
            manifest = root / "cases.json"
            manifest.write_text(json.dumps({"cases": [{"original": p.name, "notes": "fixed-plan control"}
                                                       for p in paths]}), encoding="utf-8")
            args = SimpleNamespace(case=[], manifest=manifest)
            self.assertEqual(bench.load_cases(args), [p.resolve() for p in paths])
            self.assertEqual(bench.case_notes(args)[paths[0].resolve()], "fixed-plan control")
            args.case = [paths[0]]
            with self.assertRaises(ValueError):
                bench.load_cases(args)

    def test_aggregate_is_pixel_weighted_and_keeps_worse_files(self):
        cases = [dict(pixels=100, summaries=dict(baseline=dict(bytes=100, seconds=1),
                                               route=dict(bytes=130, seconds=0.5))),
                 dict(pixels=900, summaries=dict(baseline=dict(bytes=200, seconds=2),
                                               route=dict(bytes=190, seconds=1)))]
        result = bench.aggregate(cases, ["baseline", "route"])
        self.assertEqual(result["route"]["weighted_bpp"], 320 * 8 / 1000)
        self.assertEqual(result["route"]["bytes"], 320)
        self.assertEqual(result["route"]["byte_delta"], 20)
        self.assertEqual(result["route"]["size_regressions"], 1)
        self.assertEqual(result["route"]["size_improvements"], 1)
        self.assertEqual(result["route"]["seconds"], 1.5)

    def test_real_helper_keeps_16bit_and_hidden_rgb(self):
        expected = bench.png_samples(rgba_png())
        self.assertEqual(expected.depth, 16)
        self.assertEqual(int(expected.rgba[0, 0, 0]), 1234)
        bench.compare_pixels(expected, bench.png_samples(rgba_png()))
        with self.assertRaisesRegex(ValueError, "hidden RGB"):
            bench.compare_pixels(expected, bench.png_samples(rgba_png(hidden=1235)))
        with self.assertRaisesRegex(ValueError, "bit depth"):
            bench.compare_pixels(expected, bench.png_samples(rgba_png(8, 12)))

    def run_mock_matrix(self, root, failure=None, variant="route"):
        (root / "encoder.exe").write_bytes(b"encoder")
        (root / "djxl.exe").write_bytes(b"official decoder")
        (root / "input.png").write_bytes(rgba_png())
        args = self.arguments(root, ["--mode", "baseline", "--mode", variant, "--warmups", "1"])
        calls, artifacts = [], []

        def process(command, env, timeout):
            if command[0] == str(args.encoder):
                mode = command[-1] if "--lz-scout" in command or "--lz-quick" in command else "baseline"
                calls.append(mode)
                output = Path(command[command.index("--output") + 1])
                artifacts.append(output)
                size = 120 if mode == variant else 100
                output.write_bytes(b"x" * size)
                # Deliberately different warmup size/time must not affect medians.
                measurement = self.measurement(size, 100 if len(calls) <= 2 else 0.25, mode)
                if failure == "careful-bypass" and mode == "double-fast-cost":
                    measurement["lz_quick"].update(calls=0, queries=0, query_misses=0,
                                                   cost_ranked_calls=0, preserved_careful_calls=2)
                if failure == "json" and len(calls) == 1:
                    return dict(returncode=0, stdout="invalid", stderr="")
                if failure == "activation" and len(calls) == 1:
                    measurement.update(lz_scout=None, lz_quick=None)
                return dict(returncode=0, stdout=json.dumps(measurement), stderr="", wall_seconds=999)
            output = Path(command[2])
            artifacts.append(output)
            output.write_bytes(rgba_png(hidden=1235 if failure == "pixels" and len(calls) == 1 else 1234))
            return dict(returncode=0, stdout="", stderr="")

        with (patch.object(bench, "run_process", side_effect=process),
              contextlib.redirect_stdout(io.StringIO())):
            report = bench.benchmark(args)
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertTrue(all(not path.exists() for path in artifacts))
        self.assertEqual(len(artifacts), 16)  # Every output decoded, including malformed JSON.
        self.assertEqual(calls, [mode for repeat in range(-1, 3)
                                 for mode in bench.trial_order(args.mode, repeat, 0)])
        self.assertEqual(json.loads(args.out.read_text())["status"], report["status"])
        return report

    def test_full_matrix_rotation_warmups_and_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_mock_matrix(Path(directory))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["aggregate"]["route"]["size_regressions"], 1)
        summary = report["cases"][0]["summaries"]["route"]
        self.assertEqual(summary["seconds"], 0.25)
        self.assertEqual(summary["bytes"], 120)
        self.assertTrue(summary["size_regression"])

    def test_double_fast_cost_mock_matrix_retains_receipts_and_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_mock_matrix(Path(directory), variant="double-fast-cost")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["aggregate"]), {"baseline", "double-fast-cost"})
        self.assertEqual(report["aggregate"]["double-fast-cost"]["size_regressions"], 1)
        samples = report["cases"][0]["modes"]["double-fast-cost"]
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(sample["verified"] for sample in samples))
        self.assertTrue(all(sample["measurement"]["lz_quick"]["mode"] == "double-fast-cost"
                            for sample in samples))
        self.assertTrue(all(sample["experiment_status"] == "quick-active" for sample in samples))
        self.assertEqual(report["cases"][0]["summaries"]["double-fast-cost"]["seconds"], 0.25)

    def test_double_fast_cost_all_careful_report_is_bypass_not_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_mock_matrix(Path(directory), failure="careful-bypass", variant="double-fast-cost")
        self.assertEqual(report["status"], "passed")
        self.assertIn("NOT quick activation", report["diagnostics"])
        samples = report["cases"][0]["modes"]["double-fast-cost"]
        self.assertTrue(all(sample["verified"] for sample in samples))
        self.assertTrue(all(sample["experiment_status"] == "preserved-careful-bypass" for sample in samples))
        self.assertTrue(all(sample["measurement"]["lz_quick"]["calls"] == 0 for sample in samples))

    def test_failures_remain_strict_and_do_not_stop_other_trials(self):
        for failure in ("pixels", "json", "activation"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report = self.run_mock_matrix(Path(directory), failure)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(report["errors"]), 1)
                self.assertNotIn("aggregate", report)
                samples = [sample for samples in report["cases"][0]["modes"].values() for sample in samples]
                self.assertEqual(sum(sample["verified"] for sample in samples), 7)


if __name__ == "__main__":
    unittest.main()
