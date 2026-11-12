"""Native labs and official djxl are mocked; original sample oracle is real."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_ma_rct as bench
import plot_patch_tradeoffs as plot
from test_compare_cpp_port import fixture, make_png


def measurement(mode, size=20, seconds=1.0, width=2, height=1):
    result = dict(bytes=size, seconds=seconds, bpp=8 * size / (width * height),
                  mpix_per_second=width * height / 1e6 / seconds, width=width, height=height,
                  mode=bench.native_mode(mode), effort=9, threads=8, previous_channels=0,
                  tree_learning="legacy", lookahead=None, lookahead_options=None,
                  patches="Off", patches_cleared=True,
                  native_encode_calls=1, lookahead_score_contract=bench.SCORE_CONTRACT)
    if mode != "baseline":
        result["lookahead_options"] = dict(width=int(mode[-1]), min_records=8192, relative_window=0.01)
        result["lookahead"] = dict(roots=2, eligible=1, probed_candidates=2,
                                   child_searches=4, changed=1, estimated_saved_bits=123456.5)
    return result


def rct_receipt(size=18, width=2, height=1):
    shortlisted = [6, 10, 17]
    trials = []
    for index in range(13):
        previous = 0 if index == 0 else (index - 1) % 3
        params = dict(speed_tier=1, max_properties=previous, patches="Off",
                      rct=None if index < 4 else shortlisted[(index - 4) // 3])
        trials.append(dict(index=index,
                           kind="native_e9_baseline" if index == 0 else "auto_rct_control" if index < 4 else "shortlisted_rct",
                           params=params, effective=dict(previous_channels=previous, training_previous_channels=previous,
                                                         patches_disabled=True),
                           bytes=size if index == 5 else 30, seconds=0.0 if index == 1 else 1.0,
                           error=None, reused_from=0 if index == 1 else None))
    return dict(schema="rs-jxl.rct-joint-lab.v1", effort=9, width=width, height=height,
                threads=8, shortlist_limit=3, sample_pixel_limit=8192, winner_trial=5,
                winner_bytes=size, baseline_trial=0, auto_rct_control_trials=[1, 2, 3], actual_encodes=12,
                selection="complete_standard_jxl_bytes_first_minimum", decoder_policy="unchanged",
                patch_policy="disabled", baseline_policy="native_e9_patches_off",
                screen=dict(real_ma_training=False, sampled_pixels=min(2, width * height),
                            shortlisted_rcts=shortlisted, seconds=2.0,
                            scores=[dict(rct=i, native_bits=100.0, proxy_bits=[100.0, 90.0, 80.0], seconds=0.01)
                                    for i in range(42)]),
                timing=dict(total_seconds=15.0, screen_seconds=2.0, encode_seconds=12.0,
                            overhead_seconds=1.0, pool_setup_seconds=0.5), trials=trials)


class ArgumentsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.argv = ["--encoder", "ma.exe", "--decoder", "djxl.exe", "--synthetic",
                     "--out", str(self.root / "report.json")]

    def test_defaults_and_rct_opt_in(self):
        args = bench.arguments(self.argv)
        self.assertEqual(args.modes, list(bench.MA_MODES))
        self.assertEqual((args.effort, args.threads, args.repeats, args.min_records), (9, 8, 2, 8192))
        self.assertEqual(bench.arguments(self.argv + ["--rct-encoder", "rct.exe"]).modes, list(bench.MODES))
        self.assertEqual(bench.arguments(self.argv + ["--modes", "greedy", "lookahead2"]).modes,
                         ["baseline", "lookahead2"])

    def test_invalid_arguments_before_processes(self):
        for flags in (["--effort", "3"], ["--effort", "11"], ["--threads", "0"],
                      ["--threads", "257"], ["--min-records", "3"], ["--relative-window", "nan"],
                      ["--relative-window", "inf"], ["--relative-window", "1.1"],
                      ["--max-cases", "0"], ["--max-pixels", "0"], ["--repeats", "0"],
                      ["--timeout", "nan"], ["--warmups", "-1"], ["--modes", "lookahead2"],
                      ["--modes", "baseline", "greedy"], ["--modes", "baseline", "rct-joint"],
                      ["--rct-encoder", "rct.exe", "--tree-learning", "cost"],
                      ["--rct-encoder", "rct.exe", "--effort", "8"], ["--shortlist", "9"],
                      ["--sample-pixels", "8"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), \
                    patch.object(bench.port, "run_process") as run, self.assertRaises(SystemExit):
                bench.arguments(self.argv + flags)
                run.assert_not_called()

    def test_commands_match_independent_clis_and_no_baseline_tuning(self):
        args = bench.arguments(self.argv + ["--rct-encoder", "rct.exe", "--min-records", "4"])
        for mode in bench.MA_MODES:
            cmd = bench.command(args, "in.png", "out.jxl", mode)
            self.assertEqual(cmd[cmd.index("--mode") + 1], bench.native_mode(mode))
            self.assertEqual("--min-records" in cmd, mode != "baseline")
            self.assertNotIn("--shortlist", cmd)
        self.assertEqual(bench.command(args, "in.png", "out.jxl", "rct-joint"),
                         ["rct.exe", "in.png", "--output", "out.jxl", "--threads", "8",
                          "--shortlist", "3", "--sample-pixels", "8192"])
        with self.assertRaises(ValueError):
            bench.command(args, "i", "o", "guarded")


class ReceiptTests(ArgumentsTests):
    def test_ma_receipts_zero_activation_and_estimates_not_wire_bits(self):
        args = bench.arguments(self.argv)
        for mode in bench.MA_MODES:
            m = measurement(mode)
            bench.validate_measurement(m, mode, 20, 2, 1, args)
            if mode != "baseline":
                m["lookahead"] = dict.fromkeys((*bench.COUNTERS, "estimated_saved_bits"), 0)
                bench.validate_measurement(m, mode, 20, 2, 1, args)

    def test_invalid_native_metrics_and_ma_receipts(self):
        args = bench.arguments(self.argv)
        for updates in (dict(bytes=True), dict(bytes=19), dict(seconds=0), dict(seconds=float("nan")),
                        dict(width=2.0), dict(bpp=1), dict(mpix_per_second=99), dict(native_encode_calls=2),
                        dict(tree_learning="cost"), dict(previous_channels=1), dict(threads=1),
                        dict(patches="Default"), dict(patches_cleared=False),
                        dict(lookahead_score_contract="wire bits"), dict(lookahead=None)):
            m = measurement("lookahead2") | updates
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                bench.validate_measurement(m, "lookahead2", 20, 2, 1, args)
        for updates in (dict(roots=-1), dict(eligible=True), dict(eligible=3), dict(changed=2),
                        dict(probed_candidates=3), dict(child_searches=5), dict(estimated_saved_bits=-1)):
            m = measurement("lookahead2")
            m["lookahead"].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                bench.validate_measurement(m, "lookahead2", 20, 2, 1, args)

    def test_rct_full_search_cost_errors_and_unverified_controls(self):
        args = bench.arguments(self.argv + ["--rct-encoder", "rct.exe"])
        receipt = rct_receipt()
        receipt["trials"][6].update(bytes=None, error="candidate failed")
        m = bench.rct_measurement(receipt, 18, 2, 1, args)
        self.assertEqual(m["seconds"], 15.0)
        self.assertNotEqual(m["seconds"], receipt["trials"][5]["seconds"])
        self.assertEqual(m["winner_trial"], 5)
        self.assertFalse(m["all_candidates_verified"])
        self.assertEqual(m["rct_joint"]["trials"][6]["error"], "candidate failed")
        summary = bench.summarize_rct([dict(warmup=False, measurement=m)])
        self.assertEqual(summary["winner_trials"], [5])
        self.assertFalse(summary["all_candidates_verified"])
        self.assertTrue(any(s["errors"] for s in summary["controls_and_trials"]))

    def test_rct_rejects_hidden_costs_missing_rows_and_false_winners(self):
        args = bench.arguments(self.argv + ["--rct-encoder", "rct.exe"])
        for mutate in (lambda r: r["screen"]["scores"].pop(),
                       lambda r: r["trials"].pop(),
                       lambda r: r.update(winner_trial=0),
                       lambda r: r.update(actual_encodes=1),
                       lambda r: r["timing"].update(total_seconds=1.0),
                       lambda r: r["trials"][1].update(seconds=1.0),
                       lambda r: r["trials"][0]["params"].update(max_properties=2),
                       lambda r: r["trials"][0]["params"].update(patches="Default"),
                       lambda r: r["trials"][0]["effective"].update(patches_disabled=False),
                       lambda r: r.update(patch_policy="default")):
            r = deepcopy(rct_receipt())
            mutate(r)
            with self.assertRaises(ValueError):
                bench.rct_measurement(r, 18, 2, 1, args)
        for text in ('{"seconds":NaN}', '{"seconds":Infinity}', '{"seconds":1e999}'):
            with self.assertRaises(ValueError):
                bench.native_json(text)

    def test_small_image_overlapping_tiles_count_records_not_distinct_pixels(self):
        args = bench.arguments(self.argv + ["--rct-encoder", "rct.exe"])
        receipt = rct_receipt(width=40, height=40)
        receipt["screen"]["sampled_pixels"] = 8100
        m = bench.rct_measurement(receipt, 18, 40, 40, args)
        self.assertGreater(m["rct_joint"]["screen"]["sampled_pixels"], 40 * 40)
        self.assertIn("not distinct pixels", m["sample_pixel_contract"])
        receipt["screen"]["sampled_pixels"] = args.sample_pixels + 1
        with self.assertRaises(ValueError):
            bench.rct_measurement(receipt, 18, 40, 40, args)

    def test_targeted_small_configuration_and_real_matrix_bounds(self):
        args = bench.arguments(self.argv + ["--min-records", "4", "--relative-window", "0",
                                            "--max-pixels", "16000000", "--max-cases", "16"])
        m = measurement("lookahead4")
        m["lookahead_options"].update(min_records=4, relative_window=0.0)
        bench.validate_measurement(m, "lookahead4", 20, 2, 1, args)
        cmd = bench.command(args, "i", "o", "lookahead4")
        self.assertEqual(cmd[cmd.index("--min-records") + 1], "4")
        self.assertEqual(cmd[cmd.index("--relative-window") + 1], "0.0")


class HarnessTests(unittest.TestCase):
    def exercise(self, depth=16, failure=None, rct=False, extra=()):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source, encoder, decoder, rct_encoder = (root / n for n in ("input.png", "ma.exe", "djxl.exe", "rct.exe"))
        source.write_bytes(fixture(depth))
        for binary in (encoder, decoder, rct_encoder):
            binary.write_bytes(binary.name.encode())
        out = root / "report.json"
        argv = ["--encoder", str(encoder), "--decoder", str(decoder), "--case", str(source),
                "--warmups", "1", "--out", str(out), *extra]
        if rct:
            argv += ["--rct-encoder", str(rct_encoder)]
        calls, visits = [], []

        def run(command, env, timeout):
            calls.append(command)
            self.assertNotIn("RS_JXL_PROFILE", env)
            self.assertNotIn("JXL_TEST", env)
            if command[0] != str(decoder):
                self.assertEqual(Path(command[1]).read_bytes(), fixture(depth))
                mode = "rct-joint" if command[0] == str(rct_encoder) else command[command.index("--mode") + 1]
                mode = "baseline" if mode == "greedy" else mode
                visits.append(mode)
                size = {"baseline": 20, "lookahead2": 30, "lookahead4": 25, "rct-joint": 18}[mode]
                Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                m = rct_receipt(size) if mode == "rct-joint" else measurement(mode, size, float(len(visits)))
                if failure == "receipt":
                    m.pop("lookahead")
                if failure == "nan":
                    m["seconds"] = float("nan")
                if failure == "binary_changed":
                    encoder.write_bytes(b"changed")
                return dict(returncode=int(failure == "encoder"), stdout=json.dumps(m), stderr="", wall_seconds=999)
            self.assertIn(f"--bits_per_sample={depth}", command)
            decoded = fixture(depth)
            if failure == "hidden":
                samples = [259 ^ 1, 513, 65534, 0, 1, 257, 32769, 65535] if depth == 16 else [19 ^ 1, 81, 253, 0, 7, 11, 13, 255]
                decoded = make_png(depth, 6, [samples])
            if failure == "narrowed":
                decoded = fixture(8)
            Path(command[2]).write_bytes(decoded)
            return dict(returncode=int(failure == "decoder"), stdout="", stderr="", wall_seconds=999)

        with patch.object(bench.port, "run_process", side_effect=run), \
                patch.dict(os.environ, {"RS_JXL_PROFILE": "1", "JXL_TEST": "1"}), redirect_stdout(StringIO()):
            code = bench.main(argv)
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, int(report["status"] != "passed"))
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual({p.name for p in root.iterdir()}, {"input.png", "ma.exe", "djxl.exe", "rct.exe", "report.json"})
        return report, calls, visits

    def test_native_medians_raw_regressions_and_plot_schema(self):
        for depth in (8, 16):
            report, calls, visits = self.exercise(depth, rct=True)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(visits, [m for repeat in (-1, 0, 1) for m in bench.trial_order(bench.MODES, repeat, 0)])
            self.assertEqual(len(calls), 24)
            corpus, points = plot.validate_report(report)
            self.assertEqual(len(corpus), 4)
            self.assertEqual(len(points), 4)
            case = report["cases"][0]
            self.assertEqual(case["summaries"]["lookahead2"]["bytes"], 30)
            self.assertTrue(case["summaries"]["lookahead2"]["regressions"]["bytes"])
            self.assertEqual(case["summaries"]["rct-joint"]["seconds"], 15)
            for sample in case["variants"]["rct-joint"]:
                self.assertTrue(sample["verified"])
                self.assertEqual(sample["verification_scope"], "exported_winner_only")
                self.assertEqual(sample["verified_candidate_indices"], [5])
                self.assertFalse(sample["all_candidates_verified"])

    def test_failures_preserve_report_and_do_not_verify_bad_samples(self):
        for depth in (8, 16):
            for failure in ("hidden", "receipt", "encoder", "decoder", "nan", "binary_changed"):
                with self.subTest(depth=depth, failure=failure):
                    report, _, _ = self.exercise(depth, failure)
                    self.assertEqual(report["status"], "failed")
                    self.assertTrue(report["errors"])
                    if failure == "hidden":
                        self.assertIn("hidden RGB", report["errors"][0])
        report, _, _ = self.exercise(16, "narrowed")
        self.assertEqual(report["status"], "failed")

    def test_bounds_fail_before_native_calls(self):
        for extra in (("--max-pixels", "1"), ("--synthetic", "--max-cases", "1")):
            report, calls, _ = self.exercise(extra=extra)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(calls, [])

    def test_no_overwrite_or_missing_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "report.json"
            argv = ["--encoder", str(Path(directory) / "missing"), "--decoder", "djxl.exe",
                    "--synthetic", "--out", str(out)]
            with redirect_stderr(StringIO()), patch.object(bench.port, "run_process") as run:
                self.assertEqual(bench.main(argv), 1)
                run.assert_not_called()
                before = out.read_bytes()
                with self.assertRaises(SystemExit):
                    bench.arguments(argv)
                self.assertEqual(out.read_bytes(), before)


class FixtureTests(unittest.TestCase):
    def test_six_bounded_original_rgb_rgba_formats_and_hidden_16bit(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = bench.fixtures(Path(directory) / "synthetic")
            self.assertEqual(len(paths), 6)
            formats = set()
            for path in paths:
                _, info = bench.png_framing(path.read_bytes())
                formats.add((info["bit_depth"], info["color_type"]))
                self.assertLess(info["width"] * info["height"], 1048576)
                if info["color_type"] == 6:
                    samples = bench.port.png_samples(path.read_bytes())
                    hidden = samples.rgba[:, :, 3] == 0
                    self.assertTrue(hidden.any())
                    self.assertTrue((samples.rgba[:, :, :3][hidden] != 0).any())
                    if samples.depth == 16:
                        self.assertTrue((samples.rgba[:, :, :3][hidden] > 255).any())
                        self.assertTrue((samples.rgba[:, :, :3] % 257 != 0).any())
            self.assertEqual(formats, {(8, 2), (8, 6), (16, 2), (16, 6)})


if __name__ == "__main__":
    unittest.main()
