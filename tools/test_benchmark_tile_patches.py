"""Focused sidecar tests; no builds or real encoder/djxl subprocesses."""

from contextlib import redirect_stderr, redirect_stdout
from collections import Counter
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_tile_patches as bench
from test_compare_cpp_port import fixture


def measurement(mode="tiles", *, size=13, seconds=0.1, width=69, height=53,
                planned=True, repeated=1, tile_size=16, source_enabled=True,
                admission="screened", rejected=False):
    native = "off" if mode == "baseline" else mode
    result = dict(bytes=size, seconds=seconds, width=width, height=height, mode=native,
                  selected_mode=native, bpp=8 * size / (width * height),
                  mpix_per_second=width * height / 1e6 / seconds, tile_work=None, guard_work=None,
                  source_detector_enabled=source_enabled if mode == "source" else None,
                  tile_admission=admission if mode in ("tiles", "guarded") else None)
    if mode == "source" and not source_enabled:
        result["selected_mode"] = "off"
    if mode in ("tiles", "guarded"):
        result["selected_mode"] = "tiles" if planned else "off"
        result["tile_work"] = dict(
            tile_size=tile_size, calls=1, tiles_examined=12, constant_tiles=0,
            repeated_groups=repeated, selected_groups=int(planned),
            positions=12 if planned else 0, covered_pixels=3072 if planned else 0,
            atlas_pixels=256 if planned else 0, seconds=0.001,
            admission_rejected=int(rejected),
            estimated_saved_bits=-20.0 if rejected else 200.0 if planned else 0.0,
            estimated_baseline_bits=1000.0 if planned or rejected else 0.0,
            plan_selected=planned,
            note="rejected_by_estimate" if rejected else "plan_selected" if planned else
                 "no_selection" if repeated else "no_repeated_tiles")
    if mode == "guarded":
        result["guard_work"] = dict(off_bytes=size + int(planned), tiles_bytes=size,
                                    off_seconds=seconds / 3, tiles_seconds=seconds * 2 / 3)
    return result


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.argv = ["--encoder", "patch_lab.exe", "--decoder", "djxl.exe", "--synthetic",
                     "--out", str(self.root / "report.json")]

    def test_defaults_and_off_alias(self):
        args = bench.arguments(self.argv)
        self.assertEqual((args.effort, args.threads, args.repeats, args.warmups), (9, 8, 3, 0))
        self.assertEqual(args.modes, list(bench.DEFAULT_MODES))
        self.assertNotIn("guarded", args.modes)
        self.assertEqual(args.tile_admission, "screened")
        explicit = bench.arguments(self.argv + ["--modes", "baseline", "guarded", "--tile-admission", "force"])
        self.assertEqual((explicit.modes, explicit.tile_admission), (["baseline", "guarded"], "force"))
        self.assertEqual(bench.arguments(self.argv + ["--modes", "off", "tiles"]).modes,
                         ["baseline", "tiles"])

    def test_bad_args_rejected_without_execution(self):
        for extra in (["--effort", "3"], ["--effort", "11"], ["--threads", "0"],
                      ["--threads", "257"], ["--tile-size", "0"], ["--tile-size", "-1"],
                      ["--repeats", "0"], ["--warmups", "-1"], ["--timeout", "nan"],
                      ["--timeout", "0"], ["--modes", "tiles"],
                      ["--tile-admission", "automatic"],
                      ["--modes", "baseline", "off"], ["--modes", "baseline", "unknown"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), \
                    patch.object(bench.port, "run_process") as run:
                with self.assertRaises(SystemExit):
                    bench.arguments(self.argv + extra)
                run.assert_not_called()
        self.assertFalse((self.root / "report.json").exists())
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments([arg for arg in self.argv if arg != "--synthetic"])

    def test_report_is_never_overwritten(self):
        report = self.root / "report.json"
        report.write_text("keep", encoding="utf-8")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv)
        self.assertEqual(report.read_text(), "keep")

    def test_exact_command_and_optional_tile_size(self):
        args = bench.arguments(self.argv + ["--tile-size", "8", "--effort", "10", "--threads", "2"])
        for mode in (*bench.MODES, "off"):
            with self.subTest(mode=mode):
                expected = ["patch_lab.exe", "source.png", "--output", "out.jxl", "--effort", "10",
                            "--threads", "2", "--mode", "off" if mode == "baseline" else mode]
                if mode in ("tiles", "guarded"):
                    expected += ["--tile-admission", "screened", "--tile-size", "8"]
                self.assertEqual(bench.command(args, "source.png", "out.jxl", mode), expected)
        args.tile_size = None
        args.tile_admission = "force"
        self.assertNotIn("--tile-size", bench.command(args, "s", "o", "tiles"))
        for mode in bench.MODES:
            command = bench.command(args, "s", "o", mode)
            if mode in ("tiles", "guarded"):
                self.assertEqual(command[-2:], ["--tile-admission", "force"])
            else:
                self.assertNotIn("--tile-admission", command)
        with self.assertRaises(ValueError):
            bench.command(args, "s", "o", "other")

    def test_manifest_uses_existing_relative_path_loader(self):
        source = self.root / "sample.png"
        source.write_bytes(fixture())
        manifest = self.root / "cases.json"
        manifest.write_text(json.dumps({"cases": [{"original": source.name}]}), encoding="utf-8")
        args = bench.arguments(self.argv + ["--manifest", str(manifest)])
        self.assertEqual(bench.load_cases(args), [source.resolve()])


class ReceiptTests(unittest.TestCase):
    def validate(self, m, mode="tiles", admission="screened"):
        bench.validate_measurement(m, mode, 13, 69, 53, tile_admission=admission)

    def test_all_modes_and_no_plan_receipts(self):
        for mode in bench.MODES:
            with self.subTest(mode=mode):
                self.validate(measurement(mode), mode)
        for repeated in (0, 4):
            for mode in ("tiles", "guarded"):
                with self.subTest(repeated=repeated, mode=mode):
                    m = measurement(mode, planned=False, repeated=repeated)
                    self.validate(m, mode)
                    self.assertEqual(m["tile_work"]["note"], "no_selection" if repeated else "no_repeated_tiles")

    def test_admission_policy_is_required_and_matches_request(self):
        for mode in bench.MODES:
            for admission in bench.ADMISSIONS:
                m = measurement(mode, admission=admission)
                self.validate(m, mode, admission)
                if mode in ("tiles", "guarded"):
                    other = "force" if admission == "screened" else "screened"
                    with self.assertRaises(ValueError):
                        self.validate(m, mode, other)
                else:
                    m["tile_admission"] = admission
                    with self.assertRaises(ValueError):
                        self.validate(m, mode, admission)
            m = measurement(mode)
            del m["tile_admission"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.validate(m, mode)

    def test_rejected_estimate_allows_negative_savings_and_zero_selected_work(self):
        for mode in ("tiles", "guarded"):
            m = measurement(mode, planned=False, rejected=True)
            self.validate(m, mode)
            self.assertEqual(m["selected_mode"], "off")
            self.assertEqual(m["tile_work"]["note"], "rejected_by_estimate")
            for key in ("selected_groups", "positions", "covered_pixels", "atlas_pixels"):
                self.assertEqual(m["tile_work"][key], 0)
            # Small positive estimates may also fail the admission threshold.
            m["tile_work"]["estimated_saved_bits"] = 1.0
            self.validate(m, mode)
            forced = measurement(mode, admission="force")
            forced["tile_work"]["estimated_saved_bits"] = -20.0
            self.validate(forced, mode, "force")
            self.validate(measurement(mode, planned=False, repeated=0, admission="force"), mode, "force")

    def test_rejects_false_no_plan_and_invalid_admission_evidence(self):
        for updates in (dict(repeated_groups=0), dict(selected_groups=1), dict(positions=1),
                        dict(covered_pixels=1), dict(atlas_pixels=1), dict(plan_selected=True),
                        dict(note="no_selection"), dict(admission_rejected=2)):
            m = measurement(planned=False, rejected=True)
            m["tile_work"].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.validate(m)
        m = measurement(planned=False, rejected=True, admission="force")
        with self.assertRaises(ValueError):
            self.validate(m, admission="force")
        m = measurement(planned=False, rejected=True)
        m["selected_mode"] = "tiles"
        with self.assertRaises(ValueError):
            self.validate(m)
        m = measurement(planned=False)
        m["tile_work"]["note"] = "rejected_by_estimate"
        with self.assertRaises(ValueError):
            self.validate(m)
        for key, values in (
            ("admission_rejected", (None, True, -1, 0.5)),
            ("estimated_saved_bits", (None, True, "0", float("nan"), float("inf"), -float("inf"))),
            ("estimated_baseline_bits", (None, True, -1, float("nan"), float("inf"))),
        ):
            for value in values:
                m = measurement()
                m["tile_work"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(m)
            m = measurement()
            del m["tile_work"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.validate(m)

    def test_rejects_bad_receipts(self):
        mutations = [
            lambda m: m.pop("tile_work"),
            lambda m: m.update(tile_work=None),
            lambda m: m.update(mode="off"),
            lambda m: m.update(selected_mode="off"),
            lambda m: m.update(seconds=0),
            lambda m: m.update(seconds=float("nan")),
            lambda m: m.update(bytes=True),
            lambda m: m.update(bytes=12),
            lambda m: m.update(width=1),
            lambda m: m.update(bpp=0),
            lambda m: m.update(mpix_per_second=99),
            lambda m: m["tile_work"].update(tile_size=8),
            lambda m: m["tile_work"].update(calls=0),
            lambda m: m["tile_work"].update(calls=True),
            lambda m: m["tile_work"].update(positions=-1),
            lambda m: m["tile_work"].update(covered_pixels=0),
            lambda m: m["tile_work"].update(constant_tiles=99),
            lambda m: m["tile_work"].update(repeated_groups=0),
            lambda m: m["tile_work"].update(seconds=float("inf")),
            lambda m: m["tile_work"].update(seconds=0.5),
            lambda m: m["tile_work"].update(plan_selected=False),
            lambda m: m["tile_work"].update(note="no_selection"),
        ]
        for mutate in mutations:
            m = measurement()
            mutate(m)
            with self.subTest(m=m), self.assertRaises(ValueError):
                self.validate(m)
        m = measurement(planned=False)
        m["tile_work"]["positions"] = 1
        with self.assertRaises(ValueError):
            self.validate(m)
        m = measurement("baseline")
        m["tile_work"] = measurement()["tile_work"]
        with self.assertRaises(ValueError):
            self.validate(m, "baseline")

    def test_source_activation_is_explicit_not_dictionary_emission(self):
        for enabled in (True, False):
            m = measurement("source", source_enabled=enabled)
            self.validate(m, "source")
            self.assertEqual(m["selected_mode"], "source" if enabled else "off")
            m["selected_mode"] = "off" if enabled else "source"
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                self.validate(m, "source")
        for value in (None, 0, 1, "true"):
            m = measurement("source")
            m["source_detector_enabled"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(m, "source")
        for mode in bench.MODES:
            m = measurement(mode)
            del m["source_detector_enabled"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.validate(m, mode)
            if mode != "source":
                for value in (True, False):
                    m = measurement(mode)
                    m["source_detector_enabled"] = value
                    with self.subTest(mode=mode, value=value), self.assertRaises(ValueError):
                        self.validate(m, mode)

    def test_guard_requires_min_size_both_times_and_correct_selection(self):
        m = measurement("guarded")
        self.validate(m, "guarded")
        for updates in (dict(tiles_seconds=0), dict(off_seconds=0.5), dict(tiles_bytes=12),
                        dict(off_bytes=12), dict(tiles_bytes=True)):
            bad = deepcopy(m)
            bad["guard_work"].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.validate(bad, "guarded")
        # A tie chooses off even though the detector found a valid plan.
        m["guard_work"]["off_bytes"] = 13
        m["selected_mode"] = "off"
        self.validate(m, "guarded")
        del m["guard_work"]
        with self.assertRaises(ValueError):
            self.validate(m, "guarded")


class SummaryTests(unittest.TestCase):
    def test_summary_excludes_warmups_and_uses_median_duration(self):
        samples = [dict(warmup=warmup, verified=True,
                        measurement=dict(seconds=seconds, bytes=size))
                   for warmup, seconds, size in ((True, 9, 99), (False, 1, 10), (False, 3, 14))]
        result = bench.summarize(samples, 100)
        self.assertEqual((result["seconds"], result["bytes"]), (2, 12))
        self.assertAlmostEqual(result["mpix_per_second"], 100 / 1e6 / 2)

    def test_aggregate_is_ratio_of_totals_and_reports_regressions(self):
        cases = [dict(pixels=100, summaries=dict(baseline=dict(bytes=10, seconds=2),
                                               tiles=dict(bytes=5, seconds=3))),
                 dict(pixels=300, summaries=dict(baseline=dict(bytes=90, seconds=8),
                                               tiles=dict(bytes=105, seconds=9)))]
        result = bench.aggregate(cases, ["baseline", "tiles"])
        self.assertEqual((result["tiles"]["bytes"], result["tiles"]["seconds"]), (110, 12))
        self.assertAlmostEqual(result["tiles"]["ratios_to_baseline"]["bytes"], 1.1)
        self.assertEqual(result["tiles"]["regressions"], dict(bytes=True, seconds=True))
        self.assertEqual(result["baseline"]["regressions"], dict(bytes=False, seconds=False))
        self.assertAlmostEqual(result["tiles"]["mpix_per_second"], 400 / 1e6 / 12)

    def test_aggregate_never_divides_by_zero_or_nonfinite_values(self):
        with self.assertRaises(ValueError):
            bench.aggregate([], ["baseline"])
        for key in ("pixels", "bytes", "seconds"):
            for value in (0, -1, float("nan"), float("inf"), True):
                case = dict(pixels=100, summaries=dict(baseline=dict(bytes=10, seconds=1)))
                if key == "pixels":
                    case[key] = value
                else:
                    case["summaries"]["baseline"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    bench.aggregate([case], ["baseline"])


class FixtureTests(unittest.TestCase):
    def test_genuine_textured_tiles_all_channels_depths_hidden_colors_and_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = bench.fixtures(Path(temp) / "fixtures")
            self.assertEqual(len(paths), 11)
            formats = set()
            for path in paths:
                data = path.read_bytes()
                _, header = bench.port.png_payload(data)
                pixels = bench.port.png_samples(data)
                if path.name.startswith("tiles-multi-"):
                    self.assertEqual((pixels.width, pixels.height, pixels.depth, header["color_type"]),
                                     (69, 53, 8, 2))
                    groups = Counter(pixels.rgba[y:y + 16, x:x + 16].tobytes()
                                     for y in range(0, 48, 16) for x in range(0, 64, 16))
                    self.assertEqual(sorted(groups.values()), [4, 4, 4])
                    self.assertTrue(all(len(set(tile)) > 64 for tile in groups))
                    continue
                if not path.name.startswith("tiles-"):
                    tiles = {pixels.rgba[y:y + 16, x:x + 16].tobytes()
                             for y in range(0, pixels.height - 15, 16)
                             for x in range(0, pixels.width - 15, 16)}
                    self.assertEqual(len(tiles), (pixels.width // 16) * (pixels.height // 16))
                    continue
                formats.add((header["bit_depth"], header["color_type"]))
                self.assertEqual((pixels.width % 16, pixels.height % 16), (5, 5))
                first = pixels.rgba[:16, :16]
                self.assertGreater(len(set(first[:, :, 0].flat)), 64)
                for y in range(0, 48, 16):
                    for x in range(0, 64, 16):
                        self.assertEqual(first.tobytes(), pixels.rgba[y:y + 16, x:x + 16].tobytes())
                if pixels.depth == 16:
                    self.assertTrue((first[:, :, 0] > 255).any())
                    self.assertTrue((first[:, :, 0] % 256 != 0).any())
                if header["color_type"] in (4, 6):
                    hidden = first[:, :, 3] == 0
                    self.assertTrue(hidden.any())
                    self.assertTrue((first[:, :, :3][hidden] != 0).any())
                    modified = bench.port.png_samples(data)
                    modified.rgba[0, 0, 0] ^= 1
                    with self.assertRaisesRegex(ValueError, "hidden RGB"):
                        bench.port.compare_pixels(pixels, modified)
            self.assertEqual(formats, {(d, c) for d in (8, 16) for c in (0, 4, 2, 6)})


class HarnessTests(unittest.TestCase):
    def test_rotating_repeats_cleanup_no_hash_equality_and_failure_reports(self):
        scenarios = [(None, guarded, admission) for guarded in (False, True) for admission in bench.ADMISSIONS]
        scenarios += [(failure, False, "screened") for failure in ("pixels", "receipt", "encoder")]
        for failure, guarded, admission in scenarios:
            with self.subTest(failure=failure, guarded=guarded, admission=admission), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, encoder, decoder = (root / name for name in ("input.png", "patch_lab.exe", "djxl.exe"))
                source.write_bytes(fixture())
                encoder.write_bytes(b"encoder")
                decoder.write_bytes(b"decoder")
                out = root / "report.json"
                argv = ["--encoder", str(encoder), "--decoder", str(decoder), "--case", str(source),
                        "--repeats", "2", "--warmups", "1", "--out", str(out), "--tile-admission", admission]
                modes = list(bench.MODES if guarded else bench.DEFAULT_MODES)
                if guarded:
                    argv += ["--modes", *modes]
                calls, visits = [], []

                def run(command, env, timeout):
                    calls.append(command)
                    self.assertNotIn("RS_JXL_PROFILE", env)
                    stdout, code = "", 0
                    if command[0] == str(encoder):
                        mode = command[command.index("--mode") + 1]
                        visits.append(mode)
                        size = 12 + len(visits)
                        Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                        if mode in ("tiles", "guarded"):
                            self.assertEqual(command[command.index("--tile-admission") + 1], admission)
                        else:
                            self.assertNotIn("--tile-admission", command)
                        m = measurement(mode, width=2, height=1, size=size, planned=False, repeated=0,
                                        admission=admission)
                        if failure == "receipt":
                            m["seconds"] = 0
                        if failure == "encoder":
                            code = 1
                        stdout = json.dumps(m)
                    else:
                        Path(command[2]).write_bytes(fixture(8 if failure == "pixels" else 16))
                    return dict(command=command, returncode=code, stdout=stdout, stderr="failed" if code else "",
                                wall_seconds=0.2)

                with patch.object(bench.port, "run_process", side_effect=run), \
                        patch.dict(os.environ, {"RS_JXL_PROFILE": "1"}), \
                        redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(bench.main(argv), int(failure is not None))
                report = json.loads(out.read_text(encoding="utf-8"))
                self.assertTrue(report["temporary_outputs_cleaned"])
                self.assertEqual({p.name for p in root.iterdir()}, {source.name, encoder.name, decoder.name, out.name})
                if failure:
                    self.assertEqual(report["status"], "failed")
                    self.assertTrue(report["errors"])
                    self.assertNotIn("aggregate", report)
                    self.assertEqual(len(calls), 1 if failure == "encoder" else 2)
                    self.assertFalse(report["cases"][0]["variants"]["tiles"][0]["verified"])
                else:
                    expected = ["off" if m == "baseline" else m for r in (-1, 0, 1)
                                for m in bench.trial_order(modes, r, 0)]
                    self.assertEqual(visits, expected)
                    self.assertEqual("guarded" in visits, guarded)
                    self.assertEqual(len(calls), 6 * len(modes))
                    self.assertEqual(report["status"], "passed")
                    self.assertEqual(report["config"]["tile_admission"], admission)
                    for mode in modes:
                        self.assertEqual(len(report["cases"][0]["variants"][mode]), 3)
                        self.assertEqual(len(report["cases"][0]["summaries"][mode]["sizes"]), 2)


if __name__ == "__main__":
    unittest.main()
