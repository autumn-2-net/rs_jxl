"""Sidecar contract tests; encoder/djxl subprocesses are always mocked."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Support direct invocation as well as unittest discovery from tools.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_qlic_candidates as bench
from test_benchmark_tile_patches import measurement as tile_measurement
from test_compare_cpp_port import fixture, make_png


def measurement(mode="baseline", **kwargs):
    result = tile_measurement(bench.native_mode(mode), **kwargs)
    result.update(block_size=None, block_work=None, block_plan_leaves=None)
    if mode in bench.PARALLEL_MODES:
        result["parallel_gather"] = True
    if mode in ("tiles", "offgrid"):
        result["tile_work"].update(off_grid=mode == "offgrid", windows_examined=29,
                                   discovery_limited=0)
    target, size = bench.block_settings(mode)
    if size is not None:
        result.update(block_size=size, block_plan_leaves=4096, block_work=block_work())
    if mode in bench.SOURCE_BLOCKS:
        result.update(block_target=target, block_work={"reference": None, "residual": None})
        for part in ("reference", "residual"):
            if target in (part, "both"):
                configured = part == "residual" or result["source_detector_enabled"]
                work = block_work() if configured else block_work(used=False)
                work.update(configured=configured, used=configured)
                result["block_work"][part] = work
    return result


def block_work(used=True):
    if not used:
        return dict.fromkeys(bench.BLOCK_COUNTERS, 0) | {"predictor_leaves": [0, 0, 0, 0]}
    return dict(trees=2, pixels=600, views=6, requested_tiles=12, leaves=5,
                spatial_splits=1, channel_splits=1, stream_splits=1,
                coarsened_views=1, predictor_leaves=[1, 2, 1, 1])


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.argv = ["--encoder", "patch_lab.exe", "--official-djxl", "djxl.exe",
                     "--synthetic", "--out", str(self.root / "report.json")]

    def test_defaults_and_single_repeat(self):
        args = bench.arguments(self.argv)
        self.assertEqual(args.modes, list(bench.MODES))
        self.assertEqual((args.effort, args.threads, args.repeats, args.warmups), (9, 8, 3, 0))
        self.assertEqual(bench.arguments(self.argv + ["--repeats", "1"]).repeats, 1)

    def test_invalid_arguments_never_execute(self):
        for extra in (["--effort", "3"], ["--effort", "11"], ["--threads", "0"],
                      ["--threads", "257"], ["--tile-size", "0"], ["--warmups", "-1"],
                      ["--repeats", "0"], ["--timeout", "nan"], ["--timeout", "inf"],
                      ["--timeout", "0"], ["--modes", "offgrid"],
                      ["--modes", "baseline", "baseline"], ["--modes", "baseline", "guarded"],
                      ["--modes", "baseline", "source-ref32"], ["--block-leaves", "0"],
                      ["--modes", "baseline", "source-ref8-parallel"],
                      ["--tile-admission", "auto"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), \
                    patch.object(bench.port, "run_process") as run:
                with self.assertRaises(SystemExit):
                    bench.arguments(self.argv + extra)
                run.assert_not_called()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments([arg for arg in self.argv if arg != "--synthetic"])

    def test_report_no_overwrite(self):
        out = self.root / "report.json"
        out.write_text("keep", encoding="utf-8")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv)
        self.assertEqual(out.read_text(), "keep")

    def test_commands_only_receive_their_own_flags(self):
        args = bench.arguments(self.argv + ["--tile-size", "8", "--tile-admission", "force"])
        for mode in bench.MODES:
            expected = ["patch_lab.exe", "in.png", "--output", "out.jxl", "--effort", "9",
                        "--threads", "8", "--mode",
                        "tiles" if mode in ("tiles", "offgrid") else
                        "source" if mode in ("source", "source-parallel") or mode in bench.SOURCE_BLOCKS else "off"]
            if mode in ("tiles", "offgrid"):
                expected += ["--tile-admission", "force", "--tile-size", "8"]
            if mode == "offgrid":
                expected += ["--off-grid"]
            if mode in ("block8", "block16"):
                expected += ["--block-size", mode[5:]]
            if mode in bench.SOURCE_BLOCKS:
                target, size = bench.SOURCE_BLOCKS[mode]
                expected += ["--block-size", str(size), "--block-target", target]
            if mode in bench.PARALLEL_MODES:
                expected += ["--parallel-gather"]
            with self.subTest(mode=mode):
                self.assertEqual(bench.command(args, "in.png", "out.jxl", mode), expected)
        args.tile_size = None
        self.assertNotIn("--tile-size", bench.command(args, "i", "o", "offgrid"))
        with self.assertRaises(ValueError):
            bench.command(args, "i", "o", "components")

    def test_manifest_relative_paths_and_no_skipping(self):
        source = self.root / "sample.png"
        source.write_bytes(fixture())
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(cases=[dict(original=source.name)])), encoding="utf-8")
        args = bench.arguments(self.argv + ["--manifest", str(manifest)])
        self.assertEqual(bench.load_cases(args), [source.resolve()])
        args.case = [source]
        with self.assertRaises(ValueError):
            bench.load_cases(args)
        args.case = []
        for value in ({"cases": []}, {"cases": [{}]}, {"cases": [{"original": "missing.png"}]}):
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(value=value), self.assertRaises((ValueError, FileNotFoundError)):
                bench.load_cases(args)


class ReceiptTests(unittest.TestCase):
    def test_explicit_planning_budget_is_forwarded_and_verified(self):
        args = bench.arguments(["--encoder", "lab", "--decoder", "djxl", "--synthetic",
                                "--out", "unused-qlic-report.json", "--block-leaves", "65536"])
        self.assertNotIn("--block-leaves", bench.command(args, "i", "o", "baseline"))
        for mode in ("block8", "block16", *bench.SOURCE_BLOCKS):
            with self.subTest(mode=mode):
                self.assertIn("65536", bench.command(args, "i", "o", mode))
                m = measurement(mode)
                with self.assertRaisesRegex(ValueError, "planning budget"):
                    bench.validate_measurement(m, mode, m["bytes"], m["width"], m["height"], block_leaves=65536)
                m["block_plan_leaves"] = 65536
                bench.validate_measurement(m, mode, m["bytes"], m["width"], m["height"], block_leaves=65536)

    def validate(self, m, mode):
        bench.validate_measurement(m, mode, 13, 69, 53)

    def test_every_mode_and_source_gate(self):
        for mode in bench.MODES:
            with self.subTest(mode=mode):
                self.validate(measurement(mode), mode)
        for mode in ("source", "source-parallel", *bench.SOURCE_BLOCKS):
            self.validate(measurement(mode, source_enabled=False), mode)
        for mode in ("tiles", "offgrid"):
            self.validate(measurement(mode, planned=False, repeated=0), mode)
            self.validate(measurement(mode, planned=False, rejected=True), mode)
        # Constant/palette shortcuts can bypass MA learning entirely.
        m = measurement("block8")
        m["block_work"] = dict.fromkeys(bench.BLOCK_COUNTERS, 0)
        m["block_work"]["predictor_leaves"] = [0, 0, 0, 0]
        self.validate(m, "block8")

    def test_source_targets_are_explicit_and_independently_validated(self):
        for mode, (target, _) in bench.SOURCE_BLOCKS.items():
            for value in (None, "wrong", "residual" if target != "residual" else "both"):
                m = measurement(mode)
                m["block_target"] = value
                with self.subTest(mode=mode, target=value), self.assertRaises(ValueError):
                    self.validate(m, mode)
            for key in ("block_target",):
                m = measurement(mode)
                del m[key]
                with self.assertRaises(ValueError):
                    self.validate(m, mode)
            for value in (None, {}, block_work()):
                m = measurement(mode)
                m["block_work"] = value
                with self.subTest(mode=mode, work=value), self.assertRaises(ValueError):
                    self.validate(m, mode)
            for part in ("reference", "residual"):
                requested = target in (part, "both")
                for value in (None, {}, block_work()):
                    if not requested and value is None:
                        continue
                    m = measurement(mode)
                    m["block_work"][part] = value
                    with self.subTest(mode=mode, part=part, work=value), self.assertRaises(ValueError):
                        self.validate(m, mode)
                if not requested:
                    continue
                for key, value in (("trees", -1), ("predictor_leaves", [0] * 4),
                                   ("used", False), ("used", 1), ("configured", False),
                                   ("configured", 1)):
                    m = measurement(mode)
                    m["block_work"][part][key] = value
                    with self.subTest(mode=mode, part=part, key=key), self.assertRaises(ValueError):
                        self.validate(m, mode)
                # Configuration does not prove use: palettes/constants/no atlas can bypass it.
                m = measurement(mode)
                m["block_work"][part] = dict(block_work(used=False), configured=True, used=False)
                self.validate(m, mode)
                for key in ("used", "configured"):
                    m = measurement(mode)
                    del m["block_work"][part][key]
                    with self.assertRaises(ValueError):
                        self.validate(m, mode)
            if target in ("reference", "both"):
                m = measurement(mode, source_enabled=False)
                m["block_work"]["reference"] = dict(block_work(), configured=False, used=True)
                with self.assertRaisesRegex(ValueError, "actual usage"):
                    self.validate(m, mode)

    def test_new_target_field_on_legacy_receipts(self):
        for mode in ("baseline", "source", "block8", "block16"):
            m = measurement(mode)
            m["block_target"] = bench.block_settings(mode)[0]
            self.validate(m, mode)
            m["block_target"] = "reference"
            with self.assertRaises(ValueError):
                self.validate(m, mode)

    def test_parallel_gather_activation_is_explicit_and_does_not_leak(self):
        self.assertEqual(set(bench.PARALLEL_MODES), {
            "baseline-parallel", "source-parallel", "source-ref16-parallel",
            "source-res16-parallel", "source-both16-parallel",
        })
        for mode in bench.MODES:
            for value in (None, 0, 1, "true", mode not in bench.PARALLEL_MODES):
                m = measurement(mode)
                m["parallel_gather"] = value
                with self.subTest(mode=mode, value=value), self.assertRaises(ValueError):
                    self.validate(m, mode)
            m = measurement(mode)
            if mode in bench.PARALLEL_MODES:
                del m["parallel_gather"]
                with self.assertRaises(ValueError):
                    self.validate(m, mode)
            else:
                m["parallel_gather"] = False
                self.validate(m, mode)

    def test_receipts_required_and_inactive_receipts_null(self):
        for mode in bench.MODES:
            for key in ("block_size", "block_work", "source_detector_enabled", "tile_work", "guard_work"):
                m = measurement(mode)
                del m[key]
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    self.validate(m, mode)
            m = measurement(mode)
            m["block_size"] = 16 if bench.block_settings(mode)[1] == 8 else 8
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.validate(m, mode)
        for key, value in (("block_work", {}), ("source_detector_enabled", False),
                           ("tile_work", {}), ("guard_work", {}), ("tile_admission", "screened")):
            m = measurement()
            m[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(m, "baseline")

    def test_wrong_mode_and_offgrid_receipts_rejected(self):
        for mode in bench.MODES:
            m = measurement(mode)
            m["mode"] = "source" if m["mode"] != "source" else "off"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.validate(m, mode)
        for mode in ("tiles", "offgrid"):
            for key, value in (("off_grid", mode != "offgrid"), ("off_grid", 1),
                               ("tile_size", 16.0), ("windows_examined", -1),
                               ("windows_examined", True), ("discovery_limited", 0.0)):
                m = measurement(mode)
                m["tile_work"][key] = value
                with self.subTest(mode=mode, key=key, value=value), self.assertRaises(ValueError):
                    self.validate(m, mode)
            for key in ("off_grid", "windows_examined", "discovery_limited"):
                m = measurement(mode)
                del m["tile_work"][key]
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    self.validate(m, mode)
        # Discovery limits do not invent a new note or imply a rejected plan.
        m = measurement("offgrid", planned=False, repeated=0)
        m["tile_work"]["discovery_limited"] = 1
        self.validate(m, "offgrid")

    def test_block_counter_types_and_accounting(self):
        for key in bench.BLOCK_COUNTERS:
            for value in (True, -1, 0.5, None):
                m = measurement("block16")
                m["block_work"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(m, "block16")
            m = measurement("block16")
            del m["block_work"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(m, "block16")
        for updates in (dict(predictor_leaves=[1, 2, 1]), dict(predictor_leaves=[1, 2, 1, True]),
                        dict(predictor_leaves=[1, 2, 1, -1]), dict(predictor_leaves=[0, 0, 0, 0]),
                        dict(predictor_leaves=None), dict(leaves=6), dict(trees=0),
                        dict(channel_splits=2), dict(coarsened_views=7)):
            m = measurement("block16")
            m["block_work"].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.validate(m, "block16")
        for value in (None, {}, []):
            m = measurement("block8")
            m["block_work"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(m, "block8")

    def test_existing_timing_size_selection_contract_preserved(self):
        for updates in (dict(bytes=True), dict(bytes=12), dict(seconds=0),
                        dict(seconds=float("nan")), dict(seconds=float("inf")),
                        dict(width=69.0), dict(height=54), dict(bpp=9),
                        dict(mpix_per_second=9), dict(selected_mode="source")):
            m = measurement()
            m.update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.validate(m, "baseline")


class FixtureTests(unittest.TestCase):
    def test_formats_shifted_copies_mixed_directions_and_hidden_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = bench.fixtures(Path(temp) / "fixtures")
            self.assertEqual(len(paths), 15)
            formats = set()
            for path in paths:
                data = path.read_bytes()
                _, info = bench.port.png_payload(data)
                formats.add((info["bit_depth"], info["color_type"]))
                if not path.name.startswith(("shifted-tiles", "mixed-direction")):
                    continue
                pixels = bench.port.png_samples(data)
                rgba = pixels.rgba
                hidden = rgba[:, :, 3] == 0
                self.assertTrue(hidden.any())
                self.assertTrue((rgba[:, :, :3][hidden] != 0).any())
                if pixels.depth == 16:
                    self.assertEqual(str(rgba.dtype), "uint16")
                    self.assertTrue((rgba[:, :, :3] > 255).any())
                    self.assertTrue((rgba[:, :, :3] % 257 != 0).any())
                if path.name.startswith("shifted"):
                    seed = rgba[:16, :16].tobytes()
                    for x, y in ((37, 5), (73, 41), (17, 79)):
                        self.assertEqual(seed, rgba[y:y + 16, x:x + 16].tobytes())
                    self.assertGreater(len(set(rgba[:16, :16, 0].flat)), 64)
                else:
                    self.assertEqual(rgba[1, 2:20, :3].tobytes(), rgba[2, 2:20, :3].tobytes())
                    self.assertEqual(rgba[2:20, 74, :3].tobytes(), rgba[2:20, 75, :3].tobytes())
            self.assertEqual(formats, {(d, c) for d in (8, 16) for c in (0, 2, 4, 6)})


class HarnessTests(unittest.TestCase):
    def test_packed_png_uses_original_payload_and_expanded_oracle(self):
        from test_corpus_inputs import png
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "packed.png"
            original = png(2, 3, [[0, 1, 2, 3]], 4,
                           [(b"PLTE", bytes([7, 11, 13, 23, 29, 31, 41, 43, 47, 59, 61, 67])),
                            (b"tRNS", bytes([0, 255, 127, 255]))])
            source.write_bytes(original)
            payload, expected, label = bench.prepare_png(source, original, root)
            self.assertEqual(payload, original)
            self.assertEqual(payload[24], 2)
            self.assertEqual(expected.depth, 8)
            self.assertEqual(expected.rgba[0, 0].tolist(), [7, 11, 13, 0])
            self.assertIn("oracle-expanded", label)

    def exercise(self, depth=16, failure=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source, encoder, decoder = (root / name for name in ("input.png", "patch_lab.exe", "djxl.exe"))
        samples = [259, 513, 65534, 0, 1, 257, 32769, 65535] if depth == 16 else [19, 81, 253, 0, 7, 11, 13, 255]
        source.write_bytes(make_png(depth, 6, [samples], extra=[(b"tEXt", b"key\x00value")]))
        encoder.write_bytes(b"encoder")
        decoder.write_bytes(b"decoder")
        out = root / "report.json"
        argv = ["--encoder", str(encoder), "--decoder", str(decoder), "--case", str(source),
                "--repeats", "3", "--warmups", "1", "--out", str(out)]
        calls, visits = [], []

        def run(command, env, timeout):
            calls.append(command)
            self.assertNotIn("RS_JXL_PROFILE", env)
            if command[0] == str(encoder):
                self.assertEqual(Path(command[1]).read_bytes(), fixture(depth))
                self.assertNotIn("--profile", command)
                mode = command[command.index("--mode") + 1]
                mode = "baseline" if mode == "off" else mode
                if "--block-size" in command:
                    size = command[command.index("--block-size") + 1]
                    if "--block-target" in command:
                        target = command[command.index("--block-target") + 1]
                        mode = "source-" + {"reference": "ref", "residual": "res", "both": "both"}[target] + size
                    else:
                        mode = "block" + size
                elif "--off-grid" in command:
                    mode = "offgrid"
                if "--parallel-gather" in command:
                    mode += "-parallel"
                visits.append(mode)
                size = 12 + len(visits)
                Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                m = measurement(mode, width=2, height=1, size=size,
                                seconds=float(len(visits)), planned=False, repeated=0)
                if failure == "receipt":
                    del m["block_work"]
                if failure == "nan":
                    m["seconds"] = float("nan")
                return dict(command=command, returncode=1 if failure == "encoder" else 0,
                            stdout=json.dumps(m), stderr="", wall_seconds=999)
            self.assertIn(f"--bits_per_sample={depth}", command)
            decoded = make_png(depth, 6, [[samples[0] ^ 1, *samples[1:]]]) if failure == "hidden" else fixture(depth)
            Path(command[2]).write_bytes(decoded)
            return dict(command=command, returncode=1 if failure == "decoder" else 0,
                        stdout="", stderr="", wall_seconds=999)

        with patch.object(bench.port, "run_process", side_effect=run), \
                patch.dict(os.environ, {"RS_JXL_PROFILE": "1"}), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(bench.main(argv), int(failure is not None))
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(report["temporary_outputs_cleaned"])
        self.assertEqual({p.name for p in root.iterdir()}, {source.name, encoder.name, decoder.name, out.name})
        return report, calls, visits

    def test_all_modes_round_robin_warmups_native_medians_aggregate(self):
        for depth in (8, 16):
            with self.subTest(depth=depth):
                report, calls, visits = self.exercise(depth)
                expected = [mode for repeat in (-1, 0, 1, 2)
                            for mode in bench.trial_order(bench.MODES, repeat, 0)]
                self.assertEqual(visits, expected)
                self.assertEqual(len(calls), 8 * len(bench.MODES))
                self.assertEqual(report["status"], "passed")
                case = report["cases"][0]
                self.assertEqual(case["preparation"]["removed_metadata_chunks"], ["tEXt"])
                for mode in bench.MODES:
                    summary = case["summaries"][mode]
                    self.assertLess(summary["seconds"], 999)
                    self.assertEqual(summary["seconds"], sorted(
                        s["measurement"]["seconds"] for s in case["variants"][mode] if not s["warmup"])[1])
                    self.assertEqual(report["aggregate"][mode]["bytes"], summary["bytes"])
                    self.assertAlmostEqual(summary["bpp"], summary["bytes"] * 4)
                    self.assertAlmostEqual(summary["mpix_per_second"], 2e-6 / summary["seconds"])

    def test_errors_stop_and_preserve_reports(self):
        for depth in (8, 16):
            for failure in ("hidden", "receipt", "encoder", "decoder", "nan"):
                with self.subTest(depth=depth, failure=failure):
                    report, calls, visits = self.exercise(depth, failure)
                    self.assertEqual(report["status"], "failed")
                    self.assertTrue(report["errors"])
                    self.assertNotIn("aggregate", report)
                    self.assertEqual(len(visits), 1)
                    self.assertEqual(len(calls), 1 if failure == "encoder" else 2)
                    samples = [s for items in report["cases"][0]["variants"].values() for s in items]
                    self.assertEqual(len(samples), 1)
                    self.assertFalse(samples[0]["verified"])
                    if failure == "hidden":
                        self.assertIn("hidden RGB", report["errors"][0])

    def test_missing_executable_preserves_failure_report(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "report.json"
            args = bench.arguments(["--encoder", str(Path(temp) / "missing"), "--decoder", "djxl.exe",
                                    "--synthetic", "--out", str(out)])
            with patch.object(bench.port, "run_process") as run:
                report = bench.benchmark(args)
            run.assert_not_called()
            self.assertEqual(report["status"], "failed")
            self.assertTrue(json.loads(out.read_text())["errors"])


if __name__ == "__main__":
    unittest.main()
