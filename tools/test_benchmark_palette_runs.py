"""Palette sidecar contracts; no builds or actual encoder/decoder subprocesses."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import benchmark_palette_runs as bench
from test_compare_cpp_port import fixture


def measurement(args, mode="default", size=13, width=2, height=1, route="full_image", palettes=None):
    native, hot = bench.mode_config(mode)
    order = {"default": "Default", "frequency": "FrequencyLuma", "runs": "RunFrequency",
             "screened": "ScreenedRuns", "adjacency": "Adjacency"}.get(native)
    rounds = None
    if hot is not None:
        order = f"HotRuns {{ hot_colors: {hot} }}"
    if native in ("adjacency-minla", "ezeng", "ezeng-minla"):
        rounds = 0 if native == "ezeng" else (4 if args.minla_rounds is None else args.minla_rounds)
        variant = "AdjacencyMinLa" if native == "adjacency-minla" else "EzengMinLa"
        order = f"{variant} {{ rounds: {rounds} }}"
    palettes = [] if palettes is None else palettes
    return dict(bytes=size, seconds=0.1, width=width, height=height, profile=None,
                bpp=size * 8 / (width * height), mpix_per_second=width * height / 1e6 / 0.1,
                mode=native, effort=args.effort, threads=args.threads, encode_calls=1, patches_cleared=True,
                configured=dict(order=order, hot_colors=hot, minla_rounds=rounds, groupwise=not args.full_frame,
                                palette_colors=args.palette_colors if args.palette_colors is not None else 1024),
                projected=dict(groupwise=True, palette_colors=1024),
                overrides=dict(palette_colors=args.palette_colors, full_frame=args.full_frame),
                palette_evidence=dict(scope=bench.SCOPE, route=route, input_width=width, input_height=height,
                                      input_channels=4, spatial_channels=1 if palettes else 4,
                                      meta_channels=len(palettes), global_tuple_palette_count=len(palettes),
                                      global_tuple_palettes=palettes))


class ArgumentsAndReceipts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.argv = ["--encoder", "palette_lab.exe", "--decoder", "djxl.exe", "--synthetic",
                     "--out", str(self.root / "report.json")]
        self.args = bench.arguments(self.argv)

    def validate(self, m, mode="default", args=None):
        bench.validate_measurement(m, mode, 13, 2, 1, args or self.args)

    def test_defaults_and_all_mode_commands(self):
        self.assertEqual(self.args.modes, ["default", "runs", "hot0", "hot128"])
        self.assertEqual((self.args.effort, self.args.threads, self.args.repeats), (9, 8, 3))
        self.assertFalse(self.args.full_frame)
        self.assertFalse(self.args.profile)
        self.assertTrue(bench.arguments(self.argv + ["--profile"]).profile)
        self.assertIsNone(self.args.palette_colors)
        self.assertIsNone(self.args.minla_rounds)
        for mode in bench.MODES:
            cmd = bench.command(self.args, "in.png", "out.jxl", mode)
            native, hot = bench.mode_config(mode)
            self.assertEqual(cmd[:6], ["palette_lab.exe", "in.png", "--output", "out.jxl", "--mode", native])
            self.assertNotIn("--full-frame", cmd)
            self.assertNotIn("--palette-colors", cmd)
            self.assertNotIn("--minla-rounds", cmd)
            if hot is not None:
                self.assertEqual(cmd[-2:], ["--hot-colors", str(hot)])
            else:
                self.assertNotIn("--hot-colors", cmd)
        args = bench.arguments(self.argv + ["--palette-colors", "10000", "--full-frame", "--modes", *bench.MODES])
        for mode in bench.MODES:
            self.assertEqual(bench.command(args, "i", "o", mode)[-3:], ["--palette-colors", "10000", "--full-frame"])
            self.validate(measurement(args, mode), mode, args)
        with self.assertRaises(ValueError):
            bench.mode_config("searchbest")

    def test_graph_modes_rounds_and_exact_debug_orders(self):
        for rounds in (None, 0, 4, 40):
            argv = self.argv + ["--modes", *bench.MODES]
            if rounds is not None:
                argv += ["--minla-rounds", str(rounds)]
            args = bench.arguments(argv)
            expected = {
                "adjacency": ("Adjacency", None),
                "adjacency-minla": (f"AdjacencyMinLa {{ rounds: {4 if rounds is None else rounds} }}",
                                    4 if rounds is None else rounds),
                "ezeng": ("EzengMinLa { rounds: 0 }", 0),
                "ezeng-minla": (f"EzengMinLa {{ rounds: {4 if rounds is None else rounds} }}",
                                4 if rounds is None else rounds),
            }
            for mode in bench.MODES:
                with self.subTest(rounds=rounds, mode=mode):
                    cmd = bench.command(args, "i", "o", mode)
                    if rounds is not None and mode in bench.MINLA_MODES:
                        self.assertEqual(cmd[-2:], ["--minla-rounds", str(rounds)])
                    else:
                        self.assertNotIn("--minla-rounds", cmd)
                    m = measurement(args, mode)
                    self.validate(m, mode, args)
                    if mode not in expected:
                        continue
                    self.assertEqual(bench.mode_config(mode), (mode, None))
                    order, count = expected[mode]
                    self.assertEqual((m["configured"]["order"], m["configured"]["minla_rounds"]),
                                     (order, count))
                    for bad in ("Default", "AdjacencyMinla { rounds: 4 }", "EzengMinla { rounds: 0 }"):
                        changed = deepcopy(m)
                        changed["configured"]["order"] = bad
                        with self.assertRaisesRegex(ValueError, "Configured order"):
                            self.validate(changed, mode, args)
                    for bad in (True, 0.0, -1, 41, "4"):
                        changed = deepcopy(m)
                        changed["configured"]["minla_rounds"] = bad
                        with self.assertRaisesRegex(ValueError, "Configured MinLA rounds"):
                            self.validate(changed, mode, args)

    def test_screened_order_and_nullable_profile(self):
        args = bench.arguments(self.argv + ["--modes", "default", "screened", "--profile"])
        self.assertEqual(bench.mode_config("screened"), ("screened", None))
        m = measurement(args, "screened")
        self.assertEqual(m["configured"]["order"], "ScreenedRuns")
        self.validate(m, "screened", args)
        m["profile"] = "palette.order.screen | 1 | 0.250000 | 0.250000\n"
        self.validate(m, "screened", args)
        m["configured"]["order"] = "RunFrequency"
        with self.assertRaisesRegex(ValueError, "Configured order"):
            self.validate(m, "screened", args)

    def test_bad_args_and_overwrite(self):
        for extra in (["--effort", "3"], ["--effort", "11"], ["--threads", "0"], ["--threads", "257"],
                      ["--repeats", "0"], ["--repeats", "101"], ["--warmups", "-1"], ["--warmups", "101"],
                      ["--timeout", "nan"], ["--timeout", "0"], ["--palette-colors", "-1"],
                      ["--palette-colors", "70912"], ["--modes", "runs"], ["--modes", "default", "default"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv + extra)
        for extra in (["--minla-rounds", "4"],
                      ["--modes", "default", "adjacency", "ezeng", "--minla-rounds", "0"],
                      ["--modes", "default", "adjacency-minla", "--minla-rounds", "-1"],
                      ["--modes", "default", "ezeng-minla", "--minla-rounds", "41"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv + extra)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments([a for a in self.argv if a != "--synthetic"])
        self.args.out.write_text("keep", encoding="utf-8")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv)
        with self.assertRaises(FileExistsError):
            bench.benchmark(self.args)
        self.assertEqual(self.args.out.read_text(), "keep")

    def test_actual_route_not_configured_activation(self):
        for mode in bench.MODES:
            for route in ("full_image", "groupwise"):
                self.validate(measurement(self.args, mode, route=route), mode)
            palette = dict(input_channels=[0, 4], colors=48, deltas=0, predictor="Zero")
            self.validate(measurement(self.args, mode, palettes=[palette]), mode)
            with self.assertRaises(ValueError):
                self.validate(measurement(self.args, mode, route="groupwise", palettes=[palette]), mode)

    def test_graph_configuration_does_not_claim_activation(self):
        for mode in ("adjacency", "adjacency-minla", "ezeng", "ezeng-minla"):
            for channels, colors, deltas in ((4, 48, 0), (3, 256, 0), (4, 257, 0),
                                             (4, 300, 0), (2, 48, 0), (4, 0, 8)):
                with self.subTest(mode=mode, channels=channels, colors=colors, deltas=deltas):
                    palette = dict(input_channels=[0, channels], colors=colors,
                                   deltas=deltas, predictor="Zero")
                    m = measurement(self.args, mode, palettes=[palette])
                    # Accepted tuples, including bypass cases, do not attest graph execution.
                    self.validate(m, mode)
                    self.assertIsNone(m["profile"])
                    self.assertNotIn("graph_activated", m["palette_evidence"])

    def test_invalid_receipts(self):
        mutations = [lambda m: m.update(seconds=0), lambda m: m.update(seconds=float("nan")),
                     lambda m: m.pop("profile"), lambda m: m.update(profile={}),
                     lambda m: m.update(profile=1),
                     lambda m: m.update(bytes=True), lambda m: m.update(bytes=12),
                     lambda m: m.update(width=3), lambda m: m.update(bpp=0),
                     lambda m: m.update(mpix_per_second=1), lambda m: m.update(mode="runs"),
                     lambda m: m.update(encode_calls=2), lambda m: m.update(encode_calls=True),
                     lambda m: m.update(patches_cleared=False), lambda m: m.update(threads=1),
                     lambda m: m.pop("palette_evidence"), lambda m: m.update(palette_evidence=None),
                     lambda m: m["configured"].update(order="RunFrequency"),
                     lambda m: m["configured"].update(hot_colors=0),
                     lambda m: m["configured"].update(groupwise=False),
                     lambda m: m["configured"].update(palette_colors=10000),
                     lambda m: m["overrides"].update(full_frame=True)]
        for key, value in (("scope", "all_palettes"), ("route", "unknown"), ("input_width", 1),
                           ("global_tuple_palette_count", 1), ("input_channels", True),
                           ("input_channels", 0), ("spatial_channels", -1), ("meta_channels", 0.5)):
            mutations.append(lambda m, k=key, v=value: m["palette_evidence"].update({k: v}))
        for mutate in mutations:
            m = measurement(self.args)
            mutate(m)
            with self.subTest(m=m), self.assertRaises(ValueError):
                self.validate(m)
        for update in (dict(input_channels=[0, 1]), dict(input_channels=[0, 5]), dict(colors=-1),
                       dict(deltas=True), dict(predictor=""), dict(input_channels=[0, True])):
            palette = dict(input_channels=[0, 4], colors=48, deltas=0, predictor="Zero")
            palette.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.validate(measurement(self.args, palettes=[palette]))

    def test_manifest_loader_and_aggregate_helpers(self):
        source = self.root / "sample.png"
        source.write_bytes(fixture())
        manifest = self.root / "cases.json"
        manifest.write_text(json.dumps({"cases": [{"original": source.name}]}), encoding="utf-8")
        self.args.manifest = manifest
        self.assertEqual(bench.load_cases(self.args), [source.resolve()])
        cases = [dict(pixels=100, summaries=dict(default=dict(bytes=10, seconds=2), runs=dict(bytes=5, seconds=3))),
                 dict(pixels=300, summaries=dict(default=dict(bytes=90, seconds=8), runs=dict(bytes=105, seconds=9)))]
        result = bench.aggregate(cases, ["default", "runs"])
        self.assertEqual(result["runs"]["ratios_to_baseline"]["bytes"], 1.1)
        self.assertEqual(result["runs"]["regressions"], dict(bytes=True, seconds=True))
        self.assertNotIn("baseline", result)


class Fixtures(unittest.TestCase):
    def test_packed_png_keeps_native_input_and_checks_expanded_oracle(self):
        from test_corpus_inputs import png

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "packed.png"
            scratch = root / "scratch"
            scratch.mkdir()
            for depth in (1, 2, 4):
                original = png(depth, 3, [[0, 1, 0]], 3,
                               [(b"PLTE", bytes([19, 81, 253, 7, 11, 13])), (b"tRNS", b"\0")],
                               interlace=True)
                source.write_bytes(original)
                payload, expected, preparation = bench.prepare_png(source, original, scratch)
                self.assertEqual(payload, original)
                self.assertEqual(payload[24], depth)
                self.assertEqual(expected.depth, 8)
                self.assertEqual(expected.rgba.tolist(),
                                 [[[19, 81, 253, 0], [7, 11, 13, 255], [19, 81, 253, 0]]])
                self.assertIn("original-packed", preparation)
                self.assertEqual(list(scratch.iterdir()), [])

    def test_formats_runs_and_hidden_rgb(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = bench.fixtures(Path(temp) / "fixtures")
            self.assertEqual(len(paths), 6)
            formats = set()
            for path in paths:
                data = path.read_bytes()
                _, header = bench.port.png_payload(data)
                p = bench.port.png_samples(data)
                formats.add((p.depth, header["color_type"]))
                if header["color_type"] == 0:
                    continue
                counts, runs = {}, {}
                previous = None
                for row in p.rgba:
                    for pixel in row:
                        color = tuple(int(v) for v in pixel)
                        counts[color] = counts.get(color, 0) + 1
                        if previous != color:
                            runs[color] = runs.get(color, 0) + 1
                        previous = color
                if path.stem.endswith("-300"):
                    self.assertEqual((p.width, p.height), (257, 65))
                    exact_tuples = {tuple(int(v) for v in pixel) for row in p.rgba for pixel in row}
                    self.assertEqual(len(exact_tuples), 300)
                    self.assertEqual(len(counts), 300)
                    self.assertGreater(len(set(runs.values())), 2)
                else:
                    self.assertEqual(len(counts), 48)
                most_pixels = max(counts, key=counts.get)
                self.assertLess(runs[most_pixels], max(runs.values()))
                if p.depth == 16:
                    self.assertTrue((p.rgba[:, :, :3] > 255).any())
                    self.assertTrue((p.rgba[:, :, :3] % 257 != 0).any())
                if header["color_type"] == 6:
                    hidden = p.rgba[:, :, 3] == 0
                    self.assertTrue(hidden.any())
                    self.assertTrue((p.rgba[:, :, :3][hidden] != 0).any())
                    changed = deepcopy(p)
                    changed.rgba[0, 0, 0] ^= 1
                    with self.assertRaisesRegex(ValueError, "hidden RGB"):
                        bench.port.compare_pixels(p, changed)
            self.assertEqual(formats, {(8, 0), (8, 2), (8, 6), (16, 2), (16, 6)})


class Harness(unittest.TestCase):
    def test_profile_opt_in_and_retention_without_pixel_dependencies(self):
        for profile in (False, True):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, encoder, decoder = (root / n for n in ("in.png", "lab.exe", "djxl.exe"))
                source.write_bytes(fixture())
                encoder.write_bytes(b"encoder")
                decoder.write_bytes(b"decoder")
                argv = ["--encoder", str(encoder), "--decoder", str(decoder), "--case", str(source),
                        "--out", str(root / "report.json"), "--repeats", "1", "--modes", "default", "screened",
                        "adjacency-minla", "ezeng", "ezeng-minla"]
                args = bench.arguments(argv + (["--profile"] if profile else []))
                snapshot = ("palette.order.graph | 1 | 0.040000 | 0.040000\n"
                            "palette.order.graph.build | 1 | 0.010000 | 0.010000\n"
                            "palette.order.graph.seed | 1 | 0.010000 | 0.010000\n"
                            "palette.order.graph.refine | 1 | 0.020000 | 0.020000\n") if profile else None
                notes = ("[rs-jxl profile] palette.graph.result seed=ezeng colors=48 samples=2 "
                         "edges=1 rounds=4 moves=2 before=100 after=80\n"
                         "[rs-jxl profile] palette.graph.bypass colors=300\n") if profile else ""

                def run(cmd, env, timeout):
                    self.assertEqual(env.get("RS_JXL_PROFILE"), "1" if profile else None)
                    self.assertNotIn("RS_JXL_PALETTE_ORDER", env)
                    stdout = ""
                    if cmd[0] == str(encoder):
                        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"x" * 13)
                        m = measurement(args, cmd[cmd.index("--mode") + 1])
                        m["profile"] = snapshot
                        stdout = json.dumps(m)
                    else:
                        Path(cmd[2]).write_bytes(b"mock decoded pixels")
                    return dict(command=cmd, returncode=0, stdout=stdout, stderr=notes, wall_seconds=0.2)

                with patch.object(bench.port, "run_process", side_effect=run), \
                        patch.object(bench.port, "png_samples", return_value=SimpleNamespace(width=2, height=1, depth=16)), \
                        patch.object(bench.port, "compare_pixels"), \
                        patch.dict(os.environ, {"RS_JXL_PROFILE": "trace", "RS_JXL_PALETTE_ORDER": "runs"}), \
                        redirect_stdout(StringIO()):
                    report = bench.benchmark(args)
                saved = json.loads(args.out.read_text())
                self.assertEqual(saved["status"], "passed", report["errors"])
                self.assertEqual(saved["environment"]["explicit"], {"RS_JXL_PROFILE": "1"} if profile else {})
                self.assertIn("not wall time", saved["profile_timing"])
                for samples in saved["cases"][0]["variants"].values():
                    self.assertEqual(samples[0]["measurement"]["profile"], snapshot)
                    self.assertEqual(samples[0]["measurement"]["seconds"], 0.1)
                    self.assertEqual(samples[0]["encoder"]["stderr"], notes)

    def test_rotations_oracle_failures_cleanup_and_work_receipts(self):
        for failure in (None, "encoder", "decoder", "receipt", "pixels", "timeout"):
            for diagnostic, profile in ((False, False), (True, False), (False, True)):
                with self.subTest(failure=failure, diagnostic=diagnostic, profile=profile), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    source, encoder, decoder = (root / n for n in ("input.png", "palette_lab.exe", "djxl.exe"))
                    source.write_bytes(fixture())
                    encoder.write_bytes(b"encoder")
                    decoder.write_bytes(b"decoder")
                    out = root / "report.json"
                    argv = ["--encoder", str(encoder), "--decoder", str(decoder), "--case", str(source),
                            "--out", str(out), "--repeats", "2", "--warmups", "1", "--modes", *bench.MODES]
                    if diagnostic:
                        argv += ["--full-frame", "--palette-colors", "10000", "--minla-rounds", "40"]
                    if profile:
                        argv += ["--profile"]
                    args = bench.arguments(argv)
                    visits, calls = [], []

                    def run(cmd, env, timeout):
                        calls.append(cmd)
                        self.assertEqual(env.get("RS_JXL_PROFILE"), "1" if profile else None)
                        self.assertNotIn("RS_JXL_PALETTE_ORDER", env)
                        if failure == "timeout":
                            raise TimeoutError("harness guard")
                        stdout, code = "", 0
                        if cmd[0] == str(encoder):
                            native = cmd[cmd.index("--mode") + 1]
                            mode = "hot" + cmd[cmd.index("--hot-colors") + 1] if native == "hot" else native
                            visits.append(mode)
                            # Deliberately different JXL bytes: never an identity comparison.
                            size = 12 + len(visits)
                            Path(cmd[cmd.index("--output") + 1]).write_bytes(bytes([len(visits)]) * size)
                            self.assertEqual("--full-frame" in cmd, diagnostic)
                            self.assertEqual("--minla-rounds" in cmd, diagnostic and mode in bench.MINLA_MODES)
                            m = measurement(args, mode, size=size, route="full_image" if diagnostic else "groupwise")
                            if profile:
                                # Inclusive totals can exceed the measured native wall duration.
                                m["profile"] = "palette.order.screen | 2 | 0.250000 | 0.150000\n"
                            if failure == "receipt":
                                m["encode_calls"] = 2
                            stdout = json.dumps(m)
                            code = int(failure == "encoder")
                        else:
                            Path(cmd[2]).write_bytes(fixture(8 if failure == "pixels" else 16))
                            code = int(failure == "decoder")
                        return dict(command=cmd, returncode=code, stdout=stdout,
                                    stderr="failed" if code else "[rs-jxl profile] palette decision: default" if profile else "",
                                    wall_seconds=0.2)

                    with patch.object(bench.port, "run_process", side_effect=run), \
                            patch.dict(os.environ, {"RS_JXL_PROFILE": "trace", "RS_JXL_PALETTE_ORDER": "runs"}), \
                            redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        report = bench.benchmark(args)
                    self.assertEqual(json.loads(out.read_text())["status"], report["status"])
                    self.assertTrue(report["temporary_outputs_cleaned"])
                    self.assertEqual({p.name for p in root.iterdir()}, {source.name, encoder.name, decoder.name, out.name})
                    if failure:
                        self.assertEqual(report["status"], "failed")
                        self.assertTrue(report["errors"])
                        self.assertNotIn("aggregate", report)
                        self.assertFalse(report["cases"][0]["variants"][bench.trial_order(bench.MODES, -1, 0)[0]][0]["verified"])
                    else:
                        self.assertEqual(report["status"], "passed")
                        self.assertIn("not wall time", report["profile_timing"])
                        self.assertEqual(report["environment"]["explicit"], {"RS_JXL_PROFILE": "1"} if profile else {})
                        self.assertEqual(visits, [m for r in (-1, 0, 1) for m in bench.trial_order(bench.MODES, r, 0)])
                        self.assertEqual(len(calls), 6 * len(bench.MODES))
                        for mode in bench.MODES:
                            self.assertEqual(len(report["cases"][0]["summaries"][mode]["sizes"]), 2)
                            self.assertEqual(len(report["cases"][0]["variants"][mode]), 3)
                            sample = report["cases"][0]["variants"][mode][0]
                            self.assertEqual(sample["measurement"]["profile"],
                                             "palette.order.screen | 2 | 0.250000 | 0.150000\n" if profile else None)
                            self.assertEqual(sample["encoder"]["stderr"],
                                             "[rs-jxl profile] palette decision: default" if profile else "")
                            self.assertEqual(sample["measurement"]["seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
