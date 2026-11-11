"""Palette adapter tests. All native processes are mocked; no timing runs."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

import benchmark_palette_groups as bench
import benchmark_fast_offsets as offsets
import benchmark_refinement as refinement
from benchmark_ma_models import trial_order
import compare_cpp_port as port


def measurement(variant, size=100, pixels=8192, colors=4096, applied=0):
    mode = bench.mode_for(variant)
    receipt = None if mode is None else dict(
        mode=mode, colors=colors, attempts=1, applied=applied,
        tables=applied * (2 if mode == "ya-cocg" else 1))
    return dict(bytes=size, seconds=0.1, bpp=size * 8 / pixels,
                mpix_per_second=pixels / 1e6 / 0.1, search=False,
                model_costs=dict(tree_bits=1, entropy_header_bits=1, total_bits=2),
                fast_context=None, refinement=None, ma_seed=None, palette_grouping=receipt)


def samples(data):
    """Read our unfiltered fixtures without an optional image-library dependency."""
    payload, info = port.png_payload(data)
    offset, compressed = 8, bytearray()
    while offset < len(payload):
        length, kind = struct.unpack_from(">I4s", payload, offset)
        if kind == b"IDAT":
            compressed.extend(payload[offset + 8:offset + 8 + length])
        offset += length + 12
    raw = zlib.decompress(compressed)
    channels = {0: 1, 4: 2, 2: 3, 6: 4}[info["color_type"]]
    width, height, depth = info["width"], info["height"], info["bit_depth"]
    stride = 1 + width * channels * (depth // 8)
    rgba = []
    for y in range(height):
        assert raw[y * stride] == 0
        values = struct.unpack(">" + ("B" if depth == 8 else "H") * width * channels,
                               raw[y * stride + 1:(y + 1) * stride])
        for x in range(width):
            pixel = tuple(values[x * channels:(x + 1) * channels])
            rgba.append(pixel if channels == 4 else pixel + ((1 << depth) - 1,))
    return port.Pixels(width, height, depth, rgba)


class PaletteGroupsTests(unittest.TestCase):
    def argv(self, folder):
        return ["--encoder", str(folder / "encoder"), "--official-djxl", str(folder / "djxl"),
                "--synthetic", "--out", str(folder / "report.json")]

    def test_arguments_cap_variants_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            args = bench.arguments(self.argv(folder))
            self.assertEqual(args.variants, ["baseline", "group-rg", "group-yco", "group-ycg",
                                             "group-cocg", "group-ya-cocg"])
            self.assertEqual((args.effort, args.buffering, args.palette_group_colors), (9, 0, 4096))
            for cap in (2, 17, 70911):
                self.assertEqual(bench.arguments(self.argv(folder) + ["--palette-group-colors", str(cap)]
                                                ).palette_group_colors, cap)
            for flag, value in (("--palette-group-colors", "-1"), ("--palette-group-colors", "0"),
                                ("--palette-group-colors", "1"), ("--palette-group-colors", "70912"),
                                ("--palette-group-colors", "1.5"), ("--effort", "8"),
                                ("--threads", "0"), ("--repeats", "0"), ("--warmups", "-1"),
                                ("--timeout", "nan")):
                with self.subTest(flag=flag, value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    bench.arguments(self.argv(folder) + [flag, value])
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments([v for v in self.argv(folder) if v != "--synthetic"])
            args.out.write_text("preserve", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv(folder))
            with patch.object(bench, "make_runner", return_value=bench.make_runner()), self.assertRaises(FileExistsError):
                bench.benchmark(args)
            self.assertEqual(args.out.read_text(encoding="utf-8"), "preserve")

    def test_commands_same_explicit_cap_fullframe_and_no_other_experiments(self):
        with TemporaryDirectory() as directory:
            args = bench.arguments(self.argv(Path(directory)) + ["--palette-group-colors", "37"])
            base = refinement.command(args, "in.png", "out.jxl", "baseline")
            for variant in bench.VARIANTS:
                mode = bench.mode_for(variant)
                command = bench.command(args, "in.png", "out.jxl", variant)
                self.assertEqual(command, base + ["--palette-group-colors", "37"]
                                 + ([] if mode is None else ["--palette-grouping", mode]))
                self.assertEqual(command[command.index("--buffering") + 1], "0")
                for flag in ("--fast-context", "--palette-colors", "--search", "--force-palette",
                             "--refine-leaves", "--ma-seed", "--family", "--leaf-offset"):
                    self.assertNotIn(flag, command)
            with self.assertRaises(ValueError):
                bench.command(args, "in", "out", "unknown")

    def test_receipt_applied_declined_and_corruption(self):
        for variant in bench.VARIANTS:
            bench.validate_receipt(measurement(variant), variant)
        for variant in bench.VARIANTS[1:]:
            bench.validate_receipt(measurement(variant, applied=1, colors=37), variant, 37)
            ineligible = measurement(variant)
            ineligible["palette_grouping"]["attempts"] = 0
            bench.validate_receipt(ineligible, variant)
            for key, value in (("mode", "wrong"), ("colors", 37), ("colors", True),
                               ("attempts", -1), ("attempts", True), ("applied", 0.5),
                               ("applied", 2), ("tables", -1), ("tables", True), ("tables", 1)):
                data = measurement(variant)
                data["palette_grouping"][key] = value
                with self.subTest(variant=variant, key=key, value=value), self.assertRaises(ValueError):
                    bench.validate_receipt(data, variant)
            for key in ("mode", "colors", "attempts", "applied", "tables"):
                data = measurement(variant)
                del data["palette_grouping"][key]
                with self.assertRaises(ValueError):
                    bench.validate_receipt(data, variant)
            for key, value in (("palette_grouping", None), ("refinement", {}),
                               ("fast_context", {}), ("ma_seed", {})):
                with self.assertRaises(ValueError):
                    bench.validate_receipt(dict(measurement(variant), **{key: value}), variant)
        for receipt in ({}, measurement("group-rg")["palette_grouping"]):
            with self.assertRaises(ValueError):
                bench.validate_receipt(dict(measurement("baseline"), palette_grouping=receipt), "baseline")
        data = measurement("baseline")
        del data["palette_grouping"]
        with self.assertRaises(ValueError):
            bench.validate_receipt(data, "baseline")

    def test_measurement_and_pairing_allow_growth(self):
        for changes in (dict(bytes=99), dict(seconds=0), dict(bpp=1),
                        dict(mpix_per_second=1), dict(search=True)):
            with self.assertRaises(ValueError):
                bench.validate_measurement(dict(measurement("baseline"), **changes), "baseline", "auto", 100, 8192)
        variants = {v: [dict(repetition=r, measurement=dict(bytes=50 if v == "baseline" else 900))
                        for r in (-1, 0, 1)] for v in bench.VARIANTS}
        variants["group-rg"].reverse()
        bench.validate_sizes(variants)
        variants["baseline"][0]["repetition"] = 0
        with self.assertRaisesRegex(ValueError, "paired repetitions"):
            bench.validate_sizes(variants)
        variants["baseline"][0]["repetition"] = -1
        variants["group-yco"].pop()
        with self.assertRaisesRegex(ValueError, "paired repetitions"):
            bench.validate_sizes(variants)

    def test_synthetics_pair_cardinality_hidden_rgb_and_16bit(self):
        with TemporaryDirectory() as directory:
            paths = bench.fixtures(Path(directory))
            self.assertEqual(len(paths), 26)
            for path in paths:
                if not path.name.startswith("pair-"):
                    continue
                image = samples(path.read_bytes())
                mode = path.name.removeprefix("pair-").split(f"-{image.depth}bit")[0]
                first, second = set(), set()
                for r, g, b, a in image.rgba:
                    co = r - b
                    t = b + (co >> 1)
                    cg = g - t
                    y = t + (cg >> 1)
                    first.add({"rg": (r, g), "yco": (y, co), "ycg": (y, cg),
                               "cocg": (co, cg), "ya-cocg": (co, cg)}[mode])
                    second.add((y, a))
                self.assertLessEqual(len(first), 128)
                if mode == "ya-cocg":
                    self.assertLessEqual(len(second), 128)
                self.assertGreater(len(set(image.rgba)), 4096)
                if "c4" in path.name:
                    self.assertGreater(len({p[:3] for p in image.rgba if p[3] == 0}), 100)
                if image.depth == 16:
                    self.assertGreater(max(p[0] for p in image.rgba), 255)
                    self.assertTrue(any(p[0] % 257 for p in image.rgba))

    def test_metadata_neutral_preparation_preserves_original_oracle(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            path = next(p for p in bench.fixtures(folder) if p.name == "pair-ya-cocg-16bit-c4.png")
            plain = path.read_bytes()
            original = plain[:33] + bench._chunk(b"tEXt", b"Comment\x00metadata") + plain[33:]
            path.write_bytes(original)
            expected = samples(original)
            runner = bench.make_runner()
            with patch.object(port, "png_samples", side_effect=samples) as oracle:
                payload, actual, _ = runner.prepare_png(path, original, folder)
            oracle.assert_called_once_with(original)
            self.assertEqual(payload, plain)
            self.assertEqual(actual, expected)
            self.assertEqual(path.read_bytes(), original)

    def test_private_runner_hooks_and_main_reuse(self):
        runner = bench.make_runner()
        self.assertIs(runner.command, bench.command)
        self.assertIs(runner.validate_sizes, bench.validate_sizes)
        self.assertIs(runner.validate_receipt, bench.validate_receipt)
        self.assertIs(runner.fixtures, bench.fixtures)
        self.assertIs(runner.prepare_png, refinement.prepare_png)
        self.assertIsNot(refinement.command, bench.command)
        self.assertIsNot(offsets.make_runner().command, bench.command)
        with TemporaryDirectory() as directory:
            for status in ("passed", "failed"):
                with (patch.object(bench, "benchmark", return_value=dict(status=status, errors=[])) as run,
                      redirect_stdout(StringIO())):
                    self.assertEqual(bench.main(self.argv(Path(directory))), int(status != "passed"))
                self.assertEqual(run.call_args.args[0].variants, list(bench.VARIANTS))

    def test_mocked_runner_original_pixels_cap_rotation_and_failures(self):
        for depth in (8, 16):
            for failure in (None, "pixels", "receipt", "process", "identity"):
                with self.subTest(depth=depth, failure=failure), TemporaryDirectory() as directory:
                    folder = Path(directory)
                    source = next(p for p in bench.fixtures(folder) if p.name == f"pair-ya-cocg-{depth}bit-c4.png")
                    original = source.read_bytes()
                    for name in ("encoder", "djxl"):
                        (folder / name).write_bytes(name.encode("ascii"))
                    argv = [v for v in self.argv(folder) if v != "--synthetic"]
                    args = bench.arguments(argv + ["--case", str(source), "--repeats", "2",
                                                   "--warmups", "1", "--palette-group-colors", "37"])
                    runner, calls = bench.make_runner(), []

                    def process(command, env, timeout):
                        if command[0] == str(args.encoder):
                            self.assertEqual(Path(command[1]).read_bytes(), original)
                            self.assertEqual(command[command.index("--palette-group-colors") + 1], "37")
                            mode = command[command.index("--palette-grouping") + 1] if "--palette-grouping" in command else None
                            variant = "baseline" if mode is None else f"group-{mode}"
                            calls.append(variant)
                            size = 50 if mode is None else 150
                            Path(command[command.index("--output") + 1]).write_bytes(b"x" * size)
                            data = measurement(variant, size, 128 * 128, 37, int(mode is not None))
                            if failure == "receipt" and mode:
                                data["palette_grouping"]["colors"] = 4096
                            if failure == "identity":
                                args.encoder.write_bytes(b"changed")
                            return dict(returncode=int(failure == "process"), stdout=json.dumps(data),
                                        stderr="", wall_seconds=0.2)
                        self.assertEqual(command[0], str(args.decoder))
                        self.assertEqual(command, port.decoder_command(args.decoder, Path(command[1]),
                                                                 Path(command[2]), depth))
                        Path(command[2]).write_bytes(original)
                        return dict(returncode=0, stdout="", stderr="", wall_seconds=0.3)

                    def compare(left, right):
                        self.assertEqual(left, samples(original))
                        self.assertEqual(right, left)
                        if failure == "pixels":
                            raise ValueError("Original samples differ including hidden RGB")

                    with (patch.object(runner, "run_process", side_effect=process),
                          patch.object(port, "png_samples", side_effect=samples),
                          patch.object(runner, "png_samples", side_effect=samples),
                          patch.object(runner, "compare_pixels", side_effect=compare) as oracle,
                          patch.object(bench, "make_runner", return_value=runner), redirect_stdout(StringIO())):
                        report = bench.benchmark(args)
                    saved = json.loads(args.out.read_text(encoding="utf-8"))
                    self.assertEqual(report["status"], "passed" if failure is None else "failed", report["errors"])
                    self.assertEqual(saved["status"], report["status"])
                    self.assertEqual(source.read_bytes(), original)
                    if failure is None:
                        self.assertEqual(calls, sum((trial_order(list(bench.VARIANTS), r, 0) for r in (-1, 0, 1)), []))
                        self.assertEqual(oracle.call_count, 18)
                        case = saved["cases"][0]
                        self.assertEqual(case["depth"], depth)
                        self.assertEqual(case["summaries"]["group-ya-cocg"]["ratios_to_baseline"]["bytes"], 3)
                        self.assertEqual(case["variants"]["group-ya-cocg"][0]["measurement"]["palette_grouping"]["tables"], 2)
                    else:
                        self.assertTrue(saved["errors"])


if __name__ == "__main__":
    unittest.main()
