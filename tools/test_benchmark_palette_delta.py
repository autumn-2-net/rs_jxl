"""Palette delta contracts only; no builds or real encoder/decoder processes."""
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import benchmark_palette_delta as bench
import summarize_patch_ma_followup as followup
from test_compare_cpp_port import fixture

try:
    bench.port.pixel_libraries()
    HAVE_PIXELS = True
except RuntimeError:
    HAVE_PIXELS = False


def attempt(mode, scope="local", outcome="accepted", fallback=False, zero_matches=0):
    delta, zero = mode.startswith("exact-delta-"), mode.startswith("exact-zero-")
    produced = outcome != "kernel_declined"
    exact = delta and not fallback and produced
    deltas = 2 if exact else 1 if zero and not fallback and produced else 0
    return dict(scope=scope, requested_mode=bench.native_mode(mode), width=2, height=1, begin=0,
                channels=1 if fallback else 4, limit=1024,
                exact_delta_attempted=delta, exact_delta_produced=exact,
                exact_zero_attempted=zero, zero_mask_eligible=zero and not fallback,
                zero_mask_matches=zero_matches, absolute_attempted=not exact,
                absolute_produced=produced and not exact, colors=0 if exact or not produced else 2,
                deltas=deltas, predictor=mode.rsplit("-", 1)[1].title() if deltas else "Zero",
                delta_pixels=2 if exact else zero_matches if deltas else 0,
                cost_before=10, cost_after=11 if outcome == "cost_rejected" else None,
                cost_bypassed=outcome != "cost_rejected", outcome=outcome)


def measurement(args, mode="absolute", events=None, size=13, fingerprint="same native options"):
    return dict(mode=mode, effort=9, threads=args.threads, encode_calls=1, search=False,
                patches_cleared=True, width=2, height=1, bytes=size, seconds=0.1,
                bpp=size * 4, mpix_per_second=2e-5,
                configured=dict(mode=bench.native_mode(mode), groupwise=not args.full_frame,
                                palette_colors=args.palette_colors or 1024),
                projected=dict(groupwise=True, palette_colors=1024), invariant_options=fingerprint,
                overrides=dict(full_frame=args.full_frame, palette_colors=args.palette_colors),
                palette_evidence=dict(scope=bench.SCOPE, route="full_image" if args.full_frame else "groupwise",
                                      attempts=events or [], global_tuple_palettes=[], limitations="Preparation only"))


class Contracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.argv = ["--encoder", "lab.exe", "--decoder", "djxl.exe", "--synthetic",
                     "--out", str(self.root / "report.json")]
        self.args = bench.arguments(self.argv)

    def test_all_seven_modes_and_equal_controls(self):
        self.assertEqual(tuple(self.args.modes), bench.MODES)
        self.assertEqual(self.args.effort, 9)
        self.assertFalse(self.args.full_frame)
        self.assertIsNone(self.args.palette_colors)
        for cap in (1024, 4096, 10000):
            args = bench.arguments(self.argv + ["--full-frame", "--palette-colors", str(cap)])
            for mode in bench.MODES:
                cmd = bench.command(args, "input.png", "output.jxl", mode)
                self.assertEqual(cmd[-3:], ["--palette-colors", str(cap), "--full-frame"])
                self.assertNotIn("--force-palette", cmd)
                bench.validate_measurement(measurement(args, mode), mode, 13, 2, 1, args)

    def test_invalid_args_and_no_overwrite(self):
        for extra in (["--effort", "8"], ["--effort", "10"], ["--threads", "0"],
                      ["--threads", "257"], ["--palette-colors", "0"], ["--palette-colors", "70912"],
                      ["--timeout", "nan"], ["--timeout", "0"], ["--repeats", "0"],
                      ["--warmups", "-1"], ["--modes", "exact-delta-left"],
                      ["--modes", "absolute", "absolute"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(self.argv + extra)
        self.args.out.write_text("keep", encoding="utf-8")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench.arguments(self.argv)
        with self.assertRaises(FileExistsError):
            bench.benchmark(self.args)
        self.assertEqual(self.args.out.read_text(), "keep")

    def test_local_acceptance_declines_fallback_and_cost_rejection(self):
        for mode in bench.MODES:
            events = [attempt(mode), attempt(mode, outcome="cost_rejected"),
                      attempt(mode, outcome="kernel_declined"), attempt(mode, fallback=True)]
            m = measurement(self.args, mode, events)
            bench.validate_measurement(m, mode, 13, 2, 1, self.args)
            activation = bench.activation(m)
            self.assertEqual(activation["global"]["attempts"], 0)
            self.assertEqual(activation["local"]["attempts"], 4)
            self.assertEqual(activation["local"]["cost_rejected"], 1)
            self.assertEqual(activation["local"]["kernel_declined"], 1)
            if mode.startswith("exact-delta-"):
                self.assertEqual(activation["local"]["exact_delta_produced"], 2)
                self.assertEqual(activation["local"]["exact_delta_accepted"], 1)
                self.assertEqual(activation["local"]["absolute_fallback_accepted"], 1)
            if mode.startswith("exact-zero-"):
                self.assertEqual(activation["local"]["exact_zero_accepted"], 1)
                # An accepted zero delta column is not evidence that any pixel used it.
                self.assertEqual(activation["local"]["delta_used"], 0)
        empty = bench.activation(measurement(self.args, "exact-delta-left"))
        self.assertEqual(empty["local"]["exact_delta_attempted"], 0)
        # A geometry gate can decline a requested mode before calling any kernel.
        declined = attempt("exact-delta-left", outcome="kernel_declined")
        declined.update(exact_delta_attempted=False, absolute_attempted=False)
        bench.validate_attempt(declined, "exact-delta-left")

    def test_global_tuple_cross_check_and_full_frame_local_evidence(self):
        args = bench.arguments(self.argv + ["--full-frame", "--palette-colors", "1024"])
        mode = "exact-delta-gradient"
        m = measurement(args, mode, [attempt(mode, scope="global"), attempt(mode)])
        with self.assertRaisesRegex(ValueError, "contradicts"):
            bench.validate_measurement(m, mode, 13, 2, 1, args)
        m["palette_evidence"]["global_tuple_palettes"] = [dict(colors=0, deltas=2, predictor="Gradient", input_channels=[0, 4])]
        bench.validate_measurement(m, mode, 13, 2, 1, args)
        self.assertEqual(bench.activation(m)["local"]["delta_accepted"], 1)
        m["palette_evidence"]["route"] = "groupwise"
        with self.assertRaises(ValueError):
            bench.validate_measurement(m, mode, 13, 2, 1, args)

    def test_dishonest_evidence_and_timings_rejected(self):
        mode = "exact-delta-left"
        good = measurement(self.args, mode, [attempt(mode)])
        mutations = [lambda m: m.update(seconds=0), lambda m: m.update(seconds=float("nan")),
                     lambda m: m.update(bytes=True), lambda m: m.update(bpp=1),
                     lambda m: m.update(mpix_per_second=1), lambda m: m.update(encode_calls=True),
                     lambda m: m.update(search=True), lambda m: m.update(invariant_options=""),
                     lambda m: m["configured"].update(palette_colors=10000),
                     lambda m: m["palette_evidence"].update(scope="global_tuple_only")]
        for key, value in (("scope", "unknown"), ("channels", True), ("limit", -1),
                           ("exact_delta_produced", False), ("exact_delta_attempted", False),
                           ("delta_pixels", 0), ("absolute_attempted", True), ("cost_after", 11),
                           ("outcome", "cost_rejected"), ("deltas", 0), ("predictor", "Top")):
            mutations.append(lambda m, k=key, v=value: m["palette_evidence"]["attempts"][0].update({k: v}))
        for mutate in mutations:
            m = deepcopy(good)
            mutate(m)
            with self.subTest(m=m), self.assertRaises(ValueError):
                bench.validate_measurement(m, mode, 13, 2, 1, self.args)

    def test_regressions_are_results_not_failures(self):
        cases = [dict(pixels=2, summaries=dict(absolute=dict(bytes=10, seconds=1, bpp=40, mpix_per_second=2e-6),
                                             candidate=dict(bytes=15, seconds=2, bpp=60, mpix_per_second=1e-6)))]
        bench.add_comparisons(cases[0]["summaries"])
        self.assertEqual(cases[0]["summaries"]["candidate"]["regressions"], dict(bytes=True, seconds=True))
        result = bench.aggregate(cases, ["absolute", "candidate"])
        self.assertEqual(result["candidate"]["ratios_to_baseline"]["bytes"], 1.5)
        self.assertEqual(result["candidate"]["ratios_to_baseline"]["seconds"], 2)


class Fixtures(unittest.TestCase):
    def test_patterns_depth_hidden_rgb_and_low_bits(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = bench.fixtures(Path(temp) / "fixtures")
            self.assertEqual(len(paths), 18)
            for source in paths:
                payload = source.read_bytes()
                validated, info = bench.port.png_payload(payload)
                self.assertEqual(validated, payload)
                # These generated fixtures deliberately use filter 0 only.
                offset, idat = 8, bytearray()
                while offset < len(payload):
                    size, kind = struct.unpack_from(">I4s", payload, offset)
                    if kind == b"IDAT":
                        idat.extend(payload[offset+8:offset+8+size])
                    offset += size + 12
                raw = zlib.decompress(idat)
                depth, width, height = info["bit_depth"], info["width"], info["height"]
                stride = width * 4 * (depth // 8) + 1
                tuples = []
                for y in range(height):
                    self.assertEqual(raw[y * stride], 0)
                    values = struct.unpack(">" + ("B" if depth == 8 else "H") * width * 4,
                                           raw[y*stride+1:(y+1)*stride])
                    tuples.extend(tuple(values[x:x+4]) for x in range(0, len(values), 4))
                self.assertTrue(any(p[3] == 0 and any(p[:3]) for p in tuples))
                if "natural-like" in source.name:
                    self.assertGreater(len(set(tuples)), 1024)
                if depth == 16:
                    self.assertTrue(any(v % 257 != 0 for p in tuples for v in p[:3]))
                if "single-row" in source.name:
                    self.assertEqual(height, 1)

    @unittest.skipUnless(HAVE_PIXELS, "numpy/opencv not installed; parent oracle environment required")
    def test_source_features_and_independent_sample_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            for source in bench.fixtures(Path(temp) / "fixtures"):
                payload = source.read_bytes()
                pixels = bench.port.png_samples(payload)
                features = bench.content_features(pixels)
                self.assertLessEqual(features["sampled_pixels"], 65536)
                self.assertGreater(features["hidden_rgb_nonzero_fraction"], 0)
                if "natural-like" in source.name:
                    self.assertGreater(features["unique_tuples"], 1024)
                if pixels.depth == 16:
                    self.assertTrue((pixels.rgba[:, :, :3] % 257 != 0).any())
                if "single-row" in source.name:
                    self.assertEqual(pixels.height, 1)
                    self.assertFalse(features["residuals"])
                damaged = bench.port.png_samples(payload)
                damaged.rgba[0, 0, 0] ^= 1
                with self.assertRaisesRegex(ValueError, "hidden RGB"):
                    bench.port.compare_pixels(pixels, damaged)


class MockedRunner(unittest.TestCase):
    def test_rotations_unique_outputs_failure_retention_and_configuration(self):
        for failure in (None, "encoder", "decoder", "receipt", "pixels", "timeout", "invariants", "nonfinite"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, encoder, decoder = (root / name for name in ("input.png", "lab.exe", "djxl.exe"))
                source.write_bytes(fixture())
                encoder.write_bytes(b"encoder")
                decoder.write_bytes(b"decoder")
                args = bench.arguments(["--encoder", str(encoder), "--decoder", str(decoder), "--case", str(source),
                                        "--out", str(root / "report.json"), "--repeats", "2", "--warmups", "1",
                                        "--full-frame", "--palette-colors", "1024"])
                visits, outputs, decodes = [], [], []
                def run(cmd, env, timeout):
                    self.assertNotIn("RS_JXL_PROFILE", env)
                    self.assertNotIn("JXL_LOSSLESS_DELTA_PALETTE", env)
                    code, stdout = 0, ""
                    if cmd[0] == str(encoder):
                        mode = cmd[cmd.index("--mode") + 1]
                        visits.append(mode)
                        output = Path(cmd[cmd.index("--output") + 1])
                        self.assertFalse(output.exists())
                        outputs.append(output)
                        if failure == "timeout":
                            return dict(command=cmd, returncode=None, stdout="", stderr="", error="timeout", wall_seconds=10)
                        size = 13 if mode == "absolute" else 30
                        output.write_bytes(b"x" * size)
                        m = measurement(args, mode, [attempt(mode)], size=size)
                        if failure == "receipt":
                            m["encode_calls"] = 2
                        if failure == "invariants" and mode == "absolute":
                            m["invariant_options"] = "different group/RCT configuration"
                        stdout = json.dumps(m)
                        if failure == "nonfinite":
                            stdout = stdout.replace('"seconds": 0.1', '"seconds": NaN')
                        code = int(failure == "encoder")
                    else:
                        decodes.append(cmd)
                        Path(cmd[2]).write_bytes(fixture(8 if failure == "pixels" else 16))
                        code = int(failure == "decoder")
                    return dict(command=cmd, returncode=code, stdout=stdout, stderr="diagnostics retained", wall_seconds=0.2)
                expected = SimpleNamespace(width=2, height=1, depth=16)
                def compare(expected, actual):
                    if expected.depth != actual.depth:
                        raise ValueError("Decoded bit depth differs from source")
                with patch.object(bench.port, "run_process", side_effect=run), \
                        patch.object(bench, "prepare_png", return_value=(fixture(), expected, "mock preparation")), \
                        patch.object(bench, "content_features", return_value={"scope": "mock descriptor"}), \
                        patch.object(bench.port, "png_samples", side_effect=lambda data: SimpleNamespace(depth=data[24])), \
                        patch.object(bench.port, "compare_pixels", side_effect=compare), \
                        patch.dict(os.environ, {"RS_JXL_PROFILE": "1", "JXL_LOSSLESS_DELTA_PALETTE": "6"}):
                    report = bench.benchmark(args)
                saved = json.loads(args.out.read_text())
                self.assertEqual(saved["status"], "failed" if failure else "passed", report["errors"])
                self.assertEqual(visits, [m for repeat in (-1, 0, 1) for m in bench.trial_order(args.modes, repeat, 0)])
                self.assertEqual(len(outputs), len(set(outputs)))
                self.assertEqual(len(decodes), 0 if failure == "timeout" else len(outputs))
                self.assertTrue(report["temporary_outputs_cleaned"])
                self.assertTrue(all(not path.exists() for path in outputs))
                for samples in saved["cases"][0]["variants"].values():
                    self.assertEqual(len(samples), 3)
                    self.assertIn("diagnostics retained" if failure != "timeout" else "timeout",
                                  json.dumps(samples))
                if failure:
                    self.assertTrue(report["errors"])
                    self.assertNotIn("aggregate", report)
                else:
                    common = followup.summarize(saved, "absolute")
                    self.assertEqual(common["images"], 1)
                    self.assertEqual(common["overall"]["absolute"]["native_seconds"], 0.1)
                    summary = report["cases"][0]["summaries"]["exact-delta-left"]
                    self.assertEqual(summary["seconds"], 0.1)
                    self.assertTrue(summary["regressions"]["bytes"])
                    self.assertEqual(summary["ratios_to_baseline"]["seconds"], 1)
                    self.assertEqual(saved["cases"][0]["variants"]["exact-zero-left"][0]["activation"]["local"]["delta_used"], 0)


if __name__ == "__main__":
    unittest.main()
