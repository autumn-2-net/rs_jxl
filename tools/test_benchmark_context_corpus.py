"""Bounded tests: subprocesses/sample boundary mocked; no encoder is executed."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
from io import StringIO
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_context_corpus as bench
from test_compare_cpp_port import fixture, make_png
from test_corpus_inputs import png as packed_png


def measurement(point, case, size=20, seconds=.01, threads=8):
    mode = point["mode"]
    pixels = case["width"] * case["height"]
    fast, seed, seed_options, offsets, refinement, lookahead, lookahead_options = (None,) * 7
    if mode in bench.FAST_MODES:
        full = mode.startswith("full14")
        fast = dict(mode=mode if mode in ("activity25", "activity81") else "auto12",
                    bias="static" if full else "off", predictors="full" if full else "basic",
                    scan="auto", cross_channel=False, **dict.fromkeys(bench.FAST_COUNTERS, 0))
    if mode in bench.SEED_MODES:
        topology, method = mode.split("-")
        seed_options = dict(topology=topology, mode=method, depth=3)
        seed = dict.fromkeys(bench.SEED_COUNTERS, 0)
    if mode in bench.OFFSET_MODES:
        offsets = dict(mode="distribution", samples_per_leaf=4096)
        refinement = dict(offset_mode="distribution", offset_proposals=0, offset_seconds=0.)
    if mode in ("lookahead2", "lookahead4"):
        lookahead_options = dict(width=int(mode[-1]), min_records=8192, relative_window=.01)
        lookahead = dict(roots=0, eligible=0, probed_candidates=0, child_searches=0, changed=0,
                         estimated_saved_bits=0.)
    patch_options = dict(legacy=False, source=False, source_reference=False, tiles=False)
    result = dict(bytes=size, seconds=seconds, bpp=size * 8 / pixels,
                  mpix_per_second=pixels / 1e6 / seconds, width=case["width"], height=case["height"],
                  mode=mode, effort=point["effort"], threads=threads, patches="Auto", previous_channels=0,
                  search=False, native_encode_calls=1, timing=bench.NATIVE_TIMING,
                  fast_context=fast, seed=seed, seed_options=seed_options, model_costs=None,
                  model_cost_contract=bench.MODEL_COST_CONTRACT, offset_options=offsets,
                  refinement=refinement, lookahead_options=lookahead_options, lookahead=lookahead,
                  lookahead_score_contract="estimated local scoring only; not emitted wire bits",
                  patch_options=patch_options)
    result["options_signature"] = dict(schema="static_context_lab.options.v1",
        **{k: result[k] for k in ("mode", "effort", "threads", "patches", "previous_channels")},
        patch_options=patch_options, fast_context=None if fast is None else {
            k: fast[k] for k in ("mode", "bias", "predictors", "scan", "cross_channel")},
        seed=seed_options, offsets=offsets, lookahead=lookahead_options)
    return result


def tiny_identity(widths=(2,), modes=("auto12",), repeats=1):
    return dict(schema=bench.SCHEMA, config=dict(threads=8, efforts=[1, 9], experiment_modes=list(modes),
                repeats=repeats, max_cases=200, timeout=3600., patches="auto", serial=True,
                backend="direct native; no outer portfolio"),
                binaries={name: dict(path=f"/{name}.exe") for name in ("encoder", "cjxl", "decoder")},
                inputs=[dict(path=f"/case-{i}.png", sha256=str(i) * 64, width=w, height=1)
                        for i, w in enumerate(widths)])


def verified_row(identity, index, point, repetition=0, size=20, seconds=.01):
    case = identity["inputs"][index]
    process = dict(returncode=0, stdout="", stderr="", wall_seconds=1.,
                   command=bench.command(identity, point, "/snapshot.png", "/output.jxl"))
    if point["family"] == "cjxl":
        process["stderr"] = f"{case['width']} x {case['height']}, 1.000 MP/s, 8 threads.\n"
        m = bench.parse_cjxl_stats(process["stderr"], case["width"], case["height"], 8)
    else:
        m = measurement(point, case, size, seconds)
    decoder = dict(returncode=0, stdout="", stderr="", wall_seconds=1.,
                   command=bench.port.decoder_command(identity["binaries"]["decoder"]["path"],
                                                       "/output.jxl", "/oracle.png", 16))
    return bench.seal_row(dict(fingerprint=bench.fingerprint(identity), case_index=index,
        variant_id=point["id"], repetition=repetition, source_sha256=case["sha256"],
        verified=True, status="verified", bytes=size, wall_seconds=1., measurement=m,
        encoder_process=process, decoder_process=decoder,
        verification=dict(original_samples_equal=True, includes_hidden_rgb=True, depth=16,
                          jxl_sha256="a" * 64, decoded_png_sha256="b" * 64, snapshot_sha256="c" * 64)))


def complete_report(widths=(2, 10), modes=("auto12",), repeats=1):
    identity = tiny_identity(widths, modes, repeats)
    rows = [verified_row(identity, i, point, repetition=r, size=20 + i, seconds=.01 * (i + 1))
            for i in range(len(widths)) for point in bench.variants(identity["config"]) for r in range(repeats)]
    return bench.make_report(identity, bench.fingerprint(identity), rows)


class ParsingTests(unittest.TestCase):
    def test_default_matrix_and_limits(self):
        argv = ["--manifest", "sample.json", "--encoder", "lab.exe", "--cjxl", "cjxl.exe",
                "--decoder", "djxl.exe", "--outdir", "result"]
        args = bench.arguments(argv)
        self.assertEqual((args.threads, args.repeats, args.max_cases), (8, 1, 200))
        self.assertGreaterEqual(args.timeout, 1200)
        points = bench.variants(vars(args))
        self.assertEqual(len(points), 30)
        self.assertEqual(len(points) * 200, 6000)
        self.assertEqual([p["effort"] for p in points[:9]], list(range(1, 10)))
        self.assertTrue(all(p["effort"] == 9 for p in points[18:]))
        self.assertFalse(any("rct" in p["mode"] for p in points))
        self.assertEqual(bench.arguments(argv + ["--max-cases", "201"]).max_cases, 201)
        for flags in (["--efforts", "10"], ["--efforts", "1", "1"], ["--threads", "0"],
                      ["--repeats", "0"], ["--timeout", "nan"], ["--max-cases", "0"],
                      ["--experiment-modes", "auto12", "auto12"], ["--experiment-modes", "rct-joint"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                bench.arguments(argv + flags)

    def test_exact_final_stats_and_inverse_rate_quantization(self):
        stderr = ("Read 8100x11708 image, 1000 bytes, 123.4 MP/s\n"
                  "Compressed to 12.3 kB (0.001 bpp).\n8100 x 11708, 8.123 MP/s, 8 threads.\n")
        m = bench.parse_cjxl_stats(stderr, 8100, 11708, 8)
        self.assertAlmostEqual(m["seconds"], 8100 * 11708 / 1e6 / 8.123)
        self.assertLess(m["native_seconds_lower"], m["seconds"])
        self.assertGreater(m["native_seconds_upper"], m["seconds"])
        self.assertAlmostEqual(m["rate_lower"], 8.1225)
        self.assertNotIn("bytes", m)
        self.assertEqual(bench.parse_cjxl_stats(stderr.replace("\n", "\r\n"), 8100, 11708, 8), m)

    def test_reject_zero_nonfinite_ambiguous_or_nonfinal_rate(self):
        good = "2 x 1, 1.000 MP/s, 8 threads.\n"
        for text in ("Read 2x1 image, 22 bytes, 1.0 MP/s\n", good + good,
                     good + "2 x 1, nan MP/s, 8 threads.\n", good.replace("1.000", "0.000"),
                     good.replace("1.000", "nan"), good.replace("1.000", "inf"),
                     good.replace("1.000", "1.00"), good.replace("8 threads", "4 threads"),
                     good.replace("2 x 1", "3 x 1"), good.replace("MP/s", "MP/s, 2 reps")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_cjxl_stats(text, 2, 1, 8)

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.assertRaises(ValueError):
                bench.loads(text)

    def test_commands_auto_patches_and_rotated_serial_order(self):
        identity = tiny_identity(modes=bench.EXPERIMENT_MODES)
        identity["binaries"] = {name: dict(path=name + ".exe") for name in ("encoder", "cjxl", "decoder")}
        points = bench.variants(identity["config"])
        for point in points:
            cmd = bench.command(identity, point, Path("input.png"), Path("output.jxl"))
            if point["family"] == "cjxl":
                self.assertEqual(cmd[-8:], ["-d", "0", "-e", str(point["effort"]), "--num_threads=8",
                                          "--lossless_jpeg=0", "--keep_invisible=1", "--num_reps=1"])
                self.assertNotIn("--patches=0", cmd)
            else:
                self.assertEqual(cmd[-2:], ["--patches", "auto"])
        self.assertEqual(bench.trial_order(points, 1, 0), points[1:] + points[:1])

    def test_cjxl_short_option_values_are_separate_for_all_nine_efforts(self):
        identity = tiny_identity()
        identity["config"]["efforts"] = list(range(1, 10))
        for point in bench.variants(identity["config"]):
            if point["family"] != "cjxl":
                continue
            cmd = bench.command(identity, point, "input.png", "output.jxl")
            self.assertEqual(cmd[3:7], ["-d", "0", "-e", str(point["effort"])])
            self.assertNotIn("-d0", cmd)
            self.assertNotIn(f"-e{point['effort']}", cmd)


class ReceiptTests(unittest.TestCase):
    def test_lookahead_accepts_native_f32_window_and_keeps_signature_exact(self):
        case = dict(width=2, height=1)
        native_window = struct.unpack("<f", struct.pack("<f", .01))[0]
        self.assertEqual(native_window, 0.009999999776482582)
        for mode in ("lookahead2", "lookahead4"):
            with self.subTest(mode=mode):
                point = dict(mode=mode, effort=9)
                m = measurement(point, case)
                m["lookahead_options"]["relative_window"] = native_window
                m["options_signature"]["lookahead"] = copy.deepcopy(m["lookahead_options"])
                bench.validate_native(m, point, 20, case, 8, 1.)
                self.assertEqual(m["lookahead_options"]["relative_window"], native_window)
                m["options_signature"]["lookahead"]["relative_window"] = .01
                with self.assertRaisesRegex(ValueError, "signature mismatch"):
                    bench.validate_native(m, point, 20, case, 8, 1.)

    def test_lookahead_rejects_invalid_keys_exact_fields_and_nonfinite_window(self):
        case, point = dict(width=2, height=1), dict(mode="lookahead2", effort=9)
        valid = dict(width=2, min_records=8192, relative_window=.01)
        invalid = [dict(valid, width=4), dict(valid, width=2.), dict(valid, width=True),
                   dict(valid, min_records=8191), dict(valid, min_records=8192.),
                   dict(valid, extra=True), dict(width=2, min_records=8192)]
        invalid += [dict(valid, relative_window=value)
                    for value in (float("nan"), float("inf"), float("-inf"), .011, 0, True, ".01")]
        for options in invalid:
            with self.subTest(options=options):
                m = measurement(point, case)
                m["lookahead_options"] = options
                m["options_signature"]["lookahead"] = copy.deepcopy(options)
                with self.assertRaisesRegex(ValueError, "Lookahead experiment identity mismatch"):
                    bench.validate_native(m, point, 20, case, 8, 1.)

    def test_all_current_modes_and_efforts(self):
        identity = tiny_identity(modes=bench.EXPERIMENT_MODES)
        for point in bench.variants(identity["config"]):
            if point["family"] != "cjxl":
                case = identity["inputs"][0]
                bench.validate_native(measurement(point, case), point, 20, case, 8, 1.)

    def test_reject_wrong_receipts_and_experiment_identities(self):
        case = dict(width=2, height=1)
        for mode, field, value in (
            ("baseline", "patches", "Off"), ("baseline", "native_encode_calls", True),
            ("baseline", "native_encode_calls", 2), ("baseline", "search", True),
            ("baseline", "seconds", 0), ("baseline", "bytes", 21), ("baseline", "width", 3),
            ("baseline", "bpp", 1), ("baseline", "effort", 8), ("baseline", "fast_context", {}),
            ("auto12", "fast_context", None), ("activity81-refine", "seed_options", {}),
            ("offset-distribution", "offset_options", dict(mode="mean", samples_per_leaf=100)),
            ("lookahead2", "lookahead_options", dict(width=4, min_records=8192, relative_window=.01)),
            ("full14-static", "options_signature", {})):
            point = dict(mode=mode, effort=9)
            m = measurement(point, case)
            m[field] = value
            with self.subTest(mode=mode, field=field), self.assertRaises(ValueError):
                bench.validate_native(m, point, 20, case, 8, 1.)
        point = dict(mode="full14-static", effort=9)
        m = measurement(point, case)
        m["fast_context"]["predictors"] = "basic"
        with self.assertRaises(ValueError):
            bench.validate_native(m, point, 20, case, 8, 1.)

    def test_arithmetic_mean_not_pixel_weighted_and_native_medians(self):
        identity = tiny_identity((10, 100), repeats=3)
        rows = [verified_row(identity, i, point, repetition=r, size=(10, 20)[i], seconds=(1., 2., 90.)[r])
                for i in range(2) for point in bench.variants(identity["config"]) for r in range(3)]
        images = bench.per_image(identity, rows)
        total = bench.aggregate(images)["rust-e1"]
        self.assertAlmostEqual(total["mean_bpp"], (8 + 1.6) / 2)
        self.assertAlmostEqual(total["pixel_weighted_bpp"], 30 * 8 / 110)
        self.assertEqual(total["native_seconds"], 4.)
        self.assertEqual(total["summed_bytes"], 30)
        self.assertEqual(total["summed_mpix"], 110 / 1e6)

    def test_partial_report_does_not_drop_failures_or_claim_aggregate(self):
        identity = tiny_identity((2, 10))
        point = bench.variants(identity["config"])[0]
        row = verified_row(identity, 0, point)
        report = bench.make_report(identity, bench.fingerprint(identity), [row])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["verified_trials"], 1)
        self.assertEqual(report["missing_trials"], 9)
        self.assertNotIn("aggregate", report)
        self.assertEqual(len(report["case_statuses"]), 2)
        self.assertEqual(bench.make_report(identity, bench.fingerprint(identity), [], integrity=False)["status"],
                         "incomplete")


class HarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "tools").mkdir()
        (self.root / "crates").mkdir()
        self.harness_source = self.root / "tools" / "benchmark_context_corpus.py"
        self.harness_source.write_text("# frozen test helper\n", encoding="utf-8")
        (self.root / "tools" / "plot_context_corpus.py").write_text("# frozen plot helper\n", encoding="utf-8")
        self.native_source = self.root / "crates" / "source.rs"
        self.native_source.write_text("// frozen native source\n", encoding="utf-8")
        for name in ("lab", "cjxl", "djxl"):
            (self.root / (name + ".exe")).write_bytes(name.encode())
        self.source = self.root / "rgba.png"
        self.source.write_bytes(fixture(16))
        self.manifest = self.root / "sample.json"
        self.write_manifest()
        self.argv = ["--manifest", str(self.manifest), "--encoder", str(self.root / "lab.exe"),
                     "--cjxl", str(self.root / "cjxl.exe"), "--decoder", str(self.root / "djxl.exe"),
                     "--outdir", str(self.root / "out"), "--efforts", "1", "--experiment-modes", "auto12"]
        self.fail_mode = None
        self.hidden_mismatch = False
        self.interrupt_after = None
        self.encode_calls = 0
        self.encoded = {}
        self.calls = []

    def write_manifest(self):
        self.manifest.write_text(json.dumps(dict(cases=[dict(original=self.source.name,
            sha256=bench.port.file_sha256(self.source), width=2, height=1)])), encoding="utf-8")

    def fake_process(self, command, env, timeout):
        self.calls.append((list(command), env.copy()))
        process = dict(returncode=0, stdout="", stderr="", command=list(command), wall_seconds=1.)
        if "--version" in command:
            process["stdout"] = "cjxl v0.12.0 a1ebdf2 [_AVX2_,SSE4,SSE2] Clang 17\n"
        elif "--help" in command:
            process["stdout"] = "--num_reps --keep_invisible --lossless_jpeg --num_threads --effort"
        elif any(str(item).startswith("--bits_per_sample=") for item in command):
            data = self.encoded[str(command[1])]
            if self.hidden_mismatch:
                data = make_png(16, 6, [[260, 513, 65534, 0, 1, 257, 32769, 65535]])
            Path(command[2]).write_bytes(data)
        else:
            if any(item == "-d0" or (item.startswith("-e") and item != "-e") for item in command):
                raise AssertionError("Official cjxl rejects attached short-option values")
            if self.interrupt_after is not None and self.encode_calls >= self.interrupt_after:
                raise KeyboardInterrupt("mock interruption")
            self.encode_calls += 1
            mode = command[command.index("--mode") + 1] if "--mode" in command else "official"
            if mode == self.fail_mode:
                process.update(returncode=1, stderr="mock native failure")
                return process
            output = Path(command[command.index("--output") + 1] if "--output" in command else command[2])
            data = Path(command[1]).read_bytes()
            self.encoded[str(output)] = data
            output.write_bytes(b"j" * 20)
            pixels = bench.port.png_samples(data)
            if mode == "official":
                process["stderr"] = f"Compressed to 999.9 kB\n{pixels.width} x {pixels.height}, 1.000 MP/s, 8 threads.\n"
            else:
                effort = int(command[command.index("--effort") + 1])
                m = measurement(dict(mode=mode, effort=effort), dict(width=pixels.width, height=pixels.height))
                process["stdout"] = json.dumps(m)
        return process

    def run_harness(self, flags=()):
        def mock_samples(data):
            _, info = bench.port.png_payload(data)
            return SimpleNamespace(width=info["width"], height=info["height"], depth=info["bit_depth"],
                                   original=data)

        def mock_compare(expected, actual):
            if expected.original != actual.original:
                raise ValueError("Mock original sample mismatch; hidden RGB is included")

        with patch.object(bench, "__file__", str(self.harness_source)), \
             patch.object(bench, "helper_sources", return_value=[self.harness_source]), \
             patch.object(bench.port, "run_process", side_effect=self.fake_process), \
             patch.object(bench.port, "pixel_libraries", return_value=(None, None)), \
             patch.object(bench.port, "png_samples", side_effect=mock_samples), \
             patch.object(bench.port, "compare_pixels", side_effect=mock_compare), \
             redirect_stdout(StringIO()):
            return bench.benchmark(bench.arguments(self.argv + list(flags)))

    def test_frozen_cohort_actual_file_bytes_clean_environment_and_resume(self):
        with patch.dict(bench.os.environ, {"RS_JXL_ROUTE": "bad", "JXL_TEST": "bad"}):
            report = self.run_harness()
        self.assertEqual((report["status"], report["verified_trials"]), ("complete", 3))
        self.assertEqual(self.encode_calls, 3)
        self.assertTrue(all(r["bytes"] == 20 for r in report["trials"]))
        self.assertTrue(all(r["verification"]["depth"] == 16 for r in report["trials"]))
        self.assertTrue(all("RS_JXL_ROUTE" not in env and "JXL_TEST" not in env for _, env in self.calls))
        outdir = self.root / "out"
        self.assertFalse(any(outdir.glob("context-corpus-*")))
        self.assertEqual(len((outdir / "trials.jsonl").read_text().splitlines()), 3)
        self.assertEqual(bench.loads((outdir / "progress.json").read_text())["verified_trials"], 3)
        resumed = self.run_harness(["--resume"])
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(self.encode_calls, 3)

    def test_failure_checkpoint_retried_without_reusing_unverified(self):
        self.fail_mode = "auto12"
        report = self.run_harness()
        self.assertEqual((report["status"], report["failed_trials"], report["missing_trials"]), ("incomplete", 1, 0))
        self.assertNotIn("aggregate", report)
        self.fail_mode = None
        report = self.run_harness(["--resume"])
        self.assertEqual((report["status"], report["attempts"], self.encode_calls), ("complete", 4, 4))
        self.assertEqual(len((self.root / "out" / "trials.jsonl").read_text().splitlines()), 4)

    def test_interruption_partial_report_torn_tail_and_recovery(self):
        self.interrupt_after = 1
        report = self.run_harness()
        self.assertEqual((report["status"], report["verified_trials"], report["missing_trials"]), ("incomplete", 1, 2))
        journal = self.root / "out" / "trials.jsonl"
        with journal.open("ab") as stream:
            stream.write(b'{"partial":')
        self.interrupt_after = None
        report = self.run_harness(["--resume"])
        self.assertEqual((report["status"], self.encode_calls), ("complete", 3))
        self.assertEqual(len(list(journal.parent.glob("trials.jsonl.torn-*.bin"))), 1)

    def test_hidden_rgb_mismatch_fails_every_output(self):
        self.hidden_mismatch = True
        report = self.run_harness()
        self.assertEqual((report["status"], report["failed_trials"]), ("incomplete", 3))
        self.assertTrue(all("hidden RGB" in r["error"] for r in report["trials"]))

    def test_refuse_config_binary_manifest_input_or_source_change_on_resume(self):
        self.run_harness()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_harness(["--resume", "--threads", "4"])
        for path in (self.root / "cjxl.exe", self.manifest, self.native_source):
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "identity changed"):
                self.run_harness(["--resume"])
            path.write_bytes(before)
        self.source.write_bytes(fixture(8))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_harness(["--resume"])

    def test_modified_receipt_checksum_rejected_before_processes(self):
        self.run_harness()
        journal = self.root / "out" / "trials.jsonl"
        lines = journal.read_text().splitlines()
        row = json.loads(lines[0])
        row["bytes"] += 1
        lines[0] = json.dumps(row)
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        previous = len(self.calls)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_harness(["--resume"])
        self.assertEqual(len(self.calls), previous)

    def test_no_filtering_no_resize_and_manifest_hash_validation(self):
        self.manifest.write_text(json.dumps(dict(cases=[dict(original=self.source.name, sha256="f" * 64)])),
                                 encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_harness()
        self.assertEqual(self.encode_calls, 0)

    def test_all_twelve_experiments_have_verified_independent_rows(self):
        report = self.run_harness(["--experiment-modes", *bench.EXPERIMENT_MODES])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["verified_trials"], 14)
        modes = {p["mode"] for p in report["variants"] if p["family"] == "experiment"}
        self.assertEqual(modes, set(bench.EXPERIMENT_MODES))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "Canonical packed oracle requires OpenCV")
    def test_packed_preparation_keeps_native_input_unchanged(self):
        # The canonical helper handles unpacking only for the independent oracle.
        data = packed_png(2, 0, [[0, 1, 2, 3]], width=4)
        self.source.write_bytes(data)
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            payload, expected, _ = bench.prepare_png(self.source, data, Path(folder))
        self.assertEqual(payload, data)
        self.assertEqual((expected.width, expected.height, expected.depth), (4, 1, 8))


if __name__ == "__main__":
    unittest.main()
